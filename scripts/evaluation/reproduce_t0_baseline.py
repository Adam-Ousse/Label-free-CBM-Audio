#!/usr/bin/env python3
"""Reproduce global metrics from the canonical release CBM checkpoints.

This utility intentionally evaluates existing checkpoints only.  It does not
train, tune, regenerate concepts, or write into the release bundle.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_utils
from models.ast_classifier import build_ast_classifier
from models.ast_backbone import build_ast_backbone


DATASETS = {
    "esc50": {
        "model": "Adam-ousse/ast-esc50-finetuned-fold1",
        "backbone": "ast_esc50",
        "split": "fold1_test",
    },
    "urbansound8k": {
        "model": "Adam-ousse/ast-urbansound8k-finetuned-fold10",
        "backbone": "ast_urbansound8k",
        "split": "fold10_test",
    },
    "cremad": {
        "model": "Adam-ousse/ast-cremad-finetuned",
        "backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
        "split": "test",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-bundle", type=Path, required=True)
    parser.add_argument("--expected-comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance-pp", type=float, default=0.1)
    return parser.parse_args()


def load_expected(args: argparse.Namespace) -> tuple[dict, dict]:
    manifest = json.loads((args.artifact_bundle / "manifest.json").read_text(encoding="utf-8"))
    cbm_expected = {(item["dataset"], item["variant"]): item for item in manifest["models"]}
    with args.expected_comparison.open(newline="", encoding="utf-8") as handle:
        ast_expected = {
            (row["dataset"], row["model"]): row
            for row in csv.DictReader(handle)
            if row["model"] == "Fine-tuned AST"
        }
    return cbm_expected, ast_expected


def config_labels(classifier: torch.nn.Module) -> list[str]:
    return [
        str(classifier.id2label.get(index, classifier.id2label.get(str(index))))
        for index in range(classifier.num_labels)
    ]


def extract_features(
    dataset_name: str, args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = DATASETS[dataset_name]
    classes = data_utils.get_dataset_classes(dataset_name)
    classifier = build_ast_classifier(spec["model"], args.device)
    if classifier.num_labels != len(classes) or config_labels(classifier) != classes:
        raise ValueError(f"{dataset_name} AST checkpoint label order/count mismatch")
    classifier_head = copy.deepcopy(classifier.model.classifier).to(args.device).eval()
    del classifier
    torch.cuda.empty_cache()

    backbone = build_ast_backbone(spec["backbone"], args.device)
    backbone.eval()
    dataset = data_utils.get_audio_dataset(dataset_name, spec["split"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )

    features_all, logits_all, labels_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{dataset_name} global"):
            features = backbone(batch["audio"], sample_rates=batch["sr"])
            features_all.append(features.detach().cpu())
            logits_all.append(classifier_head(features).detach().cpu())
            labels_all.append(batch["target"].long().cpu())

    del backbone, classifier_head
    torch.cuda.empty_cache()
    return torch.cat(features_all), torch.cat(logits_all), torch.cat(labels_all)


def metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    true = labels.numpy()
    predicted = torch.argmax(logits, dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
    }


def compare(actual: float, expected: float, tolerance_pp: float) -> dict[str, float | bool]:
    delta_pp = (actual - expected) * 100.0
    return {
        "actual": actual,
        "expected": expected,
        "delta_pp": delta_pp,
        "pass": abs(delta_pp) <= tolerance_pp,
    }


def evaluate_dataset(dataset_name: str, args: argparse.Namespace, cbm_expected: dict, ast_expected: dict):
    features, ast_logits, labels = extract_features(dataset_name, args)
    ast_row = ast_expected[(dataset_name, "Fine-tuned AST")]
    ast_metrics = metrics(labels, ast_logits)
    rows = [
        {
            "dataset": dataset_name,
            "model_type": "fine_tuned_ast",
            "variant": "fine_tuned_ast",
            "checkpoint": DATASETS[dataset_name]["model"],
            "split": DATASETS[dataset_name]["split"],
            "num_samples": int(labels.numel()),
            "accuracy": ast_metrics["accuracy"],
            "macro_f1": ast_metrics["macro_f1"],
            "weighted_f1": ast_metrics["weighted_f1"],
            "balanced_accuracy": ast_metrics["balanced_accuracy"],
            "sparsity": None,
            "comparison": {
                "accuracy": compare(ast_metrics["accuracy"], float(ast_row["accuracy"]), args.tolerance_pp),
                "macro_f1": compare(ast_metrics["macro_f1"], float(ast_row["macro_f1"]), args.tolerance_pp),
            },
        }
    ]

    for variant in ("lf", "lf_broad", "lf_contrastive", "full"):
        expected = cbm_expected[(dataset_name, variant)]
        checkpoint = args.artifact_bundle / expected["checkpoint_dir"]
        tensors = {
            name: torch.load(checkpoint / f"{name}.pt", map_location="cpu").float()
            for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")
        }
        concepts = [
            line.strip()
            for line in (checkpoint / "concepts.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if tensors["W_c"].shape[0] != len(concepts) or tensors["W_g"].shape[1] != len(concepts):
            raise ValueError(f"{checkpoint}: concept count/order mismatch")
        projected = F.linear(features.float(), tensors["W_c"])
        projected = (projected - tensors["proj_mean"]) / tensors["proj_std"]
        cbm_logits = F.linear(projected, tensors["W_g"], tensors["b_g"])
        cbm_metrics = metrics(labels, cbm_logits)
        nonzero = int((tensors["W_g"].abs() > 1e-5).sum().item())
        total = int(tensors["W_g"].numel())
        sparsity = 1.0 - nonzero / float(total)
        rows.append(
            {
                "dataset": dataset_name,
                "model_type": "cbm",
                "variant": variant,
                "checkpoint": str(checkpoint),
                "split": DATASETS[dataset_name]["split"],
                "num_samples": int(labels.numel()),
                "accuracy": cbm_metrics["accuracy"],
                "macro_f1": cbm_metrics["macro_f1"],
                "weighted_f1": cbm_metrics["weighted_f1"],
                "balanced_accuracy": cbm_metrics["balanced_accuracy"],
                "sparsity": sparsity,
                "comparison": {
                    "accuracy": compare(cbm_metrics["accuracy"], expected["accuracy"], args.tolerance_pp),
                    "macro_f1": compare(cbm_metrics["macro_f1"], expected["macro_f1"], args.tolerance_pp),
                    "sparsity": compare(sparsity, expected["sparsity"], args.tolerance_pp),
                },
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cbm_expected, ast_expected = load_expected(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name in args.datasets:
        rows.extend(evaluate_dataset(dataset_name, args, cbm_expected, ast_expected))

    checks = [
        check
        for row in rows
        for check in row["comparison"].values()
    ]
    payload = {
        "protocol": {
            "datasets": args.datasets,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "tolerance_pp": args.tolerance_pp,
            "artifact_bundle": str(args.artifact_bundle.resolve()),
            "expected_comparison": str(args.expected_comparison.resolve()),
        },
        "rows": rows,
        "pass": all(check["pass"] for check in checks),
    }
    (args.output_dir / "global_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "dataset",
        "model_type",
        "variant",
        "checkpoint",
        "split",
        "num_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "sparsity",
    ]
    with (args.output_dir / "global_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    print(json.dumps({"pass": payload["pass"], "rows": len(rows)}, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
