#!/usr/bin/env python3
"""Run Task 1 segmented inference with padding and silence controls."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_utils
from models.ast_backbone import build_ast_backbone
from models.ast_classifier import build_ast_classifier
from scripts.evaluation.run_segmented_audio_ablation import (
    DATASETS,
    config_labels,
    split_to_segments,
)
from scripts.evaluation.silence_padding import (
    build_window_masks,
    energy_weights,
    ensure_highest_rms_fallback,
    masked_pool,
    replace_masked_with_baseline,
    segment_rms_dbfs,
    window_valid_fractions,
    window_valid_sample_counts,
)


STANDARD_POOLS = ("mean", "max", "lme")
THRESHOLDS = (-50.0, -60.0)
VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("saved_activations/revision/t1_silence"))
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=Path("results/revision/t0_baseline/segmented/summary.json"),
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--lme-alpha", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restart-cache", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_path(args: argparse.Namespace, dataset_name: str, split: str) -> Path:
    tag = "w{:g}_h{:g}".format(args.window_sec, args.hop_sec).replace(".", "p")
    return args.cache_root / dataset_name / f"{split}_{tag}_features.pt"


def build_segment_cache(dataset_name: str, split: str, args: argparse.Namespace) -> dict:
    spec = DATASETS[dataset_name]
    path = cache_path(args, dataset_name, split)
    if path.is_file() and not args.restart_cache:
        bundle = torch.load(path, map_location="cpu")
        expected = {
            "dataset": dataset_name,
            "split": split,
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
        }
        if all(bundle.get(key) == value for key, value in expected.items()):
            print(f"[cache] {dataset_name}/{split} -> {path}")
            return bundle

    classes = data_utils.get_dataset_classes(dataset_name)
    classifier = build_ast_classifier(spec["model"], args.device)
    if classifier.num_labels != len(classes) or config_labels(classifier) != classes:
        raise ValueError(f"{dataset_name} AST fine-tuned label order/count mismatch")
    classifier_head = copy.deepcopy(classifier.model.classifier).to(args.device).eval()
    del classifier
    if args.device == "cuda":
        torch.cuda.empty_cache()

    backbone = build_ast_backbone(spec["backbone"], args.device)
    backbone.eval()
    dataset = data_utils.get_audio_dataset(dataset_name, split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )
    sample_rate = int(data_utils.AUDIO_DEFAULTS[dataset_name]["sample_rate"])
    duration_sec = float(data_utils.AUDIO_DEFAULTS[dataset_name]["duration_sec"])
    expected_segments = int(math.floor((duration_sec - args.window_sec) / args.hop_sec)) + 1
    features_all, logits_all, labels_all = [], [], []
    durations_all, valid_fractions_all, valid_counts_all, rms_all = [], [], [], []
    sample_ids, sample_paths = [], []
    times_ref = None

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{dataset_name} {split} segments"):
            if set(int(sr) for sr in batch["sr"].tolist()) != {sample_rate}:
                raise ValueError(f"Unexpected sample rate in {dataset_name}")
            segments, times = split_to_segments(
                batch["audio"], sample_rate, args.window_sec, args.hop_sec
            )
            if segments.shape[1] != expected_segments:
                raise ValueError(
                    f"Expected {expected_segments} segments, got {segments.shape[1]}"
                )
            if times_ref is None:
                times_ref = times
            elif not torch.allclose(times_ref, times):
                raise ValueError("Segment timestamps changed across batches")

            true_duration = batch["duration_sec"].float().clamp(0.0, duration_sec)
            valid_counts = window_valid_sample_counts(
                true_duration, times, sample_rate, args.window_sec
            )
            valid_fractions = window_valid_fractions(
                true_duration, times, args.window_sec
            )
            rms_dbfs = segment_rms_dbfs(segments, valid_counts)
            flat = segments.reshape(-1, segments.shape[-1])
            feature_chunks, logit_chunks = [], []
            for start in range(0, flat.shape[0], args.segment_batch_size):
                current = flat[start : start + args.segment_batch_size]
                sr_batch = torch.full(
                    (current.shape[0],), sample_rate, dtype=torch.long
                )
                features = backbone(current, sample_rates=sr_batch)
                feature_chunks.append(features.detach().cpu())
                logit_chunks.append(classifier_head(features).detach().cpu())
            features = torch.cat(feature_chunks).reshape(segments.shape[0], segments.shape[1], -1)
            ast_logits = torch.cat(logit_chunks).reshape(segments.shape[0], segments.shape[1], -1)
            features_all.append(features)
            logits_all.append(ast_logits)
            labels_all.append(batch["target"].long().cpu())
            durations_all.append(true_duration.cpu())
            valid_fractions_all.append(valid_fractions.cpu())
            valid_counts_all.append(valid_counts.cpu())
            rms_all.append(rms_dbfs.cpu())
            sample_ids.extend(batch["id"])
            sample_paths.extend(batch["path"])

    zero_audio = torch.zeros((1, sample_rate), dtype=torch.float32)
    zero_sr = torch.tensor([sample_rate], dtype=torch.long)
    with torch.no_grad():
        zero_feature = backbone(zero_audio, sample_rates=zero_sr)
        zero_ast_logits = classifier_head(zero_feature).detach().cpu().squeeze(0)
        zero_feature = zero_feature.detach().cpu().squeeze(0)
    bundle = {
        "dataset": dataset_name,
        "split": split,
        "backbone": spec["backbone"],
        "model": spec["model"],
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "times": times_ref,
        "segment_windows_sec": torch.stack([times_ref, times_ref + args.window_sec], dim=1),
        "features": torch.cat(features_all),
        "ast_logits": torch.cat(logits_all),
        "labels": torch.cat(labels_all),
        "duration_sec": torch.cat(durations_all),
        "valid_fractions": torch.cat(valid_fractions_all),
        "valid_sample_counts": torch.cat(valid_counts_all),
        "rms_dbfs": torch.cat(rms_all),
        "zero_feature": zero_feature,
        "zero_ast_logits": zero_ast_logits,
        "sample_ids": sample_ids,
        "sample_paths": sample_paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    print(f"[saved cache] {dataset_name}/{split} -> {path}")
    del backbone, classifier_head
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return bundle


def load_cbm(model_dir: Path) -> tuple[dict[str, torch.Tensor], list[str]]:
    required = ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {model_dir}")
    tensors = {
        name: torch.load(model_dir / f"{name}.pt", map_location="cpu").float()
        for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")
    }
    concepts = [
        line.strip()
        for line in (model_dir / "concepts.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if tensors["W_c"].shape[0] != len(concepts) or tensors["W_g"].shape[1] != len(concepts):
        raise ValueError(f"Concept order/count mismatch in {model_dir}")
    return tensors, concepts


def load_runs(bundle_root: Path) -> list[dict]:
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    return [
        {
            "dataset": item["dataset"],
            "variant": item["variant"],
            "model_dir": bundle_root / item["checkpoint_dir"],
        }
        for item in manifest["models"]
    ]


def method_names(thresholds: tuple[float, ...]) -> list[str]:
    names = ["unmasked", "padding_mask", "energy_weighted"]
    for threshold in thresholds:
        suffix = "m{}".format(int(abs(threshold)))
        names.extend([f"silence_mask_dbfs_{suffix}", f"silence_baseline_dbfs_{suffix}"])
    return names


def threshold_from_method(method: str) -> float | None:
    if "dbfs_m" not in method:
        return None
    return -float(method.rsplit("m", 1)[1])


def prepare_method(
    method: str,
    valid_fractions: torch.Tensor,
    rms_dbfs: torch.Tensor,
    thresholds: tuple[float, ...],
) -> dict[str, torch.Tensor | None | str]:
    padding_mask = valid_fractions > 0.0
    threshold = threshold_from_method(method)
    if method == "unmasked":
        raw_mask = torch.ones_like(padding_mask)
        pool_kind = "standard"
    elif method == "padding_mask" or method == "energy_weighted":
        raw_mask = padding_mask
        pool_kind = "energy" if method == "energy_weighted" else "standard"
    elif method.startswith("silence_mask_dbfs_") or method.startswith("silence_baseline_dbfs_"):
        if threshold not in thresholds:
            raise ValueError(f"Unsupported silence threshold in {method}")
        raw_mask = build_window_masks(valid_fractions, rms_dbfs, threshold)["content_mask"]
        pool_kind = "silence_baseline" if method.startswith("silence_baseline") else "standard"
    else:
        raise ValueError(f"Unknown method: {method}")

    effective_mask, fallback_rows = ensure_highest_rms_fallback(raw_mask, rms_dbfs)
    weights = energy_weights(rms_dbfs, valid_fractions, effective_mask) if pool_kind == "energy" else None
    return {
        "raw_mask": raw_mask,
        "effective_mask": effective_mask,
        "fallback_rows": fallback_rows,
        "weights": weights,
        "pool_kind": pool_kind,
    }


def classification_metrics(labels: torch.Tensor, logits: torch.Tensor) -> dict:
    true = labels.cpu().numpy()
    predictions = torch.argmax(logits.cpu(), dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(true, predictions)),
        "macro_f1": float(f1_score(true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predictions)),
        "predictions": predictions.tolist(),
    }


def save_diagnostics(
    output_root: Path,
    dataset_name: str,
    variant: str,
    method: str,
    pool: str,
    labels: torch.Tensor,
    predictions: list[int],
) -> tuple[str, str]:
    classes = data_utils.get_dataset_classes(dataset_name)
    directory = output_root / "diagnostics" / method / dataset_name / variant / pool
    directory.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels.numpy(), np.asarray(predictions), labels=list(range(len(classes))))
    matrix_path = directory / "confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class"] + classes)
        for index, row in enumerate(matrix):
            writer.writerow([classes[index]] + row.tolist())
    histogram = np.bincount(predictions, minlength=len(classes))
    histogram_path = directory / "predicted_class_histogram.csv"
    with histogram_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_idx", "class_name", "count"])
        writer.writerows((index, classes[index], int(count)) for index, count in enumerate(histogram))
    return str(matrix_path), str(histogram_path)


def evaluate_model(
    dataset_name: str,
    variant: str,
    model_type: str,
    tensors: dict[str, torch.Tensor] | None,
    bundle: dict,
    method: str,
    thresholds: tuple[float, ...],
    args: argparse.Namespace,
    save_outputs: bool,
) -> list[dict]:
    device = torch.device(args.device)
    features = bundle["features"].to(device=device, dtype=torch.float32)
    valid_fractions = bundle["valid_fractions"].to(device)
    rms_dbfs = bundle["rms_dbfs"].to(device)
    if model_type == "ast":
        values = bundle["ast_logits"].to(device=device, dtype=torch.float32)
        zero_values = bundle["zero_ast_logits"].to(device=device, dtype=torch.float32)
    else:
        concepts = F.linear(features, tensors["W_c"])
        proj_mean = tensors["proj_mean"].reshape(-1)
        proj_std = tensors["proj_std"].reshape(-1)
        values = (concepts - proj_mean) / proj_std
        zero_concepts = F.linear(bundle["zero_feature"].to(device=device, dtype=torch.float32), tensors["W_c"])
        zero_values = (zero_concepts - proj_mean) / proj_std

    context = prepare_method(method, valid_fractions, rms_dbfs, thresholds)
    raw_mask = context["raw_mask"]
    effective_mask = context["effective_mask"]
    fallback_rows = context["fallback_rows"]
    pool_kind = context["pool_kind"]
    weights = context["weights"]
    if pool_kind == "energy":
        pools = ("energy_weighted",)
    else:
        pools = STANDARD_POOLS

    if pool_kind == "silence_baseline":
        pooled_values, pool_mask = replace_masked_with_baseline(
            values, zero_values, raw_mask, effective_mask, fallback_rows
        )
    else:
        pooled_values, pool_mask = values, effective_mask

    rows = []
    for pool in pools:
        pooled, _, _ = masked_pool(
            pooled_values,
            pool,
            pool_mask,
            rms_dbfs,
            alpha=args.lme_alpha,
            weights=weights,
        )
        if model_type == "ast":
            logits = pooled
        else:
            logits = F.linear(pooled, tensors["W_g"], tensors["b_g"])
        metric = classification_metrics(bundle["labels"], logits)
        row = {
            "dataset": dataset_name,
            "variant": variant,
            "model_type": "fine_tuned_ast_segmented" if model_type == "ast" else "cbm_segmented",
            "split": bundle["split"],
            "method": method,
            "pool": pool,
            "threshold_dbfs": threshold_from_method(method),
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "lme_alpha": args.lme_alpha,
            "num_samples": int(bundle["labels"].numel()),
            "num_segments": int(values.shape[1]),
            "accuracy": metric["accuracy"],
            "macro_f1": metric["macro_f1"],
            "weighted_f1": metric["weighted_f1"],
            "balanced_accuracy": metric["balanced_accuracy"],
            "raw_retained_fraction": float(raw_mask.float().mean()),
            "effective_retained_fraction": float(effective_mask.float().mean()),
            "fallback_count": int(fallback_rows.sum()),
        }
        if save_outputs:
            matrix_path, histogram_path = save_diagnostics(
                args.output_root,
                dataset_name,
                variant,
                method,
                pool,
                bundle["labels"],
                metric["predictions"],
            )
            row["confusion_matrix"] = matrix_path
            row["predicted_class_histogram"] = histogram_path
        rows.append(row)
    return rows


def run_split(
    split: str,
    datasets: list[str],
    methods: list[str],
    runs: list[dict],
    caches: dict[tuple[str, str], dict],
    args: argparse.Namespace,
    save_outputs: bool,
) -> list[dict]:
    rows = []
    for dataset_name in datasets:
        bundle = caches[(dataset_name, split)]
        ast_rows = evaluate_model(
            dataset_name,
            "fine_tuned_ast",
            "ast",
            None,
            bundle,
            methods[0],
            args.thresholds,
            args,
            save_outputs,
        )
        rows.extend(ast_rows)
        for method in methods[1:]:
            rows.extend(
                evaluate_model(
                    dataset_name,
                    "fine_tuned_ast",
                    "ast",
                    None,
                    bundle,
                    method,
                    args.thresholds,
                    args,
                    save_outputs,
                )
            )
        dataset_runs = [run for run in runs if run["dataset"] == dataset_name]
        if len(dataset_runs) != 4:
            raise RuntimeError(f"Expected four CBMs for {dataset_name}, found {len(dataset_runs)}")
        for run in dataset_runs:
            tensors, _ = load_cbm(run["model_dir"])
            tensors = {name: value.to(args.device) for name, value in tensors.items()}
            for method in methods:
                rows.extend(
                    evaluate_model(
                        dataset_name,
                        run["variant"],
                        "cbm",
                        tensors,
                        bundle,
                        method,
                        args.thresholds,
                        args,
                        save_outputs,
                    )
                )
    return rows


def select_threshold(validation_rows: list[dict], datasets: list[str], thresholds: tuple[float, ...]) -> dict:
    selection = {}
    for dataset_name in datasets:
        scores = {}
        for threshold in thresholds:
            method = "silence_mask_dbfs_m{}".format(int(abs(threshold)))
            values = [
                row["macro_f1"]
                for row in validation_rows
                if row["dataset"] == dataset_name
                and row["method"] == method
                and row["pool"] in STANDARD_POOLS
            ]
            scores[str(threshold)] = float(np.mean(values))
        selected = max(thresholds, key=lambda threshold: (scores[str(threshold)], threshold))
        selection[dataset_name] = {
            "scores_macro_f1_mean_over_models_and_pools": scores,
            "selected_threshold_dbfs": selected,
            "selection_rule": "highest validation macro-F1 averaged over AST and all four CBMs and mean/max/LME pools",
        }
    return selection


def write_summary(path: Path, protocol: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"protocol": protocol, "rows": rows}, indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["dataset"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def zero_waveform_diagnostic(
    args: argparse.Namespace,
    runs: list[dict],
    caches: dict[tuple[str, str], dict],
) -> dict:
    diagnostics = []
    with torch.no_grad():
        for dataset_name in args.datasets:
            cache = caches[(dataset_name, "test")]
            ast_feature = cache["zero_feature"].float()
            ast_logits = cache["zero_ast_logits"].float()
            dataset_runs = [run for run in runs if run["dataset"] == dataset_name]
            for run in dataset_runs:
                tensors, _ = load_cbm(run["model_dir"])
                tensors = {name: value.to(args.device) for name, value in tensors.items()}
                concept = F.linear(ast_feature.to(args.device), tensors["W_c"])
                standardized = (concept - tensors["proj_mean"].reshape(-1)) / tensors["proj_std"].reshape(-1)
                logits = F.linear(standardized, tensors["W_g"], tensors["b_g"])
                top_values, top_indices = torch.topk(logits, k=min(5, logits.numel()))
                classes = data_utils.get_dataset_classes(dataset_name)
                diagnostics.append(
                {
                    "dataset": dataset_name,
                    "variant": run["variant"],
                    "checkpoint": str(run["model_dir"]),
                    "waveform": {"duration_sec": 1.0, "rms_dbfs": -math.inf},
                    "ast": {
                        "feature_shape": list(ast_feature.shape),
                        "feature_l2": float(ast_feature.norm()),
                        "feature_min": float(ast_feature.min()),
                        "feature_max": float(ast_feature.max()),
                        "logit_min": float(ast_logits.min()),
                        "logit_max": float(ast_logits.max()),
                    },
                    "W_c": {
                        "output_shape": list(concept.shape),
                        "output_l2": float(concept.norm()),
                        "output_min": float(concept.min()),
                        "output_max": float(concept.max()),
                    },
                    "standardization": {
                        "output_l2": float(standardized.norm()),
                        "output_min": float(standardized.min()),
                        "output_max": float(standardized.max()),
                    },
                    "W_F": {
                        "source_tensor": "release checkpoint W_g.pt",
                        "logits": logits.tolist(),
                        "top5": [
                            {"class_idx": int(index), "class_name": classes[int(index)], "logit": float(value)}
                            for value, index in zip(top_values, top_indices)
                        ],
                        "predicted_class_idx": int(torch.argmax(logits)),
                        "predicted_class_name": classes[int(torch.argmax(logits))],
                    },
                }
            )
    return {"description": "AST zero 1-second waveform through AST -> W_c -> standardization -> W_F", "rows": diagnostics}


def compare_baseline(rows: list[dict], baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = {
        (row["dataset"], row["variant"], row["model_type"], row["pool"]): row
        for row in baseline["rows"]
    }
    checks = []
    for row in rows:
        if row["method"] != "unmasked" or row["pool"] not in STANDARD_POOLS:
            continue
        key = (row["dataset"], row["variant"], row["model_type"], row["pool"])
        previous = expected[key]
        for metric in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"):
            delta = row[metric] - previous[metric]
            checks.append(
                {
                    "key": list(key),
                    "metric": metric,
                    "current": row[metric],
                    "task0": previous[metric],
                    "delta": delta,
                    "pass": abs(delta) <= 1e-6,
                }
            )
    return {"tolerance_absolute": 1e-6, "checks": checks, "pass": all(item["pass"] for item in checks)}


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_seed(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.thresholds = tuple(THRESHOLDS)
    runs = load_runs(args.artifact_bundle.resolve())
    caches = {}
    for dataset_name in args.datasets:
        for split in ("val", "test"):
            split_name = DATASETS[dataset_name]["split"] if split == "test" else {
                "esc50": "fold1_val",
                "urbansound8k": "fold10_val",
                "cremad": "val",
            }[dataset_name]
            caches[(dataset_name, split)] = build_segment_cache(dataset_name, split_name, args)

    base_methods = method_names(args.thresholds)
    validation_rows = run_split("val", args.datasets, base_methods, runs, caches, args, save_outputs=False)
    validation_protocol = {
        "split": "validation only",
        "seed": args.seed,
        "thresholds_dbfs": list(args.thresholds),
        "selection_rule": "validation macro-F1 only; test labels are never used for threshold selection",
    }
    write_summary(args.output_root / "validation" / "summary.json", validation_protocol, validation_rows)
    selection = select_threshold(validation_rows, args.datasets, args.thresholds)
    (args.output_root / "validation" / "threshold_selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    test_rows = []
    for dataset_name in args.datasets:
        selected_suffix = "m{}".format(int(abs(selection[dataset_name]["selected_threshold_dbfs"])))
        dataset_methods = base_methods + [
            f"silence_mask_dbfs_{selected_suffix}",
            f"silence_baseline_dbfs_{selected_suffix}",
        ]
        dataset_rows = run_split(
            "test",
            [dataset_name],
            dataset_methods,
            runs,
            caches,
            args,
            save_outputs=True,
        )
        # Keep the selected method names explicit in the machine-readable table.
        selected_threshold = selection[dataset_name]["selected_threshold_dbfs"]
        selected_method_names = {
            f"silence_mask_dbfs_m{int(abs(selected_threshold))}": f"silence_mask_selected_m{int(abs(selected_threshold))}",
            f"silence_baseline_dbfs_m{int(abs(selected_threshold))}": f"silence_baseline_selected_m{int(abs(selected_threshold))}",
        }
        seen_methods = {}
        for row in dataset_rows:
            method = row["method"]
            occurrence = seen_methods.get((row["variant"], row["model_type"], row["pool"], method), 0)
            if occurrence > 0 and method in selected_method_names:
                row["method"] = selected_method_names[method]
            seen_methods[(row["variant"], row["model_type"], row["pool"], method)] = occurrence + 1
        test_rows.extend(dataset_rows)

    test_protocol = {
        "split": "test",
        "seed": args.seed,
        "thresholds_dbfs": list(args.thresholds),
        "selected_thresholds": selection,
        "padding_definition": "window overlap with the true manifest duration; padded-only windows are masked",
        "silence_definition": "RMS dBFS computed over true, non-padding samples; windows below threshold are masked",
        "energy_definition": "RMS energy times true-duration fraction, normalized over retained windows",
        "fallback_definition": "if all windows are masked, retain the highest-RMS window",
        "silence_baseline_definition": "replace masked windows with the AST representation of a zero 1-second waveform",
    }
    write_summary(args.output_root / "segmented" / "summary.json", test_protocol, test_rows)
    baseline_result = compare_baseline(
        [row for row in test_rows if row["method"] == "unmasked"], args.baseline_summary
    )
    (args.output_root / "baseline_preservation.json").write_text(
        json.dumps(baseline_result, indent=2), encoding="utf-8"
    )
    zero_diagnostic = zero_waveform_diagnostic(args, runs, caches)
    (args.output_root / "zero_waveform_diagnostic.json").write_text(
        json.dumps(zero_diagnostic, indent=2), encoding="utf-8"
    )
    final = {
        "validation_threshold_selection": selection,
        "baseline_preserved": baseline_result["pass"],
        "test_rows": len(test_rows),
        "output_root": str(args.output_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    if not baseline_result["pass"]:
        raise SystemExit("Unmasked Task 0 baseline changed")


if __name__ == "__main__":
    main()
