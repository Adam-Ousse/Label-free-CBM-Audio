#!/usr/bin/env python3
"""Evaluate all AST/CBM ablations with the established segmented protocol.

This shared-feature, multi-model runner preserves the established segmented
inference windowing, concept normalization, and mean/max/LME pooling while
computing each dataset's AST segment features only once.
"""

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_utils
from models.ast_classifier import build_ast_classifier


DATASETS = {
    "esc50": {
        "backbone": "ast_esc50",
        "model": "Adam-ousse/ast-esc50-finetuned-fold1",
        "split": "fold1_test",
    },
    "urbansound8k": {
        "backbone": "ast_urbansound8k",
        "model": "Adam-ousse/ast-urbansound8k-finetuned-fold10",
        "split": "fold10_test",
    },
    "cremad": {
        "backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
        "model": "Adam-ousse/ast-cremad-finetuned",
        "split": "test",
    },
}
POOLS = ("mean", "max", "lme")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lme-alpha", type=float, default=5.0)
    parser.add_argument("--cbm-summary", default="results/audio_concept_ablation/cbm/summary.json")
    parser.add_argument(
        "--artifact-bundle",
        type=Path,
        help=(
            "Canonical bundle produced by scripts.release.build_google_drive_bundle. "
            "When set, its manifest replaces --cbm-summary and includes the latest "
            "speech-targeted CREMA-D checkpoints."
        ),
    )
    parser.add_argument("--output-root", default="results/audio_concept_ablation/segmented")
    parser.add_argument("--cache-root", default="saved_activations/audio_concept_ablation/segmented")
    parser.add_argument("--restart-cache", action="store_true")
    return parser.parse_args()


def load_cbm_runs(args):
    """Load model locations from a training summary or canonical release bundle."""
    if args.artifact_bundle is not None:
        bundle_root = args.artifact_bundle.resolve()
        manifest_path = bundle_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [
            {
                "dataset": model["dataset"],
                "variant": model["variant"],
                "model_dir": str(bundle_root / model["checkpoint_dir"]),
            }
            for model in manifest["models"]
        ]

    summary = json.loads(Path(args.cbm_summary).read_text(encoding="utf-8"))
    return summary["runs"]


def split_to_segments(audio, sample_rate, window_sec, hop_sec):
    """Match the existing segmented_cbm_inference.py window construction."""
    if audio.dim() == 3 and audio.shape[1] == 1:
        audio = audio.squeeze(1)
    if audio.dim() != 2:
        raise ValueError("Expected [B,T] or [B,1,T], got {}".format(tuple(audio.shape)))
    window = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    if window <= 0 or hop <= 0:
        raise ValueError("window_sec and hop_sec must be positive")
    if audio.shape[-1] < window:
        audio = F.pad(audio, (0, window - audio.shape[-1]))
    starts = list(range(0, audio.shape[-1] - window + 1, hop)) or [0]
    segments = torch.stack([audio[:, start : start + window] for start in starts], dim=1)
    times = torch.tensor([start / float(sample_rate) for start in starts], dtype=torch.float32)
    return segments, times


def lme_pool(x, alpha):
    return torch.logsumexp(float(alpha) * x, dim=1) / float(alpha) - math.log(x.shape[1]) / float(alpha)


def pool_tensor(x, pool, alpha):
    x = x.float()
    if pool == "mean":
        return x.mean(dim=1)
    if pool == "max":
        return x.max(dim=1).values
    if pool == "lme":
        return lme_pool(x, alpha)
    raise ValueError("Unknown pool: {}".format(pool))


def classification_metrics(labels, logits):
    true = labels.cpu().numpy()
    pred = torch.argmax(logits.cpu(), dim=1).numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "predictions": pred.tolist(),
    }


def config_labels(classifier):
    return [
        str(classifier.id2label.get(i, classifier.id2label.get(str(i))))
        for i in range(classifier.num_labels)
    ]


def cache_path(args, dataset_name):
    tag = "w{:g}_h{:g}".format(args.window_sec, args.hop_sec).replace(".", "p")
    return Path(args.cache_root) / dataset_name / "{}_segment_ast_features.pt".format(tag)


def build_segment_cache(dataset_name, args):
    spec = DATASETS[dataset_name]
    path = cache_path(args, dataset_name)
    if path.is_file() and not args.restart_cache:
        bundle = torch.load(path, map_location="cpu")
        expected = {
            "dataset": dataset_name,
            "split": spec["split"],
            "backbone": spec["backbone"],
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
        }
        if all(bundle.get(key) == value for key, value in expected.items()):
            print("[cache] {} -> {}".format(dataset_name, path))
            return bundle

    classes = data_utils.get_dataset_classes(dataset_name)
    classifier = build_ast_classifier(spec["model"], args.device)
    if classifier.num_labels != len(classes) or config_labels(classifier) != classes:
        raise ValueError("{} fine-tuned AST label order/count mismatch".format(dataset_name))
    # ASTModel.pooler_output is exactly the input to this checkpoint's classifier.
    classifier_head = copy.deepcopy(classifier.model.classifier).to(args.device).eval()
    del classifier
    torch.cuda.empty_cache()

    backbone, _ = data_utils.get_target_model(spec["backbone"], args.device)
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
    sample_rate = int(data_utils.AUDIO_DEFAULTS[dataset_name]["sample_rate"])
    duration_sec = float(data_utils.AUDIO_DEFAULTS[dataset_name]["duration_sec"])
    expected_segments = int(math.floor((duration_sec - args.window_sec) / args.hop_sec)) + 1
    all_features, all_ast_logits, all_labels = [], [], []
    sample_ids, sample_paths = [], []
    times_ref = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="{} segments".format(dataset_name)):
            if set(int(sr) for sr in batch["sr"].tolist()) != {sample_rate}:
                raise ValueError("Unexpected sample rate in {}".format(dataset_name))
            segments, times = split_to_segments(
                batch["audio"], sample_rate, args.window_sec, args.hop_sec
            )
            if segments.shape[1] != expected_segments:
                raise ValueError("Expected {} segments, got {}".format(expected_segments, segments.shape[1]))
            if times_ref is None:
                times_ref = times
            elif not torch.allclose(times_ref, times):
                raise ValueError("Segment timestamps changed across batches")

            flat = segments.reshape(-1, segments.shape[-1])
            feature_chunks, logit_chunks = [], []
            for start in range(0, flat.shape[0], args.segment_batch_size):
                curr = flat[start : start + args.segment_batch_size]
                sr_batch = torch.full((curr.shape[0],), sample_rate, dtype=torch.long)
                features = backbone(curr, sample_rates=sr_batch)
                feature_chunks.append(features.detach().cpu())
                logit_chunks.append(classifier_head(features).detach().cpu())
            features = torch.cat(feature_chunks).reshape(segments.shape[0], segments.shape[1], -1)
            ast_logits = torch.cat(logit_chunks).reshape(segments.shape[0], segments.shape[1], -1)
            all_features.append(features)
            all_ast_logits.append(ast_logits)
            all_labels.append(batch["target"].long().cpu())
            sample_ids.extend(batch["id"])
            sample_paths.extend(batch["path"])

    bundle = {
        "dataset": dataset_name,
        "split": spec["split"],
        "backbone": spec["backbone"],
        "model": spec["model"],
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "lme_alpha": float(args.lme_alpha),
        "times": times_ref,
        "segment_windows_sec": torch.stack([times_ref, times_ref + float(args.window_sec)], dim=1),
        "features": torch.cat(all_features),
        "ast_logits": torch.cat(all_ast_logits),
        "labels": torch.cat(all_labels),
        "sample_ids": sample_ids,
        "sample_paths": sample_paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    print("[saved cache] {} shape={} -> {}".format(dataset_name, tuple(bundle["features"].shape), path))
    del backbone, classifier_head
    torch.cuda.empty_cache()
    return bundle


def load_cbm(model_dir):
    model_dir = Path(model_dir)
    required = ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing {} in {}".format(missing, model_dir))
    tensors = {
        name: torch.load(model_dir / "{}.pt".format(name), map_location="cpu").float()
        for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")
    }
    concepts = [line.strip() for line in (model_dir / "concepts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if tensors["W_c"].shape[0] != len(concepts) or tensors["W_g"].shape[1] != len(concepts):
        raise ValueError("Concept order/count mismatch in {}".format(model_dir))
    return tensors, concepts


def save_predictions(path, ids, labels, results):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "mean_pred", "max_pred", "lme_pred"])
        for index, sample_id in enumerate(ids):
            writer.writerow(
                [sample_id, int(labels[index])]
                + [results[pool]["predictions"][index] for pool in POOLS]
            )


def evaluate_ast(dataset_name, bundle, args):
    output_dir = Path(args.output_root) / dataset_name / "fine_tuned_ast"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        pool: classification_metrics(bundle["labels"], pool_tensor(bundle["ast_logits"], pool, args.lme_alpha))
        for pool in POOLS
    }
    payload = {
        "dataset": dataset_name,
        "model_type": "fine_tuned_ast_segmented",
        "model": DATASETS[dataset_name]["model"],
        "split": bundle["split"],
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "lme_alpha": args.lme_alpha,
        "num_samples": int(bundle["labels"].numel()),
        "num_segments": int(bundle["ast_logits"].shape[1]),
        "pools": {pool: {k: v for k, v in results[pool].items() if k != "predictions"} for pool in POOLS},
    }
    (output_dir / "segmented_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_predictions(output_dir / "segmented_predictions.csv", bundle["sample_ids"], bundle["labels"], results)
    return payload


def evaluate_cbm(run, bundle, args):
    tensors, concepts = load_cbm(run["model_dir"])
    temporal = F.linear(bundle["features"].float(), tensors["W_c"])
    temporal = (temporal - tensors["proj_mean"]) / tensors["proj_std"]
    results = {}
    for pool in POOLS:
        pooled = pool_tensor(temporal, pool, args.lme_alpha)
        logits = F.linear(pooled, tensors["W_g"], tensors["b_g"])
        results[pool] = classification_metrics(bundle["labels"], logits)

    output_dir = Path(args.output_root) / run["dataset"] / run["variant"]
    output_dir.mkdir(parents=True, exist_ok=True)
    temporal_path = output_dir / "test_temporal_concepts.pt"
    torch.save(
        {
            "temporal_concepts": temporal.half(),
            "times": bundle["times"],
            "segment_times_sec": bundle["times"],
            "segment_windows_sec": bundle["segment_windows_sec"],
            "labels": bundle["labels"],
            "sample_ids": bundle["sample_ids"],
            "sample_paths": bundle["sample_paths"],
            "dataset": run["dataset"],
            "split": bundle["split"],
            "cbm_dir": run["model_dir"],
            "backbone": bundle["backbone"],
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "tensor_layout": ["sample", "segment", "concept"],
            "concept_dtype": "float16",
            "concepts": concepts,
        },
        temporal_path,
    )
    payload = {
        "dataset": run["dataset"],
        "variant": run["variant"],
        "model_type": "cbm_segmented",
        "cbm_dir": run["model_dir"],
        "split": bundle["split"],
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "lme_alpha": args.lme_alpha,
        "num_samples": int(bundle["labels"].numel()),
        "num_segments": int(temporal.shape[1]),
        "num_concepts": int(temporal.shape[2]),
        "temporal_concepts_path": str(temporal_path),
        "pools": {pool: {k: v for k, v in results[pool].items() if k != "predictions"} for pool in POOLS},
    }
    (output_dir / "segmented_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_predictions(output_dir / "segmented_predictions.csv", bundle["sample_ids"], bundle["labels"], results)
    return payload


def flat_rows(payload):
    rows = []
    for pool in POOLS:
        metric = payload["pools"][pool]
        rows.append(
            {
                "dataset": payload["dataset"],
                "model_type": payload["model_type"],
                "variant": payload.get("variant", "fine_tuned_ast"),
                "pool": pool,
                "accuracy": metric["accuracy"],
                "macro_f1": metric["macro_f1"],
                "weighted_f1": metric["weighted_f1"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "num_samples": payload["num_samples"],
                "num_segments": payload["num_segments"],
                "num_concepts": payload.get("num_concepts", ""),
                "model_dir": payload.get("cbm_dir", payload.get("model")),
            }
        )
    return rows


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cbm_runs = load_cbm_runs(args)
    all_payloads, rows = [], []
    for dataset_name in args.datasets:
        print("\n=== segmented {} ===".format(dataset_name))
        bundle = build_segment_cache(dataset_name, args)
        ast_payload = evaluate_ast(dataset_name, bundle, args)
        all_payloads.append(ast_payload)
        rows.extend(flat_rows(ast_payload))
        print("AST " + " ".join("{}={:.4f}/{:.4f}".format(p, ast_payload["pools"][p]["accuracy"], ast_payload["pools"][p]["macro_f1"]) for p in POOLS))
        dataset_runs = [run for run in cbm_runs if run["dataset"] == dataset_name]
        if len(dataset_runs) != 4:
            raise RuntimeError("Expected four CBMs for {}, found {}".format(dataset_name, len(dataset_runs)))
        for run in dataset_runs:
            payload = evaluate_cbm(run, bundle, args)
            all_payloads.append(payload)
            rows.extend(flat_rows(payload))
            print("{} ".format(run["variant"]) + " ".join("{}={:.4f}/{:.4f}".format(p, payload["pools"][p]["accuracy"], payload["pools"][p]["macro_f1"]) for p in POOLS))
        del bundle

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "lme_alpha": args.lme_alpha,
            "pools": list(POOLS),
            "cbm_pooling": "pool standardized temporal concepts, then apply sparse classifier",
            "ast_pooling": "pool per-segment logits",
        },
        "models": all_payloads,
        "rows": rows,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nCompleted {} segmented model evaluations ({} pooled result rows).".format(len(all_payloads), len(rows)))


if __name__ == "__main__":
    main()
