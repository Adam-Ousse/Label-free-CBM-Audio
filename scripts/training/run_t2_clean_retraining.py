#!/usr/bin/env python3
"""Retrain Task 2 CBMs from leakage-clean concept vocabularies.

This runner reads the canonical checkpoint arguments, changes only the concept
set and output/cache locations, and reuses the frozen canonical AST backbone.
Canonical checkpoints, release artifacts, and historical results are never
used as write targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import data_utils
import train_cbm
import utils


DATASETS = ("esc50", "urbansound8k", "cremad")
VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")
ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ROOT / "release" / "google_drive_bundle"
OUTPUT_ROOT = ROOT / "results" / "revision" / "t2_leakage" / "retrained"
CACHE_ROOT = ROOT / "saved_activations" / "revision" / "t2_leakage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nonempty_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vocabulary_equivalence(original: Path, clean: Path) -> dict:
    original_bytes = original.read_bytes()
    clean_bytes = clean.read_bytes()
    original_lines = nonempty_lines(original)
    clean_lines = nonempty_lines(clean)
    return {
        "original_path": str(original),
        "clean_path": str(clean),
        "byte_equal": original_bytes == clean_bytes,
        "content_equal": original_lines == clean_lines,
        "original_sha256": sha256(original),
        "clean_sha256": sha256(clean),
        "original_bytes": len(original_bytes),
        "clean_bytes": len(clean_bytes),
        "original_concepts": len(original_lines),
        "clean_concepts": len(clean_lines),
    }


def canonical_paths(dataset: str, variant: str) -> tuple[Path, Path]:
    checkpoint = CANONICAL_ROOT / "checkpoints" / dataset / variant
    args_path = checkpoint / "args.txt"
    if not args_path.is_file():
        raise FileNotFoundError(f"Missing canonical checkpoint args: {args_path}")
    return checkpoint, args_path


def checkpoint_input_equivalence(
    original: Path,
    canonical_args: dict,
) -> dict:
    raw_path = Path(str(canonical_args.get("concept_set", "")))
    candidate = raw_path if raw_path.is_absolute() else ROOT / raw_path
    if not candidate.is_file():
        return {
            "canonical_input_path": str(candidate),
            "canonical_input_exists": False,
            "canonical_input_content_equal": False,
        }
    return {
        "canonical_input_path": str(candidate),
        "canonical_input_exists": True,
        "canonical_input_content_equal": nonempty_lines(candidate) == nonempty_lines(original),
        "canonical_input_sha256": sha256(candidate),
    }


def safe_output_root(path: Path) -> Path:
    resolved = path.resolve()
    forbidden = [CANONICAL_ROOT.resolve(), (ROOT / "results").resolve() / "audio_concept_ablation"]
    if resolved == CANONICAL_ROOT.resolve() or CANONICAL_ROOT.resolve() in resolved.parents:
        raise ValueError(f"Refusing to write inside canonical release bundle: {resolved}")
    if any(resolved == item or item in resolved.parents for item in forbidden):
        raise ValueError(f"Refusing to write into historical results: {resolved}")
    return resolved


def complete_model_dirs(models_dir: Path) -> list[Path]:
    required = {
        "W_c.pt",
        "W_g.pt",
        "b_g.pt",
        "proj_mean.pt",
        "proj_std.pt",
        "metrics.txt",
        "concepts.txt",
        "args.txt",
    }
    if not models_dir.is_dir():
        return []
    return sorted(
        child
        for child in models_dir.iterdir()
        if child.is_dir() and required.issubset({item.name for item in child.iterdir()})
    )


def model_summary(model_dir: Path, input_concepts: int) -> dict:
    metrics = json.loads((model_dir / "metrics.txt").read_text(encoding="utf-8"))
    concepts = nonempty_lines(model_dir / "concepts.txt")
    W_c = torch.load(model_dir / "W_c.pt", map_location="cpu").float()
    W_g = torch.load(model_dir / "W_g.pt", map_location="cpu").float()
    nonzero = W_g.abs() > 1e-5
    nonzero_per_class = nonzero.sum(dim=1)
    sparsity = metrics.get("sparsity", {})
    test = metrics.get("test_metrics", {})
    return {
        "checkpoint": str(model_dir),
        "input_concepts": input_concepts,
        "retained_concepts": len(concepts),
        "W_c_shape": list(W_c.shape),
        "W_g_shape": list(W_g.shape),
        "test_accuracy": test.get("accuracy"),
        "test_macro_f1": test.get("f1_macro"),
        "test_weighted_f1": test.get("f1_weighted"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "nonzero_weights": int(sparsity.get("Non-zero weights", nonzero.sum().item())),
        "total_weights": int(sparsity.get("Total weights", W_g.numel())),
        "fraction_nonzero": float(sparsity.get("Percentage non-zero", nonzero.float().mean().item())),
        "sparsity_fraction": 1.0 - float(sparsity.get("Percentage non-zero", nonzero.float().mean().item())),
        "avg_nonzero_per_class": float(nonzero_per_class.float().mean()),
        "min_nonzero_per_class": int(nonzero_per_class.min()),
        "max_nonzero_per_class": int(nonzero_per_class.max()),
    }


def validate_checkpoint(model_dir: Path, expected_input_concepts: int, device: str) -> dict:
    required = {"W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "metrics.txt", "concepts.txt", "args.txt"}
    missing = sorted(required - {item.name for item in model_dir.iterdir()})
    if missing:
        raise FileNotFoundError(f"{model_dir} missing {missing}")
    args = json.loads((model_dir / "args.txt").read_text(encoding="utf-8"))
    concepts = nonempty_lines(model_dir / "concepts.txt")
    W_c = torch.load(model_dir / "W_c.pt", map_location="cpu").float()
    W_g = torch.load(model_dir / "W_g.pt", map_location="cpu").float()
    b_g = torch.load(model_dir / "b_g.pt", map_location="cpu").float()
    proj_mean = torch.load(model_dir / "proj_mean.pt", map_location="cpu").float().reshape(-1)
    proj_std = torch.load(model_dir / "proj_std.pt", map_location="cpu").float().reshape(-1)
    if len(concepts) != expected_input_concepts and len(concepts) != W_c.shape[0]:
        raise ValueError(f"Unexpected concept dimensions in {model_dir}: input={expected_input_concepts}, retained={len(concepts)}")
    classes = data_utils.get_dataset_classes(args["dataset"])
    if W_c.ndim != 2 or W_c.shape[0] != len(concepts):
        raise ValueError(f"W_c/concepts mismatch in {model_dir}")
    if W_g.shape != (len(classes), len(concepts)) or b_g.shape != (len(classes),):
        raise ValueError(f"Classifier shape mismatch in {model_dir}: W_g={tuple(W_g.shape)} b_g={tuple(b_g.shape)}")
    if proj_mean.shape != (len(concepts),) or proj_std.shape != (len(concepts),):
        raise ValueError(f"Projection normalization mismatch in {model_dir}")

    activation_dir = Path(args["activation_dir"])
    if not activation_dir.is_absolute():
        activation_dir = ROOT / activation_dir
    target_path, _, _ = utils.get_audio_save_names(
        args["clap_model"],
        args["backbone"],
        args["feature_layer"],
        utils.get_audio_split_cache_key(args["dataset"], args["test_split"]),
        args["concept_set"],
        "avg",
        str(activation_dir),
    )
    target_features = torch.load(target_path, map_location="cpu").float()
    labels = torch.tensor(
        [sample["label_idx"] for sample in data_utils.get_audio_dataset(args["dataset"], args["test_split"]).samples],
        dtype=torch.long,
    )
    if target_features.shape[0] != labels.numel():
        raise ValueError(f"Test feature/label count mismatch in {model_dir}")
    with torch.no_grad():
        projected = F.linear(target_features.to(device), W_c.to(device))
        projected = (projected - proj_mean.to(device)) / proj_std.to(device)
        logits = F.linear(projected, W_g.to(device), b_g.to(device)).cpu()
    predictions = logits.argmax(dim=1).numpy()
    truth = labels.numpy()
    direct = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "num_test_samples": int(labels.numel()),
    }
    recorded = json.loads((model_dir / "metrics.txt").read_text(encoding="utf-8")).get("test_metrics", {})
    recorded_to_direct = {
        "accuracy": "accuracy",
        "f1_macro": "macro_f1",
        "f1_weighted": "weighted_f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    for metric, direct_key in recorded_to_direct.items():
        if recorded.get(metric) is not None and abs(float(recorded[metric]) - direct[direct_key]) > 1e-6:
            raise ValueError(f"Recorded/direct {metric} mismatch in {model_dir}")
    return {"checkpoint": str(model_dir), "required_files": True, "shape_valid": True, **direct}


def make_train_args(canonical: dict, dataset: str, variant: str, clean_path: Path, output_dir: Path, cache_dir: Path) -> argparse.Namespace:
    values = dict(canonical)
    values.update(
        {
            "dataset": dataset,
            "concept_set": str(clean_path),
            "activation_dir": str(cache_dir),
            "save_dir": str(output_dir / "models"),
            "results_dir": str(output_dir),
            "run_name": f"t2_clean_{variant}",
            "device": canonical.get("device", "cuda"),
        }
    )
    return argparse.Namespace(**values)


def comparison_row(dataset: str, variant: str, original: dict, clean: dict, vocab: dict, provenance: dict) -> dict:
    def delta(key: str) -> float | None:
        if original.get(key) is None or clean.get(key) is None:
            return None
        return float(clean[key]) - float(original[key])

    return {
        "dataset": dataset,
        "variant": variant,
        "original_accuracy": original.get("test_accuracy"),
        "clean_accuracy": clean.get("test_accuracy"),
        "accuracy_delta": delta("test_accuracy"),
        "original_macro_f1": original.get("test_macro_f1"),
        "clean_macro_f1": clean.get("test_macro_f1"),
        "macro_f1_delta": delta("test_macro_f1"),
        "original_vocabulary_size": vocab["original_concepts"],
        "clean_vocabulary_size": vocab["clean_concepts"],
        "original_retained_concepts": original.get("retained_concepts"),
        "clean_retained_concepts": clean.get("retained_concepts"),
        "original_sparsity": original.get("sparsity_fraction"),
        "clean_sparsity": clean.get("sparsity_fraction"),
        "original_nonzero_weights": original.get("nonzero_weights"),
        "clean_nonzero_weights": clean.get("nonzero_weights"),
        "original_avg_nonzero_per_class": original.get("avg_nonzero_per_class"),
        "clean_avg_nonzero_per_class": clean.get("avg_nonzero_per_class"),
        "clean_content_equivalent_to_original": vocab["content_equal"],
        "clean_bytes_equivalent_to_original": vocab["byte_equal"],
        "canonical_checkpoint_input_equivalent": provenance.get("canonical_input_content_equal", False),
        "clean_checkpoint_mode": clean.get("mode", "retrained"),
        "clean_checkpoint": clean.get("checkpoint"),
    }


def write_reports(output_root: Path, protocol: dict, rows: list[dict], validations: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "comparison.json").write_text(json.dumps({"protocol": protocol, "rows": rows}, indent=2), encoding="utf-8")
    (output_root / "checkpoint_validation.json").write_text(json.dumps({"rows": validations, "pass": True}, indent=2), encoding="utf-8")
    with (output_root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["dataset"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_root = safe_output_root(args.output_root)
    cache_root = args.cache_root.resolve()
    if CANONICAL_ROOT.resolve() == cache_root or CANONICAL_ROOT.resolve() in cache_root.parents:
        raise ValueError("Refusing to use canonical release as a cache write target")
    output_root.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    rows = []
    validations = []
    metadata = {
        "seed": 42,
        "datasets": args.datasets,
        "variants": args.variants,
        "device": args.device,
        "canonical_root": str(CANONICAL_ROOT),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "runs": [],
    }
    run_start = time.perf_counter()

    for dataset in args.datasets:
        dataset_cache = cache_root / dataset
        for variant in args.variants:
            original_path = ROOT / "data" / "concept_sets" / dataset / f"concepts_ablation_{variant}.txt"
            clean_path = ROOT / "data" / "concept_sets" / dataset / "clean" / f"concepts_ablation_{variant}.txt"
            canonical_checkpoint, canonical_args_path = canonical_paths(dataset, variant)
            canonical_args = json.loads(canonical_args_path.read_text(encoding="utf-8"))
            vocab = vocabulary_equivalence(original_path, clean_path)
            provenance = checkpoint_input_equivalence(original_path, canonical_args)
            original = model_summary(canonical_checkpoint, vocab["original_concepts"])
            output_dir = output_root / dataset / variant
            models_dir = output_dir / "models"
            complete = [] if args.restart else complete_model_dirs(models_dir)
            mode = "retrained"
            if complete:
                model_dir = complete[-1]
                print(f"[resume] {dataset}/{variant} -> {model_dir}", flush=True)
            elif dataset == "cremad" and vocab["content_equal"] and provenance.get("canonical_input_content_equal", False):
                model_dir = canonical_checkpoint
                mode = "reused_canonical_exact_equivalence"
                print(f"[reuse] {dataset}/{variant}: exact clean/original/checkpoint vocabulary equivalence", flush=True)
            else:
                if dataset == "cremad" and vocab["content_equal"] and not provenance.get("canonical_input_content_equal", False):
                    print(
                        f"[train] {dataset}/{variant}: clean equals audit original, but canonical checkpoint input is not equivalent",
                        flush=True,
                    )
                output_dir.mkdir(parents=True, exist_ok=True)
                train_args = make_train_args(canonical_args, dataset, variant, clean_path, output_dir, dataset_cache)
                train_args.device = args.device
                model_dir = Path(train_cbm.train_cbm_and_save(train_args))

            clean = model_summary(model_dir, vocab["clean_concepts"])
            clean["mode"] = mode
            validation = validate_checkpoint(model_dir, vocab["clean_concepts"], args.device)
            validation["dataset"] = dataset
            validation["variant"] = variant
            validations.append(validation)
            row = comparison_row(dataset, variant, original, clean, vocab, provenance)
            rows.append(row)
            metadata["runs"].append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "original_checkpoint": str(canonical_checkpoint),
                    "clean_checkpoint": str(model_dir),
                    "mode": mode,
                    "vocabulary": vocab,
                    "canonical_checkpoint_input": provenance,
                    "canonical_args": canonical_args,
                }
            )
            protocol = {"seed": 42, "canonical_settings_source": "release/google_drive_bundle/checkpoints/<dataset>/<variant>/args.txt"}
            write_reports(output_root, protocol, rows, validations)
            torch.cuda.empty_cache()

    metadata["runtime_seconds"] = time.perf_counter() - run_start
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    protocol = {
        "seed": 42,
        "canonical_settings_source": "release/google_drive_bundle/checkpoints/<dataset>/<variant>/args.txt",
        "test_split_used_only_for_final_evaluation": True,
        "ast_retrained": False,
        "clean_vocabularies": "data/concept_sets/<dataset>/clean/concepts_ablation_<variant>.txt",
    }
    write_reports(output_root, protocol, rows, validations)
    print(json.dumps({"runs": len(rows), "runtime_seconds": metadata["runtime_seconds"], "output_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
