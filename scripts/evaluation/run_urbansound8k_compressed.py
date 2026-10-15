#!/usr/bin/env python3
"""Run the compressed UrbanSound8K official-fold AST + LF+broad CV.

Each official test fold is trained once with the canonical cyclic validation
fold, followed by one LF+broad alpha-CBM fit.  Test data is used only after
validation-based checkpoint/model selection.  Existing complete fold outputs
are validated and skipped, making the runner resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score

import data_utils
from scripts.evaluation import task3_baselines as t3
from scripts.evaluation import task4_cv as t4


DATASET = "urbansound8k"
VARIANT = "lf_broad"
SEED = 42
N_FOLDS = 10
AST_EPOCHS = 30
AST_EFFECTIVE_BATCH_SIZE = 32


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected tensor at {}".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_backbone_name(repo_root: Path, checkpoint: Path) -> str:
    relative = checkpoint.resolve().relative_to(repo_root.resolve())
    return "ast_local__" + str(relative).replace("/", "__")


def _dataset(repo_root: Path, split: str, manifest_root: Path):
    path = manifest_root / "manifests" / (split + ".jsonl")
    return data_utils.get_audio_dataset(DATASET, split, manifest_path=str(path))


def _checkpoint_and_summary(run_dir: Path, fold: int) -> tuple[Path, Path]:
    checkpoint = run_dir / "ast" / DATASET / ("fold{}".format(fold)) / ("fold{}".format(fold)) / "best_model"
    return checkpoint, checkpoint.parent / "fold_summary.json"


def _train_ast(args: argparse.Namespace, repo_root: Path, run_dir: Path, fold: int) -> tuple[Path, dict[str, Any]]:
    output_dir = run_dir / "ast" / DATASET / ("fold{}".format(fold))
    checkpoint, summary_path = _checkpoint_and_summary(run_dir, fold)
    if not t4.checkpoint_complete(checkpoint, summary_path):
        train_split, val_split, test_split, val_fold = t4.fold_split_names(DATASET, fold)
        command = [
            sys.executable, "-m", "scripts.training.train_urbansound8k_ast",
            "--base_model_id", "MIT/ast-finetuned-audioset-10-10-0.4593",
            "--train_split", train_split, "--val_split", val_split, "--test_split", test_split,
            "--manifest_root", str(run_dir / "manifests"), "--output_dir", str(output_dir),
            "--epochs", str(AST_EPOCHS), "--batch_size", str(args.ast_batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--lr", "5e-5", "--weight_decay", "1e-4", "--warmup_ratio", "0.1",
            "--num_workers", str(args.num_workers), "--seed", str(SEED),
            "--device", args.device, "--run_name", "urbansound8k-compressed-ast",
            "--no_prepare_manifests",
        ]
        command_text = "CUDA_VISIBLE_DEVICES={} {}".format(
            os.environ.get("CUDA_VISIBLE_DEVICES", ""), " ".join(command)
        )
        print("AST command:", command_text, flush=True)
        (output_dir / "train_command.txt").parent.mkdir(parents=True, exist_ok=True)
        (output_dir / "train_command.txt").write_text(command_text + "\n", encoding="utf-8")
        with (output_dir / "train.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not t4.checkpoint_complete(checkpoint, summary_path):
        raise RuntimeError("Incomplete Urban AST checkpoint: {}".format(checkpoint))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = t4.fold_split_names(DATASET, fold)
    if (summary.get("train_split"), summary.get("val_split"), summary.get("test_split")) != expected[:3]:
        raise ValueError("AST checkpoint split provenance mismatch: {}".format(summary))
    return checkpoint, summary


def _valid_complete(fold_dir: Path, fold: int) -> bool:
    marker = fold_dir / "fold_complete.json"
    cbm = fold_dir / "cbm_lf_broad"
    required = [
        marker, cbm / "full.json", cbm / "full.pt", cbm / "labels.pt", cbm / "predictions.pt",
        fold_dir / "ast_predictions" / "labels.pt", fold_dir / "ast_predictions" / "predictions.pt",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return int(payload["test_fold"]) == fold and payload["variant"] == VARIANT
    except Exception:
        return False


def _run_fold(args: argparse.Namespace, repo_root: Path, run_dir: Path, fold: int) -> dict[str, Any]:
    fold_dir = run_dir / ("fold{}".format(fold))
    if _valid_complete(fold_dir, fold):
        print("Skipping verified complete fold {}".format(fold), flush=True)
        return json.loads((fold_dir / "fold_complete.json").read_text(encoding="utf-8"))

    start = time.time()
    train_split, val_split, test_split, val_fold = t4.fold_split_names(DATASET, fold)
    checkpoint, ast_summary = _train_ast(args, repo_root, run_dir, fold)
    local_backbone = _local_backbone_name(repo_root, checkpoint)
    feature_dir = fold_dir / "features"
    ast_paths = t4._extract_ast_features(
        repo_root, Path(local_backbone), DATASET, (train_split, val_split, test_split),
        run_dir / "manifests", feature_dir / "ast", device=args.device,
        batch_size=args.feature_batch_size,
    )

    datasets = {split: _dataset(repo_root, split, run_dir / "manifests") for split in (train_split, val_split, test_split)}
    labels = tuple(
        torch.tensor([int(sample["label_idx"]) for sample in datasets[split].samples], dtype=torch.long)
        for split in (train_split, val_split, test_split)
    )

    vocab_source = repo_root / "data/concept_sets/urbansound8k/concepts_ablation_lf_broad.txt"
    vocab_path = run_dir / "vocabularies" / DATASET / "lf_broad.txt"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    if vocab_path.exists() and _sha256(vocab_path) != _sha256(vocab_source):
        raise ValueError("Urban LF+broad vocabulary differs from canonical input")
    if not vocab_path.exists():
        shutil.copy2(vocab_source, vocab_path)

    clap_paths = t4._ensure_clap_features(
        repo_root, DATASET, (train_split, val_split, test_split), {VARIANT: vocab_path},
        run_dir / "manifests", feature_dir / "clap", device=args.device,
        batch_size=args.feature_batch_size,
    )
    features = t3.SplitFeatures(
        _load_tensor(ast_paths[train_split]).float(),
        _load_tensor(ast_paths[val_split]).float(),
        _load_tensor(ast_paths[test_split]).float(),
    )
    cbm_out = fold_dir / "cbm_lf_broad"
    full_json = cbm_out / "full.json"
    if not (full_json.exists() and (cbm_out / "full.pt").exists() and (cbm_out / "predictions.pt").exists()):
        paths = {
            "train": (ast_paths[train_split], clap_paths[VARIANT][train_split], clap_paths[VARIANT]["text"]),
            "val": (ast_paths[val_split], clap_paths[VARIANT][val_split], clap_paths[VARIANT]["text"]),
            "test": (ast_paths[test_split], clap_paths[VARIANT][test_split], clap_paths[VARIANT]["text"]),
        }
        reference = t3._reference_stats(repo_root / "release/google_drive_bundle/checkpoints/urbansound8k/lf_broad")
        cbm_args = {
            "device": args.device, "seed": SEED, "concept_activation_cutoff": 0.25,
            "interpretability_cutoff": 0.45, "projection_steps": 1000,
            "projection_batch_size": 50000, "saga_batch_size": 256,
            "saga_step_size": 0.1, "saga_iters": 1000, "saga_epsilon": 1e-4,
            "saga_path_points": 25, "saga_group": True, "saga_include_zero": False,
        }
        metrics_list = t3._run_concept_bottleneck(
            cbm_args, DATASET, {"name": VARIANT}, vocab_path, features, paths, cbm_out,
            reference["mean_nnz_per_class"], random_word=False, labels=labels,
        )
        selected = next(item for item in metrics_list if item["selection"] == "full")
        selected.update({"fold": fold, "val_fold": val_fold, "checkpoint_path": str(cbm_out / "full.pt")})
        _json_dump(full_json, selected)
        t4._save_selected_predictions(cbm_out / "full.pt", features.test, labels[2], cbm_out)

    ast_labels, ast_predictions = t4._extract_ast_predictions(
        repo_root, checkpoint, DATASET, test_split, run_dir / "manifests", fold_dir / "ast_predictions",
        device=args.device, batch_size=args.feature_batch_size,
    )
    ast_y = _load_tensor(ast_labels).long().numpy()
    ast_p = _load_tensor(ast_predictions).long().numpy()
    cbm_y = _load_tensor(cbm_out / "labels.pt").long().numpy()
    cbm_p = _load_tensor(cbm_out / "predictions.pt").long().numpy()
    if not (len(ast_y) == len(ast_p) == len(cbm_y) == len(cbm_p)):
        raise ValueError("AST and CBM test prediction lengths differ")
    ast_metrics = {
        "accuracy": float(accuracy_score(ast_y, ast_p)),
        "macro_f1": float(f1_score(ast_y, ast_p, average="macro", zero_division=0)),
    }
    cbm_metrics = json.loads(full_json.read_text(encoding="utf-8"))
    mcnemar = t4.paired_mcnemar(ast_y.tolist(), ast_p.tolist(), cbm_p.tolist())
    _json_dump(cbm_out / "mcnemar.json", mcnemar)

    runtime = time.time() - start
    metadata = {
        "dataset": DATASET, "test_fold": fold, "val_fold": val_fold,
        "train_split": train_split, "val_split": val_split, "test_split": test_split,
        "variant": VARIANT, "seed": SEED, "selection_split": "val",
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_workers": args.num_workers, "ast_batch_size": args.ast_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ast_effective_batch_size": args.ast_batch_size * args.gradient_accumulation_steps,
        "feature_batch_size": args.feature_batch_size, "ast_epochs": AST_EPOCHS,
        "ast_checkpoint": str(checkpoint), "ast_summary": ast_summary,
        "vocabulary_path": str(vocab_path), "vocabulary_sha256": _sha256(vocab_path),
        "ast_test_metrics": ast_metrics, "cbm_test_metrics": cbm_metrics,
        "runtime_seconds": runtime, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(), "command": " ".join(sys.argv),
    }
    _json_dump(fold_dir / "run_metadata.json", metadata)
    complete = {
        "dataset": DATASET, "test_fold": fold, "val_fold": val_fold, "variant": VARIANT,
        "ast_checkpoint": str(checkpoint), "ast_summary": ast_summary,
        "ast_test_metrics": ast_metrics, "cbm_output": str(cbm_out),
        "cbm_test_metrics": cbm_metrics, "mcnemar": mcnemar,
        "runtime_seconds": runtime, "git_commit": metadata["git_commit"],
    }
    _json_dump(fold_dir / "fold_complete.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True), flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1))
    group.add_argument("--folds", type=int, nargs="+", choices=range(1, N_FOLDS + 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ast-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.ast_batch_size * args.gradient_accumulation_steps != AST_EFFECTIVE_BATCH_SIZE:
        parser.error("physical batch x accumulation must preserve effective batch size 32")
    repo_root = args.repo_root.resolve()
    run_dir = repo_root / "results/revision/t4_cv/urbansound8k_official_cv"
    run_dir.mkdir(parents=True, exist_ok=True)
    t4.ensure_urban_cv_manifests(repo_root, run_dir)
    folds = [args.fold] if args.fold is not None else args.folds
    protocol = {
        "dataset": DATASET, "variant": VARIANT, "seed": SEED,
        "official_test_folds": list(range(1, N_FOLDS + 1)),
        "validation_rule": "next cyclic official fold", "selection_split": "val",
        "ast_epochs": AST_EPOCHS, "ast_base_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        "ast_batch_size": args.ast_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ast_effective_batch_size": args.ast_batch_size * args.gradient_accumulation_steps,
        "num_workers": args.num_workers, "feature_batch_size": args.feature_batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_folds": folds, "command": " ".join(sys.argv),
    }
    _json_dump(run_dir / "protocol.json", protocol)
    for fold in folds:
        _run_fold(args, repo_root, run_dir, fold)


if __name__ == "__main__":
    main()
