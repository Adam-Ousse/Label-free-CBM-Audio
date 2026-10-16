#!/usr/bin/env python3
"""Run the compressed ESC-50 official-fold LF+broad evaluation.

This runner is intentionally narrower than the historical Task-4 runner:
one AST fine-tune per missing fold, one AST feature cache per split, and one
LF+broad alpha-CBM fit.  All projection/head choices are validation-only;
the test split is read only after the selected validation path entry exists.
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

import data_utils
from scripts.evaluation import task3_baselines as t3
from scripts.evaluation import task4_cv as t4


VARIANT = "lf_broad"
SEED = 42
N_FOLDS = 5


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            raise FileExistsError("Refusing to replace existing directory: {}".format(path))
    path.symlink_to(target.resolve())


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError("Expected tensor at {}".format(path))
    return value.float()


def _local_backbone_name(repo_root: Path, checkpoint: Path) -> str:
    """Encode a repository-local AST checkpoint for data_utils' loader."""

    relative = checkpoint.resolve().relative_to(repo_root.resolve())
    return "ast_local__" + str(relative).replace("/", "__")


def _manifest_dataset(repo_root: Path, split: str):
    path = repo_root / "data/esc50/manifests" / (split + ".jsonl")
    return data_utils.get_audio_dataset("esc50", split, manifest_path=str(path))


def _reuse_fold1(repo_root: Path, run_dir: Path) -> dict[str, Any]:
    """Register exact pre-existing canonical fold-1 artifacts read-only."""

    fold_dir = run_dir / "fold1"
    feature_source = repo_root / "results/revision/t3_baselines/t3_gpu_01_fix/cache/esc50"
    cbm_source = repo_root / "results/revision/t4_cv/t4_gpu_01/per_seed/esc50/lf_broad/seed42"
    vocab_source = repo_root / "data/concept_sets/esc50/concepts_ablation_lf_broad.txt"
    required_features = {
        "fold1_train": feature_source / "fold1_train_backbone_ast_esc50_layer4.pt",
        "fold1_val": feature_source / "fold1_val_backbone_ast_esc50_layer4.pt",
        "fold1_test": feature_source / "fold1_test_backbone_ast_esc50_layer4.pt",
    }
    required_cbm = [cbm_source / name for name in ("full.pt", "full.json", "predictions.pt", "labels.pt")]
    missing = [str(path) for path in [vocab_source, *required_features.values(), *required_cbm] if not path.exists()]
    if missing:
        raise FileNotFoundError("Fold-1 reuse artifacts missing: {}".format(", ".join(missing)))

    expected_rows = {"fold1_train": 1200, "fold1_val": 400, "fold1_test": 400}
    feature_shapes = {}
    for split, path in required_features.items():
        tensor = _load_tensor(path)
        if tuple(tensor.shape) != (expected_rows[split], 768):
            raise ValueError("Fold-1 AST feature shape mismatch for {}: {}".format(split, tuple(tensor.shape)))
        feature_shapes[split] = list(tensor.shape)

    # Official fold-1 protocol: folds 1 and 2 are held out; fold 2 is val.
    split_names = ("fold1_train", "fold1_val", "fold1_test")
    split_counts = {split: len(_manifest_dataset(repo_root, split)) for split in split_names}
    if split_counts != expected_rows:
        raise ValueError("Fold-1 manifest counts are not official: {}".format(split_counts))
    if _sha256(vocab_source) != _sha256(repo_root / "results/revision/t4_cv/t4_gpu_01/fixed_split/vocabularies/esc50/lf_broad.txt"):
        raise ValueError("Existing fold-1 LF+broad vocabulary is not byte-identical to canonical input")

    for split, source in required_features.items():
        _link(fold_dir / "features" / "ast" / (split + ".pt"), source)
    _link(fold_dir / "cbm_lf_broad", cbm_source)
    _json_dump(fold_dir / "ast_checkpoint_provenance.json", {
        "reused": True,
        "checkpoint_id": "Adam-ousse/ast-esc50-finetuned-fold1",
        "checkpoint_source": "existing canonical fold-1 AST feature cache",
        "protocol_compatible": True,
        "label_count": 50,
        "feature_layer": "layer4",
        "feature_dim": 768,
        "splits": split_names,
        "split_counts": split_counts,
    })
    summary = json.loads((cbm_source / "full.json").read_text(encoding="utf-8"))
    complete = {
        "dataset": "esc50",
        "test_fold": 1,
        "val_fold": 2,
        "variant": VARIANT,
        "reused": True,
        "ast_feature_shapes": feature_shapes,
        "cbm_source": str(cbm_source),
        "metrics": summary,
    }
    _json_dump(fold_dir / "fold_complete.json", complete)
    return complete


def _train_ast(args: argparse.Namespace, repo_root: Path, run_dir: Path, fold: int) -> tuple[Path, dict[str, Any]]:
    output_dir = run_dir / "ast" / "esc50" / ("fold{}".format(fold))
    checkpoint = output_dir / ("fold{}".format(fold)) / "best_model"
    summary_path = checkpoint.parent / "fold_summary.json"
    if not t4.checkpoint_complete(checkpoint, summary_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "scripts.training.train_esc50_ast",
            "--base_model_id", "MIT/ast-finetuned-audioset-10-10-0.4593",
            "--esc50_root", "data/esc50/raw/ESC-50-master",
            "--manifest_root", "data/esc50",
            "--output_dir", str(output_dir),
            "--test_fold", str(fold), "--val_fold", str((fold % N_FOLDS) + 1),
            "--epochs", "30", "--batch_size", str(args.ast_batch_size), "--lr", "5e-5",
            "--weight_decay", "1e-4", "--warmup_ratio", "0.1",
            "--num_workers", str(args.num_workers), "--seed", str(SEED),
            "--device", args.device, "--run_name", "esc50-compressed-ast",
            "--no_prepare_manifests",
        ]
        if args.ast_input_cache is not None:
            command.extend(["--ast-input-cache", str(args.ast_input_cache)])
        if args.resume_from is not None:
            if fold != 3:
                raise ValueError("--resume-from is currently supported only for fold 3")
            command.extend(["--resume_from", str(args.resume_from), "--resume_epoch", str(args.resume_epoch)])
        print("AST command: CUDA_VISIBLE_DEVICES={} {}".format(os.environ.get("CUDA_VISIBLE_DEVICES", ""), " ".join(command)), flush=True)
        with (output_dir / "train.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not t4.checkpoint_complete(checkpoint, summary_path):
        raise RuntimeError("Incomplete AST fold checkpoint: {}".format(checkpoint))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary["test_fold"]) != fold or int(summary["val_fold"]) != (fold % N_FOLDS) + 1:
        raise ValueError("AST checkpoint fold provenance mismatch: {}".format(summary))
    return checkpoint, summary


def _run_fold(args: argparse.Namespace, repo_root: Path, run_dir: Path, fold: int) -> dict[str, Any]:
    if fold == 1:
        return _reuse_fold1(repo_root, run_dir)

    start = time.time()
    train_split, val_split, test_split, val_fold = t4.fold_split_names("esc50", fold)
    checkpoint, ast_summary = _train_ast(args, repo_root, run_dir, fold)
    fold_dir = run_dir / "fold{}".format(fold)
    feature_dir = fold_dir / "features"
    ast_paths = t4._extract_ast_features(
        repo_root, Path(_local_backbone_name(repo_root, checkpoint)), "esc50",
        (train_split, val_split, test_split), None,
        feature_dir / "ast", device=args.device, batch_size=args.feature_batch_size,
        ast_input_cache=args.ast_input_cache,
    )
    datasets = {split: _manifest_dataset(repo_root, split) for split in (train_split, val_split, test_split)}
    labels = tuple(
        torch.tensor([int(sample["label_idx"]) for sample in datasets[split].samples], dtype=torch.long)
        for split in (train_split, val_split, test_split)
    )
    vocab_source = repo_root / "data/concept_sets/esc50/concepts_ablation_lf_broad.txt"
    vocab_path = run_dir / "vocabularies" / "esc50" / "lf_broad.txt"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    if vocab_path.exists() and _sha256(vocab_path) != _sha256(vocab_source):
        raise ValueError("Existing compressed-run LF+broad vocabulary differs from canonical input")
    if not vocab_path.exists():
        shutil.copy2(vocab_source, vocab_path)
    clap_paths = t4._ensure_clap_features(
        repo_root, "esc50", (train_split, val_split, test_split), {VARIANT: vocab_path}, None,
        feature_dir / "clap", device=args.device, batch_size=args.feature_batch_size,
    )
    features = t3.SplitFeatures(
        _load_tensor(ast_paths[train_split]), _load_tensor(ast_paths[val_split]), _load_tensor(ast_paths[test_split])
    )
    out = fold_dir / "cbm_lf_broad"
    full_json = out / "full.json"
    if not (full_json.exists() and (out / "full.pt").exists() and (out / "predictions.pt").exists()):
        paths = {
            "train": (ast_paths[train_split], clap_paths[VARIANT][train_split], clap_paths[VARIANT]["text"]),
            "val": (ast_paths[val_split], clap_paths[VARIANT][val_split], clap_paths[VARIANT]["text"]),
            "test": (ast_paths[test_split], clap_paths[VARIANT][test_split], clap_paths[VARIANT]["text"]),
        }
        reference = t3._reference_stats(repo_root / "release/google_drive_bundle/checkpoints/esc50/lf_broad")
        cbm_args = {
            "device": args.device, "seed": SEED, "concept_activation_cutoff": 0.25,
            "interpretability_cutoff": 0.45, "projection_steps": 1000,
            "projection_batch_size": 50000, "saga_batch_size": 256,
            "saga_step_size": 0.1, "saga_iters": 1000, "saga_epsilon": 1e-4,
            "saga_path_points": 25, "saga_group": True, "saga_include_zero": False,
        }
        metrics_list = t3._run_concept_bottleneck(
            cbm_args, "esc50", {"name": VARIANT}, vocab_path, features, paths, out,
            reference["mean_nnz_per_class"], random_word=False, labels=labels,
        )
        selected = next(item for item in metrics_list if item["selection"] == "full")
        selected.update({"fold": fold, "val_fold": val_fold, "checkpoint_path": str(out / "full.pt")})
        _json_dump(full_json, selected)
        t4._save_selected_predictions(out / "full.pt", features.test, labels[2], out)
    metrics = json.loads(full_json.read_text(encoding="utf-8"))
    ast_labels, ast_predictions = t4._extract_ast_predictions(
        repo_root, checkpoint, "esc50", test_split, None, fold_dir / "ast_predictions",
        device=args.device, batch_size=32, ast_input_cache=args.ast_input_cache,
    )
    cbm_predictions = _load_tensor(out / "predictions.pt").long()
    mcnemar = t4.paired_mcnemar(
        _load_tensor(ast_labels).long().tolist(), _load_tensor(ast_predictions).long().tolist(), cbm_predictions.tolist()
    )
    _json_dump(out / "mcnemar.json", mcnemar)
    metadata = {
        "dataset": "esc50", "test_fold": fold, "val_fold": val_fold, "train_split": train_split,
        "variant": VARIANT, "seed": SEED, "selection_split": "val",
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_workers": args.num_workers, "ast_batch_size": args.ast_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "ast_checkpoint": str(checkpoint), "ast_summary": ast_summary,
        "vocabulary_path": str(vocab_path), "runtime_seconds": time.time() - start,
        "command": " ".join(sys.argv),
    }
    _json_dump(fold_dir / "run_metadata.json", metadata)
    complete = {
        "dataset": "esc50", "test_fold": fold, "val_fold": val_fold, "variant": VARIANT,
        "ast_checkpoint": str(checkpoint), "ast_summary": ast_summary,
        "feature_paths": {key: str(value) for key, value in ast_paths.items()},
        "cbm_output": str(out), "metrics": metrics, "mcnemar": mcnemar,
        "runtime_seconds": time.time() - start,
    }
    _json_dump(fold_dir / "fold_complete.json", complete)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", default="esc50_compressed_gpu1")
    parser.add_argument("--fold", type=int, choices=range(1, N_FOLDS + 1), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ast-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--ast-input-cache", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--resume-epoch", type=int, default=0)
    args = parser.parse_args()
    if args.resume_from is not None:
        if args.fold != 3:
            parser.error("--resume-from is currently supported only for fold 3")
        if not args.resume_from.is_dir():
            parser.error("--resume-from directory does not exist: {}".format(args.resume_from))
        if args.resume_epoch < 1 or args.resume_epoch >= 30:
            parser.error("--resume-epoch must be in [1, 29]")
    repo_root = args.repo_root.resolve()
    run_dir = repo_root / "results/revision" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "esc50", "variant": VARIANT, "seed": SEED, "official_test_folds": list(range(1, 6)),
        "validation_rule": "next cyclic official fold", "selection_split": "val",
        "new_ast_folds": [2, 3, 4, 5], "fold1_reused": True,
        "urban_sound8k_started": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ast_input_cache": str(args.ast_input_cache) if args.ast_input_cache else None,
        "command": " ".join(sys.argv),
    }
    _json_dump(run_dir / "protocol.json", protocol)
    result = _run_fold(args, repo_root, run_dir, args.fold)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
