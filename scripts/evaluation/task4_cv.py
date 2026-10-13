#!/usr/bin/env python3
"""Task 4 multi-seed robustness and official-fold cross-validation runner.

This runner deliberately reuses the Task 3 projection, standardisation and
glm_saga helpers.  It adds only the fold orchestration, local AST checkpoint
feature extraction, aligned prediction storage, and resumable reporting needed
for Task 4.  All selection is validation-only; test predictions are written
only after a path/model has been selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_utils
import utils
from clap import core as clap_core
from models.ast_classifier import ASTAudioClassifier
from scripts.evaluation import task3_baselines as t3


SEEDS = (0, 1, 2, 42, 123)
CV_DATASETS = ("esc50", "urbansound8k")
ALL_DATASETS = ("esc50", "urbansound8k", "cremad")
CBM_VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full", "broad_plus_contrastive_clean")
CV_VARIANTS = ("lf", "lf_broad", "broad_plus_contrastive_clean")
NNZ_THRESHOLD = t3.NNZ_THRESHOLD


def fold_split_names(dataset: str, test_fold: int) -> tuple[str, str, str, int]:
    """Return train/validation/test split names using the canonical next-fold rule."""

    dataset = str(dataset)
    test_fold = int(test_fold)
    n_folds = 5 if dataset == "esc50" else 10 if dataset == "urbansound8k" else 0
    if n_folds == 0 or not 1 <= test_fold <= n_folds:
        raise ValueError("Unsupported fold {} for {}".format(test_fold, dataset))
    val_fold = (test_fold % n_folds) + 1
    return (
        "fold{}_train".format(test_fold),
        "fold{}_val".format(test_fold),
        "fold{}_test".format(test_fold),
        val_fold,
    )


def paired_mcnemar(y_true: Sequence[int], reference_pred: Sequence[int], cbm_pred: Sequence[int]) -> dict[str, Any]:
    """Compute exact paired McNemar statistics for aligned predictions."""

    truth = np.asarray(y_true, dtype=np.int64)
    reference = np.asarray(reference_pred, dtype=np.int64)
    cbm = np.asarray(cbm_pred, dtype=np.int64)
    if not (truth.shape == reference.shape == cbm.shape):
        raise ValueError("McNemar inputs must have identical shapes")
    reference_correct = reference == truth
    cbm_correct = cbm == truth
    both_correct = int(np.sum(reference_correct & cbm_correct))
    cbm_only = int(np.sum(~reference_correct & cbm_correct))
    reference_only = int(np.sum(reference_correct & ~cbm_correct))
    both_wrong = int(np.sum(~reference_correct & ~cbm_correct))
    discordant = cbm_only + reference_only
    if discordant == 0:
        p_value = 1.0
    else:
        # Two-sided exact binomial test under p=0.5.
        tail = min(cbm_only, reference_only)
        p_value = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2.0 ** discordant))
    return {
        "n": int(truth.size),
        "both_correct": both_correct,
        "cbm_correct_ast_wrong": cbm_only,
        "ast_correct_cbm_wrong": reference_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "exact_p_value": float(p_value),
    }


def checkpoint_complete(checkpoint_dir: Path, summary_path: Path) -> bool:
    """Validate a locally produced AST checkpoint before allowing a resume skip."""

    model_file = checkpoint_dir / "model.safetensors"
    pytorch_file = checkpoint_dir / "pytorch_model.bin"
    return bool(
        checkpoint_dir.is_dir()
        and (model_file.exists() or pytorch_file.exists())
        and (checkpoint_dir / "config.json").exists()
        and (checkpoint_dir / "preprocessor_config.json").exists()
        and summary_path.exists()
    )


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).float()


def _valid_tensor(path: Path, expected_rows: int | None = None, ndim: int = 2) -> bool:
    if not path.exists():
        return False
    try:
        tensor = _load_tensor(path)
    except Exception:
        return False
    return tensor.ndim == ndim and (expected_rows is None or int(tensor.shape[0]) == int(expected_rows))


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def ensure_urban_cv_manifests(repo_root: Path, out_root: Path) -> None:
    """Materialise all official UrbanSound8K train/next-val/test manifests."""

    source = repo_root / "data/urbansound8k/manifests/all.jsonl"
    if not source.exists():
        raise FileNotFoundError("Missing prepared UrbanSound8K manifest: {}".format(source))
    rows = _manifest_rows(source)
    for test_fold in range(1, 11):
        _, _, _, val_fold = fold_split_names("urbansound8k", test_fold)
        _write_jsonl(
            out_root / "manifests" / "fold{}_train.jsonl".format(test_fold),
            (row for row in rows if int(row["fold"]) not in {test_fold, val_fold}),
        )
        _write_jsonl(
            out_root / "manifests" / "fold{}_val.jsonl".format(test_fold),
            (row for row in rows if int(row["fold"]) == val_fold),
        )
        _write_jsonl(
            out_root / "manifests" / "fold{}_test.jsonl".format(test_fold),
            (row for row in rows if int(row["fold"]) == test_fold),
        )
    for name in ("label_to_idx.json", "idx_to_label.json"):
        source_map = repo_root / "data/urbansound8k" / name
        target_map = out_root / name
        target_map.parent.mkdir(parents=True, exist_ok=True)
        if not target_map.exists():
            shutil.copy2(source_map, target_map)
    _json_dump(out_root / "manifest_protocol.json", {
        "dataset": "urbansound8k",
        "validation_rule": "next cyclic official fold",
        "folds": list(range(1, 11)),
    })


def _dataset_manifest_root(repo_root: Path, dataset: str, manifest_root: Path | None = None) -> Path:
    if manifest_root is not None:
        return manifest_root
    return repo_root / "data" / dataset


def _dataset_for_split(repo_root: Path, dataset: str, split: str, manifest_root: Path | None = None):
    path = _dataset_manifest_root(repo_root, dataset, manifest_root) / "manifests" / (split + ".jsonl")
    return data_utils.get_audio_dataset(dataset, split, manifest_path=str(path))


def _ast_checkpoint_path(ast_dir: Path, dataset: str, fold: int) -> Path:
    return ast_dir / dataset / ("fold{}".format(fold)) / ("fold{}".format(fold)) / "best_model"


def ensure_ast_fold(
    repo_root: Path,
    out_root: Path,
    dataset: str,
    fold: int,
    *,
    device: str,
    urbansound_manifest_root: Path,
    urban_epochs: int,
) -> tuple[Path, dict[str, Any]]:
    """Run the existing AST training script once per official fold, resumably."""

    output_dir = out_root / "ast" / dataset / "fold{}".format(fold)
    checkpoint = _ast_checkpoint_path(out_root / "ast", dataset, fold)
    summary_path = checkpoint.parent / "fold_summary.json"
    if not checkpoint_complete(checkpoint, summary_path):
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train.log"
        if dataset == "esc50":
            train_script = repo_root / "scripts/training/train_esc50_ast.py"
            command = [
                sys.executable, str(train_script),
                "--test_fold", str(fold), "--val_fold", str(fold_split_names(dataset, fold)[3]),
                "--output_dir", str(output_dir), "--device", device, "--seed", "42",
                "--no_prepare_manifests", "--num_workers", "8",
            ]
        else:
            train_script = repo_root / "scripts/training/train_urbansound8k_ast.py"
            train_split, val_split, test_split, _ = fold_split_names(dataset, fold)
            command = [
                sys.executable, str(train_script),
                "--train_split", train_split, "--val_split", val_split, "--test_split", test_split,
                "--manifest_root", str(urbansound_manifest_root), "--output_dir", str(output_dir),
                "--device", device, "--seed", "42", "--epochs", str(urban_epochs),
                "--num_workers", "8", "--no_prepare_manifests",
            ]
        print("AST fold command:", " ".join(command), flush=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            subprocess.run(command, cwd=repo_root, stdout=log_handle, stderr=subprocess.STDOUT, check=True)
    if not checkpoint_complete(checkpoint, summary_path):
        raise RuntimeError("AST fold did not produce a complete checkpoint: {}".format(checkpoint))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _json_dump(output_dir / "complete.json", {"checkpoint": str(checkpoint), "summary": summary})
    return checkpoint, summary


def _extract_ast_features(
    repo_root: Path,
    checkpoint: Path,
    dataset: str,
    splits: tuple[str, str, str],
    manifest_root: Path | None,
    feature_root: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, Path]:
    model, _ = data_utils.get_target_model(str(checkpoint), device)
    paths: dict[str, Path] = {}
    for split in splits:
        ds = _dataset_for_split(repo_root, dataset, split, manifest_root)
        path = feature_root / (split + ".pt")
        if not _valid_tensor(path, expected_rows=len(ds)):
            if path.exists():
                path.unlink()
            utils.save_backbone_audio_features(model, ds, str(path), batch_size=batch_size, device=device)
        if not _valid_tensor(path, expected_rows=len(ds)):
            raise RuntimeError("Invalid AST feature cache: {}".format(path))
        paths[split] = path
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths


def _extract_ast_predictions(
    repo_root: Path,
    checkpoint: Path,
    dataset: str,
    test_split: str,
    manifest_root: Path | None,
    out_dir: Path,
    *,
    device: str,
    batch_size: int,
) -> tuple[Path, Path]:
    ds = _dataset_for_split(repo_root, dataset, test_split, manifest_root)
    labels_path = out_dir / "labels.pt"
    predictions_path = out_dir / "predictions.pt"
    if _valid_tensor(labels_path, expected_rows=len(ds), ndim=1) and _valid_tensor(predictions_path, expected_rows=len(ds), ndim=1):
        return labels_path, predictions_path
    classifier = ASTAudioClassifier(str(checkpoint), device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"), collate_fn=data_utils.collate_audio_batch)
    labels: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            logits = classifier.predict_logits(batch["audio"], sample_rates=batch["sr"])
            labels.append(batch["target"].long().cpu())
            predictions.append(torch.argmax(logits, dim=1).long().cpu())
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(labels), labels_path)
    torch.save(torch.cat(predictions), predictions_path)
    return labels_path, predictions_path


def _ensure_clap_features(
    repo_root: Path,
    dataset: str,
    splits: tuple[str, str, str],
    vocab_paths: dict[str, Path],
    manifest_root: Path | None,
    out_dir: Path,
    *,
    device: str,
    batch_size: int,
) -> dict[str, dict[str, Path]]:
    """Create one CLAP audio cache per split and one text cache per vocabulary."""

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = clap_core.load_clap_model(model_name=t3.DEFAULT_CLAP_MODEL, device=device)
    audio_paths: dict[str, Path] = {}
    for split in splits:
        ds = _dataset_for_split(repo_root, dataset, split, manifest_root)
        path = out_dir / (split + "_clap_audio.pt")
        if not _valid_tensor(path, expected_rows=len(ds)):
            if path.exists():
                path.unlink()
            utils.save_clap_audio_features(bundle, ds, str(path), batch_size=batch_size)
        if not _valid_tensor(path, expected_rows=len(ds)):
            raise RuntimeError("Invalid CLAP audio cache: {}".format(path))
        audio_paths[split] = path
    result: dict[str, dict[str, Path]] = {}
    for name, vocab_path in vocab_paths.items():
        concepts = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        text_path = out_dir / (name + "_clap_text.pt")
        if not _valid_tensor(text_path, expected_rows=len(concepts)):
            if text_path.exists():
                text_path.unlink()
            utils.save_clap_text_features(bundle, concepts, str(text_path), batch_size=batch_size)
        if not _valid_tensor(text_path, expected_rows=len(concepts)):
            raise RuntimeError("Invalid CLAP text cache: {}".format(text_path))
        result[name] = {"text": text_path, **{split: audio_paths[split] for split in splits}}
    del bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _variant_vocabularies(repo_root: Path, dataset: str, out_dir: Path) -> dict[str, Path]:
    """Copy exact Task 2 input vocabularies into the isolated Task 4 run."""

    result: dict[str, Path] = {}
    for name in ("lf", "lf_broad", "lf_contrastive", "full"):
        source = repo_root / "data/concept_sets" / dataset / ("concepts_ablation_{}.txt".format(name))
        target = out_dir / dataset / (name + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        result[name] = target
    source = repo_root / "data/concept_sets" / dataset / "clean" / "concepts_ablation_full.txt"
    target = out_dir / dataset / "broad_plus_contrastive_clean.txt"
    if not target.exists():
        shutil.copy2(source, target)
    result["broad_plus_contrastive_clean"] = target
    _json_dump(out_dir / dataset / "manifest.json", {
        "dataset": dataset,
        "variants": {name: {"path": str(path), "count": len(path.read_text(encoding="utf-8").splitlines())} for name, path in result.items()},
        "clean_candidate": "Task 2 clean concepts_ablation_full.txt",
    })
    return result


def _fixed_paths(run3_dir: Path, dataset: str, variant: str, text_path: Path) -> tuple[t3.SplitFeatures, dict[str, tuple[Path, Path, Path]]]:
    spec = t3.DATASET_SPECS[dataset]
    cache = run3_dir / "cache" / dataset
    paths: dict[str, tuple[Path, Path, Path]] = {}
    tensors: dict[str, torch.Tensor] = {}
    for key in ("train", "val", "test"):
        split = spec[key]
        target = cache / (split + "_backbone_{}_layer4.pt".format(spec["ast_backbone"]))
        clap_audio = cache / (split + "_clap_audio_laion_clap-htsat-unfused.pt")
        if not target.exists() or not clap_audio.exists():
            raise FileNotFoundError("Missing Task 3 fixed-split cache for {} {}: {} {}".format(dataset, key, target, clap_audio))
        paths[key] = (target, clap_audio, text_path)
        tensors[key] = _load_tensor(target)
    return t3.SplitFeatures(tensors["train"], tensors["val"], tensors["test"]), paths


def _save_selected_predictions(checkpoint_path: Path, test_features: torch.Tensor, labels: torch.Tensor, out_dir: Path) -> tuple[Path, Path]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    projected = test_features.float()
    if "W_c" in payload:
        projected = torch.nn.functional.linear(projected, payload["W_c"].float())
    if "proj_mean" in payload:
        projected = (projected - payload["proj_mean"].float()) / payload["proj_std"].float().clamp_min(1e-6)
    logits = torch.nn.functional.linear(projected, payload["W_g"].float(), payload["b_g"].float())
    predictions = torch.argmax(logits, dim=1).long()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.pt"
    predictions_path = out_dir / "predictions.pt"
    torch.save(labels.long().cpu(), labels_path)
    torch.save(predictions.cpu(), predictions_path)
    return labels_path, predictions_path


def _run_fixed_seed_robustness(repo_root: Path, run_dir: Path, run3_dir: Path, *, device: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_root = run_dir / "fixed_split"
    vocab_root = fixed_root / "vocabularies"
    for dataset in ALL_DATASETS:
        vocab_paths = _variant_vocabularies(repo_root, dataset, vocab_root)
        # One text embedding per variant; the Task 3 audio/AST caches are reused read-only.
        text_cache = fixed_root / "clap_text" / dataset
        bundle = clap_core.load_clap_model(model_name=t3.DEFAULT_CLAP_MODEL, device=device)
        text_paths: dict[str, Path] = {}
        for name, vocab_path in vocab_paths.items():
            path = text_cache / (name + ".pt")
            concepts = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not _valid_tensor(path, expected_rows=len(concepts)):
                if path.exists():
                    path.unlink()
                utils.save_clap_text_features(bundle, concepts, str(path), batch_size=64)
            text_paths[name] = path
        del bundle
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ast_pred_dir = fixed_root / "ast" / dataset
        spec = t3.DATASET_SPECS[dataset]
        if dataset == "esc50":
            checkpoint = repo_root / "saved_models/ast_esc50/fold1/best_model"
            if not checkpoint.exists():
                checkpoint = Path("Adam-ousse/ast-esc50-finetuned-fold1")
        elif dataset == "urbansound8k":
            checkpoint = repo_root / "saved_models/ast_urbansound8k/fold10/best_model"
            if not checkpoint.exists():
                checkpoint = Path("Adam-ousse/ast-urbansound8k-finetuned-fold10")
        else:
            checkpoint = Path("Adam-ousse/ast-cremad-finetuned")
        labels_path, ast_predictions_path = _extract_ast_predictions(repo_root, checkpoint, dataset, spec["test"], None, ast_pred_dir, device=device, batch_size=32)
        labels = _load_tensor(labels_path).long()
        ast_predictions = _load_tensor(ast_predictions_path).long()
        for variant in CBM_VARIANTS:
            vocab_path = vocab_paths[variant]
            features, paths = _fixed_paths(run3_dir, dataset, variant, text_paths[variant])
            reference_variant = "full" if variant == "broad_plus_contrastive_clean" else variant
            reference = t3._reference_stats(repo_root / "release/google_drive_bundle/checkpoints" / dataset / reference_variant)
            for seed in SEEDS:
                out = run_dir / "per_seed" / dataset / variant / ("seed{}".format(seed))
                full_json = out / "full.json"
                pred_path = out / "predictions.pt"
                if not (full_json.exists() and pred_path.exists()):
                    args = {
                        "device": device,
                        "seed": seed,
                        "concept_activation_cutoff": 0.25,
                        "interpretability_cutoff": 0.45,
                        "projection_steps": 1000,
                        "projection_batch_size": 50000,
                        "saga_batch_size": 256,
                        "saga_step_size": 0.1,
                        "saga_iters": 1000,
                        "saga_epsilon": 1e-4,
                        "saga_path_points": 25,
                        "saga_group": True,
                        "saga_include_zero": False,
                    }
                    metrics_list = t3._run_concept_bottleneck(
                        args, dataset, {"name": variant}, vocab_path, features, paths, out,
                        reference["mean_nnz_per_class"], random_word=False,
                    )
                    selected = next(item for item in metrics_list if item["selection"] == "full")
                    selected["seed"] = seed
                    selected["checkpoint_path"] = str(out / "full.pt")
                    _json_dump(full_json, selected)
                    _save_selected_predictions(out / "full.pt", features.test, labels, out)
                metrics = json.loads(full_json.read_text(encoding="utf-8"))
                metrics.update({"seed": seed, "dataset": dataset, "variant": variant, "prediction_path": str(pred_path)})
                tests = paired_mcnemar(labels.tolist(), ast_predictions.tolist(), _load_tensor(pred_path).long().tolist())
                rows.append(metrics)
                _json_dump(out / "mcnemar.json", tests)
    return rows


def _run_cv_fold(
    repo_root: Path,
    run_dir: Path,
    dataset: str,
    fold: int,
    *,
    device: str,
    urban_manifest_root: Path,
    urban_epochs: int,
) -> list[dict[str, Any]]:
    start = time.time()
    train_split, val_split, test_split, val_fold = fold_split_names(dataset, fold)
    manifest_root = None if dataset == "esc50" else urban_manifest_root
    checkpoint, ast_summary = ensure_ast_fold(
        repo_root, run_dir, dataset, fold, device=device,
        urbansound_manifest_root=urban_manifest_root, urban_epochs=urban_epochs,
    )
    feature_dir = run_dir / "features" / dataset / ("fold{}".format(fold))
    ast_features = _extract_ast_features(
        repo_root, checkpoint, dataset, (train_split, val_split, test_split), manifest_root,
        feature_dir / "ast", device=device, batch_size=64,
    )
    labels = {}
    for name, split in (("train", train_split), ("val", val_split), ("test", test_split)):
        ds = _dataset_for_split(repo_root, dataset, split, manifest_root)
        labels[name] = torch.tensor([int(sample["label_idx"]) for sample in ds.samples], dtype=torch.long)
    ast_label_path, ast_pred_path = _extract_ast_predictions(
        repo_root, checkpoint, dataset, test_split, manifest_root,
        run_dir / "per_fold" / dataset / ("fold{}".format(fold)) / "ast",
        device=device, batch_size=32,
    )
    vocab_paths = _variant_vocabularies(repo_root, dataset, run_dir / "vocabularies")
    clap_paths = _ensure_clap_features(
        repo_root, dataset, (train_split, val_split, test_split), vocab_paths, manifest_root,
        feature_dir / "clap", device=device, batch_size=64,
    )
    features = t3.SplitFeatures(_load_tensor(ast_features[train_split]), _load_tensor(ast_features[val_split]), _load_tensor(ast_features[test_split]))
    results: list[dict[str, Any]] = []
    for variant in CV_VARIANTS:
        out = run_dir / "per_fold" / dataset / ("fold{}".format(fold)) / variant
        if (out / "full.json").exists() and (out / "predictions.pt").exists():
            metrics = json.loads((out / "full.json").read_text(encoding="utf-8"))
        else:
            paths = {
                "train": (ast_features[train_split], clap_paths[variant][train_split], clap_paths[variant]["text"]),
                "val": (ast_features[val_split], clap_paths[variant][val_split], clap_paths[variant]["text"]),
                "test": (ast_features[test_split], clap_paths[variant][test_split], clap_paths[variant]["text"]),
            }
            reference_variant = "full" if variant == "broad_plus_contrastive_clean" else variant
            reference = t3._reference_stats(repo_root / "release/google_drive_bundle/checkpoints" / dataset / reference_variant)
            args = {
                "device": device, "seed": 42, "concept_activation_cutoff": 0.25,
                "interpretability_cutoff": 0.45, "projection_steps": 1000,
                "projection_batch_size": 50000, "saga_batch_size": 256,
                "saga_step_size": 0.1, "saga_iters": 1000, "saga_epsilon": 1e-4,
                "saga_path_points": 25, "saga_group": True, "saga_include_zero": False,
            }
            metrics_list = t3._run_concept_bottleneck(
                args, dataset, {"name": variant}, vocab_paths[variant], features, paths, out,
                reference["mean_nnz_per_class"], random_word=False,
            )
            metrics = next(item for item in metrics_list if item["selection"] == "full")
            metrics.update({"fold": fold, "val_fold": val_fold, "checkpoint_path": str(out / "full.pt")})
            _json_dump(out / "full.json", metrics)
            _save_selected_predictions(out / "full.pt", features.test, labels["test"], out)
        ast_labels = _load_tensor(ast_label_path).long()
        ast_pred = _load_tensor(ast_pred_path).long()
        cbm_pred = _load_tensor(out / "predictions.pt").long()
        tests = paired_mcnemar(labels["test"].tolist(), ast_pred.tolist(), cbm_pred.tolist())
        _json_dump(out / "mcnemar.json", tests)
        metrics.update({
            "dataset": dataset, "fold": fold, "val_fold": val_fold, "variant": variant,
            "ast_accuracy": float((ast_pred == ast_labels).float().mean().item()),
            "runtime_seconds": float(time.time() - start),
            "checkpoint_path": str(out / "full.pt"),
        })
        results.append(metrics)
    _json_dump(run_dir / "per_fold" / dataset / ("fold{}".format(fold)) / "fold_complete.json", {
        "dataset": dataset, "test_fold": fold, "val_fold": val_fold,
        "ast_checkpoint": str(checkpoint), "ast_summary": ast_summary,
        "runtime_seconds": float(time.time() - start), "variants": list(CV_VARIANTS),
    })
    return results


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append({key: value for key, value in row.items() if not isinstance(value, (dict, list))})
    fields = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scalar_rows)


def _collect_rows(run_dir: Path, prefix: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / prefix).glob("**/full.json")) if (run_dir / prefix).exists() else []:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if prefix == "per_seed":
            parts = path.relative_to(run_dir / prefix).parts
            if len(parts) >= 3:
                row.update({"dataset": parts[0], "variant": parts[1], "seed": int(parts[2].replace("seed", ""))})
        elif prefix == "per_fold":
            parts = path.relative_to(run_dir / prefix).parts
            if len(parts) >= 3 and parts[1].startswith("fold"):
                row.update({"dataset": parts[0], "fold": int(parts[1].replace("fold", "")), "variant": parts[2]})
        rows.append(row)
    return rows


def _collect_mcnemar(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    root = run_dir / ("per_seed" if mode == "seed" else "per_fold")
    rows = []
    for path in sorted(root.glob("**/mcnemar.json")) if root.exists() else []:
        parts = path.relative_to(root).parts
        row = json.loads(path.read_text(encoding="utf-8"))
        row["dataset"] = parts[0]
        if mode == "seed":
            row["variant"] = parts[1]
            row["seed"] = int(parts[2].replace("seed", ""))
        else:
            row["fold"] = int(parts[1].replace("fold", ""))
            row["variant"] = parts[2]
        rows.append(row)
    return rows


def _write_readme(run_dir: Path, config: dict[str, Any], seed_rows: list[dict[str, Any]], fold_rows: list[dict[str, Any]]) -> None:
    def aggregate(rows: list[dict[str, Any]], label: str) -> list[str]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            if row.get("selection", "full") != "full":
                continue
            key = (str(row.get("dataset", "")), str(row.get("variant", "")))
            grouped.setdefault(key, []).append(row)
        lines = []
        for (dataset, variant), values in sorted(grouped.items()):
            acc = np.asarray([float(item["accuracy"]) for item in values], dtype=float)
            f1 = np.asarray([float(item["macro_f1"]) for item in values], dtype=float)
            acc_std = float(acc.std(ddof=1)) if len(acc) > 1 else 0.0
            f1_std = float(f1.std(ddof=1)) if len(f1) > 1 else 0.0
            lines.append("| {} | {} | {} | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} |".format(
                dataset, variant, label, float(acc.mean()), acc_std, float(f1.mean()), f1_std,
            ))
        return lines

    seed_table = "\n".join(aggregate(seed_rows, "5 seeds"))
    fold_table = "\n".join(aggregate(fold_rows, "official folds"))
    text = """# Task 4 predictive robustness and official-fold CV

This run is predictive robustness only. The strong Task 3 random-word and
random-direction controls mean that CV accuracy is not evidence of semantic
interpretability.

## Reproduction and resume

```bash
cd /workspace/Label-free-CBM-Audio
bash scripts/launch_task4_cv.sh --execute --run-id {run_id}
# resume an interrupted run:
bash scripts/launch_task4_cv.sh --execute --run-id {run_id} --resume
```

Seeds: `{seeds}`. Fixed-split robustness reuses the Task 3 AST/CLAP feature
caches read-only. Official ESC-50 folds are test folds 1--5 with the next
cyclic fold as validation. Official UrbanSound8K folds are test folds 1--10
with the next cyclic fold as validation. Training and checkpoint selection use
validation only. AST settings are the existing training-script defaults:
30 epochs, batch size 32, AdamW learning rate 5e-5, weight decay 1e-4,
warmup ratio 0.1, seed 42; UrbanSound8K epochs: {urban_epochs}.
CBM settings reuse Task 3: projection steps 1000, interpretability cutoff
0.45, activation cutoff 0.25, elastic-net alpha 0.99, 25 glm_saga path
points, 1000 SAGA iterations, and validation-selected full-path entry.

Task 2 clean candidate: `broad_plus_contrastive_clean` uses
`data/concept_sets/<dataset>/clean/concepts_ablation_full.txt`. CREMA-D
original-vs-clean interpretation remains provenance-confounded as documented
in Task 2; this run does not claim leakage-removal causality for it.

The per-seed and per-fold tables, aligned predictions, and exact model/cache
paths are machine-readable in this directory. Failed or retried jobs remain
in their isolated directory's `train.log`; existing complete artifacts are
validated before being skipped.

Rows generated so far: {seed_count} fixed-split seed rows and {fold_count}
official-fold CBM rows.

## Aggregate results

| Dataset | Variant | Analysis | Accuracy mean ± std | Macro-F1 mean ± std |
|---|---|---|---:|---:|
{seed_table}
{fold_table}

## Required interpretation answers

1. **α-CBM versus CLAP zero-shot, CLAP linear probe, and sparse AST probe:**
the fixed-split comparison is the Task 3 predeclared comparison in
`results/revision/t3_baselines/README.md` and its machine-readable table. This
Task 4 run adds predictive variance and does not retune or replace those
baselines. Any claim of outperforming or remaining competitive must use that
fixed-split table together with these uncertainty results.

2. **Random controls:** Task 3 showed strong random-word and random-direction
controls. They are therefore treated as competitive predictive controls, not
semantic evidence; this Task 4 CV report does not reinterpret them.

3. **NEC capacity control:** Task 3 contains full/NEC25/NEC10/NEC5 results and
actual mean nonzeros/class. The Task 4 CBM fits use the same validation-only
glm_saga path and report their selected full-path sparsity here; no test tuning
was introduced.

4. **No-class-conditioned LF:** Task 3 is the predeclared broad-only and
broad+contrastive comparison. Task 4 evaluates the selected broad-plus-
contrastive clean candidate in official folds; it is predictive robustness,
not an interpretability claim.

5. **Task 4(f) candidate for Task 4 CV:** `broad_plus_contrastive_clean` was
fixed before this CV run by the Task 3 evaluation. No post-hoc fold result was
used to select it.

The original-vs-clean CREMA-D comparison is provenance-confounded because the
release checkpoint vocabulary provenance does not align with the audit
vocabulary. It must not be described as an isolated leakage-removal effect.
""".format(run_id=config["run_id"], seeds=", ".join(map(str, config["seeds"])), urban_epochs=config["urban_epochs"], seed_count=len(seed_rows), fold_count=len(fold_rows), seed_table=seed_table or "| _pending_ | _pending_ | 5 seeds | _pending_ | _pending_ |", fold_table=fold_table or "| _pending_ | _pending_ | official folds | _pending_ | _pending_ |")
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    output_root = repo_root / "results/revision/t4_cv"
    run_dir = output_root / args.run_id
    if run_dir.exists() and not args.resume:
        raise FileExistsError("Refusing to overwrite existing Task 4 run: {}".format(run_dir))
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    config_path = run_dir / "config.json"
    if args.resume and not config_path.exists():
        raise FileNotFoundError("Cannot resume missing Task 4 config: {}".format(config_path))
    config = {
        "run_id": args.run_id, "repo_root": str(repo_root), "seeds": list(SEEDS),
        "datasets": list(ALL_DATASETS), "cv_datasets": list(CV_DATASETS),
        "cbm_variants": list(CBM_VARIANTS), "cv_variants": list(CV_VARIANTS),
        "urban_epochs": int(args.urban_epochs), "device": args.device,
        "canonical_task3_run": str(repo_root / "results/revision/t3_baselines/t3_gpu_01_fix"),
        "validation_rule": "next cyclic official fold", "selection_split": "val",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
    }
    if not config_path.exists():
        _json_dump(config_path, config)
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    urban_manifest_root = run_dir / "manifests" / "urbansound8k"
    ensure_urban_cv_manifests(repo_root, urban_manifest_root)
    if args.part in {"all", "seeds"}:
        seed_rows = _run_fixed_seed_robustness(
            repo_root, run_dir, Path(config["canonical_task3_run"]), device=args.device
        )
    else:
        seed_rows = _collect_rows(run_dir, "per_seed")
    _write_csv(run_dir / "per_seed.csv", seed_rows)
    _write_csv(run_dir / "mcnemar.csv", _collect_mcnemar(run_dir, "seed"))
    if args.part in {"all", "cv"}:
        for dataset in CV_DATASETS:
            n_folds = 5 if dataset == "esc50" else 10
            for fold in range(1, n_folds + 1):
                _run_cv_fold(repo_root, run_dir, dataset, fold, device=args.device,
                             urban_manifest_root=urban_manifest_root, urban_epochs=args.urban_epochs)
                current_seed_rows = _collect_rows(run_dir, "per_seed")
                current_fold_rows = _collect_rows(run_dir, "per_fold")
                _write_csv(run_dir / "per_seed.csv", current_seed_rows)
                _write_csv(run_dir / "per_fold.csv", current_fold_rows)
                _write_csv(run_dir / "mcnemar.csv", _collect_mcnemar(run_dir, "fold"))
                _write_readme(run_dir, config, current_seed_rows, current_fold_rows)
    else:
        fold_rows = _collect_rows(run_dir, "per_fold")
        _write_csv(run_dir / "per_fold.csv", fold_rows)
    seed_rows = _collect_rows(run_dir, "per_seed")
    fold_rows = _collect_rows(run_dir, "per_fold")
    _write_csv(run_dir / "per_seed.csv", seed_rows)
    _write_csv(run_dir / "per_fold.csv", fold_rows)
    # Keep both modes in one machine-readable McNemar table.
    mcnemar_rows = _collect_mcnemar(run_dir, "seed") + _collect_mcnemar(run_dir, "fold")
    for row in mcnemar_rows:
        row["analysis"] = "seed" if "seed" in row else "fold"
    _write_csv(run_dir / "mcnemar.csv", mcnemar_rows)
    summary_rows = []
    for row in seed_rows:
        summary_rows.append({"analysis": "fixed_seed", **row})
    for row in fold_rows:
        summary_rows.append({"analysis": "official_fold", **row})
    _write_csv(run_dir / "summary.csv", summary_rows)
    _write_readme(run_dir, config, seed_rows, fold_rows)
    _json_dump(run_dir / "complete.json", {
        "run_id": args.run_id, "seed_rows": len(seed_rows), "fold_rows": len(fold_rows),
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--part", choices=("all", "seeds", "cv"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--urban-epochs", type=int, default=30)
    args = parser.parse_args()
    if not args.execute:
        print("Task 4 dry run. Planned fixed seeds: {}".format(SEEDS))
        print("Planned CV folds: ESC-50=5, UrbanSound8K=10")
        return
    print("Task 4 run starting: {}".format(args.run_id), flush=True)
    print("Methods: {}".format(", ".join(CBM_VARIANTS)), flush=True)
    run(args)


if __name__ == "__main__":
    main()
