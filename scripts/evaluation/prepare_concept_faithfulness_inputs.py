#!/usr/bin/env python3
"""Create exact local test manifests and AST-only feature caches.

This is a preparation utility for the faithfulness evaluator.  It delegates split
construction to the repository's existing ``data.prepare_*`` modules, writes only
under the local cache root, and extracts the same pooled 768-D AST representation
used by ``train_cbm.py``.  No CLAP model or concept text embeddings are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CACHE_ROOT = ROOT / "results" / "concept_faithfulness" / "cache"

SPECS = {
    "esc50": {
        "backbone": "ast_esc50",
        "split": "fold1_test",
        "manifest_name": "fold1_test.jsonl",
        "expected_samples": 400,
        "expected_classes": 50,
        "checkpoint": ROOT / "results/audio_concept_ablation/cbm/esc50/lf_broad/models/esc50_cbm_2026_08_27_22_22",
    },
    "urbansound8k": {
        "backbone": "ast_urbansound8k",
        "split": "fold10_test",
        "manifest_name": "fold10_test.jsonl",
        "expected_samples": 837,
        "expected_classes": 10,
        "checkpoint": ROOT / "results/audio_concept_ablation/cbm/urbansound8k/lf_broad/models/urbansound8k_cbm_2026_08_27_23_05",
    },
    "cremad": {
        "backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
        "split": "test",
        "manifest_name": "test.jsonl",
        "expected_samples": 1489,
        "expected_classes": 6,
        "checkpoint": ROOT / "results/cremad_targeted_rerun_20260828/source_ablation_canonical/models/full/cremad_cbm_2026_08_28_21_02",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest_paths(dataset: str, cache_root: Path) -> tuple[Path, Path]:
    metadata_root = cache_root / "metadata" / dataset
    return metadata_root, metadata_root / "manifests" / SPECS[dataset]["manifest_name"]


def _build_manifest(dataset: str, dataset_root: Path, cache_root: Path, overwrite: bool) -> Path:
    spec = SPECS[dataset]
    metadata_root, manifest_path = _manifest_paths(dataset, cache_root)
    if manifest_path.exists() and not overwrite:
        return manifest_path
    if dataset == "esc50":
        from data.prepare_esc50 import build_manifests

        build_manifests(dataset_root, metadata_root, ROOT, val_fold_offset=1, write_default_split=False, default_test_fold=1)
    elif dataset == "urbansound8k":
        from data.prepare_urbansound8k import build_manifests

        build_manifests(dataset_root, metadata_root, ROOT, write_default_split=False)
    else:
        from data.prepare_cremad import build_manifests

        build_manifests(dataset_root, metadata_root, ROOT, val_fraction=0.1, split_seed=42)
    if not manifest_path.is_file():
        raise RuntimeError("Manifest builder did not create {}".format(manifest_path))
    return manifest_path


def _validate_manifest(dataset: str, path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    spec = SPECS[dataset]
    if len(rows) != spec["expected_samples"]:
        raise ValueError("{} canonical test split expected {}, found {}".format(dataset, spec["expected_samples"], len(rows)))
    if {row.get("dataset") for row in rows} != {dataset}:
        raise ValueError("Manifest dataset field mismatch: {}".format(path))
    if dataset == "esc50" and {int(row["fold"]) for row in rows} != {1}:
        raise ValueError("ESC-50 manifest is not fold 1")
    if dataset == "urbansound8k" and {int(row["fold"]) for row in rows} != {10}:
        raise ValueError("UrbanSound8K manifest is not fold 10")
    labels = {int(row["label_idx"]) for row in rows}
    if not labels or min(labels) < 0 or max(labels) >= spec["expected_classes"] or len(labels) != spec["expected_classes"]:
        raise ValueError("Manifest labels do not cover the canonical class index range")
    missing_audio = []
    for row in rows:
        audio_path = Path(str(row["audio_path"]))
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        if not audio_path.is_file():
            missing_audio.append(str(audio_path))
    if missing_audio:
        raise FileNotFoundError("Manifest contains missing audio paths (first 5): {}".format(missing_audio[:5]))
    expected_mapping_path = ROOT / "data" / dataset / "idx_to_label.json"
    if expected_mapping_path.is_file():
        expected_mapping = json.loads(expected_mapping_path.read_text(encoding="utf-8"))
        actual_mapping = {str(index): str(label) for index, label in sorted({int(row["label_idx"]): row["label"] for row in rows}.items())}
        if actual_mapping != {str(key): str(value) for key, value in expected_mapping.items()}:
            raise ValueError("Manifest class ordering does not match {}".format(expected_mapping_path))
    return rows


def _load_checkpoint_shape(checkpoint_dir: Path) -> tuple[int, int]:
    try:
        W_c = torch.load(checkpoint_dir / "W_c.pt", map_location="cpu", weights_only=False)
    except TypeError:
        W_c = torch.load(checkpoint_dir / "W_c.pt", map_location="cpu")
    return int(W_c.shape[0]), int(W_c.shape[1])


def _load_feature_cache(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return {"features": payload}
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError("Unsupported AST cache payload: {}".format(path))
    return payload


def _validate_feature_cache(dataset: str, cache_path: Path, rows: list[dict[str, Any]], feature_width: int) -> None:
    payload = _load_feature_cache(cache_path)
    features = torch.as_tensor(payload["features"])
    if tuple(features.shape) != (len(rows), feature_width):
        raise ValueError("AST cache shape {} != expected {}x{}".format(tuple(features.shape), len(rows), feature_width))
    ids = payload.get("sample_ids")
    if ids is not None and [str(value) for value in ids] != [str(row["id"]) for row in rows]:
        raise ValueError("AST cache sample order does not match manifest")
    if payload.get("backbone") not in (None, SPECS[dataset]["backbone"]):
        raise ValueError("AST cache backbone mismatch")


def _extract_ast(dataset: str, manifest_path: Path, output_path: Path, device: str, batch_size: int, num_workers: int, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError("Refusing to overwrite AST cache; use --overwrite: {}".format(output_path))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # These imports are deliberately lazy: manifest dry-runs and cached evaluation
    # remain usable in environments that do not install transformers.
    import data_utils

    dataset_obj = data_utils.get_audio_dataset(dataset, SPECS[dataset]["split"], manifest_path=str(manifest_path))
    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )
    model, _ = data_utils.get_target_model(SPECS[dataset]["backbone"], device)
    model.eval()
    features = []
    ids: list[str] = []
    paths: list[str] = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            sample_rates = batch["sr"]
            output = model(batch["audio"].to(device), sample_rates=sample_rates)
            if output.ndim != 2:
                output = torch.flatten(output, start_dim=1)
            features.append(output.detach().cpu().float())
            ids.extend(str(value) for value in batch["id"])
            paths.extend(str(value) for value in batch["path"])
            labels.extend(int(value) for value in batch["target"].tolist())
    payload = {
        "format_version": 1,
        "dataset": dataset,
        "split": SPECS[dataset]["split"],
        "backbone": SPECS[dataset]["backbone"],
        "feature_representation": "ASTModel.pooler_output, falling back to last_hidden_state[:, 0, :]",
        "feature_layer_argument": "layer4 (historical cache naming; ASTAudioBackbone returns pooled 768-D embedding)",
        "features": torch.cat(features, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": ids,
        "sample_paths": paths,
        "manifest": str(manifest_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    if device == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ast-features", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-ast", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SPECS[args.dataset]
    manifest_default = _manifest_paths(args.dataset, args.cache_root)[1]
    manifest_path = args.manifest or manifest_default
    feature_path = args.ast_features or args.cache_root / "ast" / args.dataset / "test_ast_features.pt"

    report = {
        "dataset": args.dataset,
        "split": spec["split"],
        "backbone": spec["backbone"],
        "dataset_root": str(args.dataset_root),
        "manifest": str(manifest_path),
        "ast_features": str(feature_path),
        "dataset_present": args.dataset_root.exists(),
        "checkpoint_present": spec["checkpoint"].is_dir(),
        "dry_run": bool(args.dry_run),
    }
    if args.dry_run:
        if manifest_path.is_file():
            rows = _validate_manifest(args.dataset, manifest_path)
            report["manifest_samples"] = len(rows)
        else:
            report["manifest_samples"] = None
        if feature_path.is_file():
            report["feature_cache_present"] = True
        else:
            report["feature_cache_present"] = False
        print(json.dumps(report, indent=2))
        return

    if not args.dataset_root.exists() and not manifest_path.is_file():
        raise FileNotFoundError("Dataset root is absent and no prepared manifest was supplied: {}".format(args.dataset_root))
    if not manifest_path.is_file():
        manifest_path = _build_manifest(args.dataset, args.dataset_root, args.cache_root, args.overwrite)
    rows = _validate_manifest(args.dataset, manifest_path)
    _, feature_width = _load_checkpoint_shape(spec["checkpoint"])

    if feature_path.is_file() and not args.overwrite:
        _validate_feature_cache(args.dataset, feature_path, rows, feature_width)
        print("Using existing AST cache:", feature_path)
    elif args.skip_ast:
        raise FileNotFoundError("--skip-ast was set but AST cache is absent: {}".format(feature_path))
    else:
        _extract_ast(args.dataset, manifest_path, feature_path, args.device, args.batch_size, args.num_workers, args.overwrite)
        _validate_feature_cache(args.dataset, feature_path, rows, feature_width)
        print("Wrote AST cache:", feature_path)

    print(json.dumps({"manifest": str(manifest_path), "ast_features": str(feature_path), "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
