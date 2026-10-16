#!/usr/bin/env python3
"""Aggregate the preserved/completed ESC-50 official folds for the overnight run."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score


def _load(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected tensor at {}".format(path))
    return value.long().view(-1)


def _metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    if labels.shape != predictions.shape:
        raise ValueError("Prediction/label shape mismatch")
    return {
        "accuracy": float(accuracy_score(labels.numpy(), predictions.numpy())),
        "macro_f1": float(f1_score(labels.numpy(), predictions.numpy(), labels=list(range(50)), average="macro", zero_division=0)),
    }


def _paths(repo_root: Path, fold4_run_id: str, fold5_run_id: str) -> dict[int, dict[str, Path | str]]:
    canonical = repo_root / "results/revision/t4_cv/t4_gpu_01"
    old = repo_root / "results/revision/esc50_compressed_gpu1"
    fold4 = repo_root / "results/revision" / fold4_run_id
    fold5 = repo_root / "results/revision" / fold5_run_id
    return {
        1: {
            "source": "canonical fold 1",
            "ast_labels": canonical / "fixed_split/ast/esc50/labels.pt",
            "ast_predictions": canonical / "fixed_split/ast/esc50/predictions.pt",
            "cbm_labels": canonical / "per_seed/esc50/lf_broad/seed42/labels.pt",
            "cbm_predictions": canonical / "per_seed/esc50/lf_broad/seed42/predictions.pt",
            "cbm_metrics": canonical / "per_seed/esc50/lf_broad/seed42/full.json",
        },
        2: {"source": "preserved esc50_compressed_gpu1", **{key: old / "fold2" / value for key, value in {
            "ast_labels": "ast_predictions/labels.pt", "ast_predictions": "ast_predictions/predictions.pt",
            "cbm_labels": "cbm_lf_broad/labels.pt", "cbm_predictions": "cbm_lf_broad/predictions.pt",
            "cbm_metrics": "cbm_lf_broad/full.json"}.items()}},
        3: {"source": "preserved esc50_compressed_gpu1", **{key: old / "fold3" / value for key, value in {
            "ast_labels": "ast_predictions/labels.pt", "ast_predictions": "ast_predictions/predictions.pt",
            "cbm_labels": "cbm_lf_broad/labels.pt", "cbm_predictions": "cbm_lf_broad/predictions.pt",
            "cbm_metrics": "cbm_lf_broad/full.json"}.items()}},
        4: {"source": fold4_run_id, **{key: fold4 / "fold4" / value for key, value in {
            "ast_labels": "ast_predictions/labels.pt", "ast_predictions": "ast_predictions/predictions.pt",
            "cbm_labels": "cbm_lf_broad/labels.pt", "cbm_predictions": "cbm_lf_broad/predictions.pt",
            "cbm_metrics": "cbm_lf_broad/full.json"}.items()}},
        5: {"source": fold5_run_id, **{key: fold5 / "fold5" / value for key, value in {
            "ast_labels": "ast_predictions/labels.pt", "ast_predictions": "ast_predictions/predictions.pt",
            "cbm_labels": "cbm_lf_broad/labels.pt", "cbm_predictions": "cbm_lf_broad/predictions.pt",
            "cbm_metrics": "cbm_lf_broad/full.json"}.items()}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold4-run-id", required=True)
    parser.add_argument("--fold5-run-id", required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    rows = []
    for fold, paths in _paths(repo_root, args.fold4_run_id, args.fold5_run_id).items():
        required = [paths[key] for key in ("ast_labels", "ast_predictions", "cbm_labels", "cbm_predictions", "cbm_metrics")]
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError("Fold {} incomplete: {}".format(fold, ", ".join(missing)))
        ast_labels = _load(Path(paths["ast_labels"]))
        ast_predictions = _load(Path(paths["ast_predictions"]))
        cbm_labels = _load(Path(paths["cbm_labels"]))
        cbm_predictions = _load(Path(paths["cbm_predictions"]))
        if not torch.equal(ast_labels, cbm_labels):
            raise ValueError("Fold {} AST/CBM labels are not aligned".format(fold))
        ast = _metrics(ast_labels, ast_predictions)
        cbm = _metrics(cbm_labels, cbm_predictions)
        stored = json.loads(Path(paths["cbm_metrics"]).read_text(encoding="utf-8"))
        if abs(float(stored["accuracy"]) - cbm["accuracy"]) > 1e-9 or abs(float(stored["macro_f1"]) - cbm["macro_f1"]) > 1e-9:
            raise ValueError("Fold {} stored CBM metrics disagree with predictions".format(fold))
        rows.append({
            "fold": fold, "validation_fold": (fold % 5) + 1,
            "test_examples": int(ast_labels.numel()),
            "ast_accuracy": ast["accuracy"], "ast_macro_f1": ast["macro_f1"],
            "lf_broad_accuracy": cbm["accuracy"], "lf_broad_macro_f1": cbm["macro_f1"],
            "source": paths["source"],
        })

    def stats(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        return {"mean": float(statistics.mean(values)), "sd": float(statistics.stdev(values)), "min": min(values), "max": max(values)}

    summary = {
        "dataset": "esc50", "protocol": "official 5-fold CV; test fold f, validation fold f+1 cyclic; selection validation-only",
        "variant": "lf_broad", "folds": rows,
        "aggregate": {key: stats(key) for key in ("ast_accuracy", "ast_macro_f1", "lf_broad_accuracy", "lf_broad_macro_f1")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
