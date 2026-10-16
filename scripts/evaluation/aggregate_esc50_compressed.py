#!/usr/bin/env python3
"""Aggregate the compressed ESC-50 official-fold AST/LF+broad evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, f1_score


N_FOLDS = 5
VARIANT = "lf_broad"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected tensor at {}".format(path))
    return value.long().view(-1)


def _metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    if labels.shape != predictions.shape:
        raise ValueError("Prediction/label shape mismatch: {} vs {}".format(labels.shape, predictions.shape))
    y_true = labels.numpy()
    y_pred = predictions.numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(50)), average="macro", zero_division=0)),
    }


def _fold_paths(repo_root: Path, fold: int) -> dict[str, Path]:
    compressed = repo_root / "results/revision/esc50_compressed_gpu1"
    canonical = repo_root / "results/revision/t4_cv/t4_gpu_01"
    if fold == 1:
        return {
            "ast_labels": canonical / "fixed_split/ast/esc50/labels.pt",
            "ast_predictions": canonical / "fixed_split/ast/esc50/predictions.pt",
            "cbm_labels": canonical / "per_seed/esc50/lf_broad/seed42/labels.pt",
            "cbm_predictions": canonical / "per_seed/esc50/lf_broad/seed42/predictions.pt",
            "cbm_metrics": canonical / "per_seed/esc50/lf_broad/seed42/full.json",
            "checkpoint": Path("Adam-ousse/ast-esc50-finetuned-fold1"),
            "checkpoint_provenance": compressed / "fold1/ast_checkpoint_provenance.json",
        }
    fold_dir = compressed / "fold{}".format(fold)
    return {
        "ast_labels": fold_dir / "ast_predictions/labels.pt",
        "ast_predictions": fold_dir / "ast_predictions/predictions.pt",
        "cbm_labels": fold_dir / "cbm_lf_broad/labels.pt",
        "cbm_predictions": fold_dir / "cbm_lf_broad/predictions.pt",
        "cbm_metrics": fold_dir / "cbm_lf_broad/full.json",
        "checkpoint": fold_dir / "run_metadata.json",
        "checkpoint_provenance": fold_dir / "run_metadata.json",
    }


def aggregate(repo_root: Path) -> dict[str, Any]:
    compressed = repo_root / "results/revision/esc50_compressed_gpu1"
    vocab = compressed / "vocabularies/esc50/lf_broad.txt"
    if not vocab.exists():
        vocab = repo_root / "data/concept_sets/esc50/concepts_ablation_lf_broad.txt"
    rows: list[dict[str, Any]] = []
    for fold in range(1, N_FOLDS + 1):
        paths = _fold_paths(repo_root, fold)
        required = [paths[key] for key in ("ast_labels", "ast_predictions", "cbm_labels", "cbm_predictions", "cbm_metrics")]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Fold {} is incomplete: {}".format(fold, ", ".join(missing)))
        ast_labels = _load(paths["ast_labels"])
        ast_predictions = _load(paths["ast_predictions"])
        cbm_labels = _load(paths["cbm_labels"])
        cbm_predictions = _load(paths["cbm_predictions"])
        if not torch.equal(ast_labels, cbm_labels):
            raise ValueError("Fold {} AST and CBM labels are not aligned".format(fold))
        ast = _metrics(ast_labels, ast_predictions)
        cbm = _metrics(cbm_labels, cbm_predictions)
        cbm_json = json.loads(paths["cbm_metrics"].read_text(encoding="utf-8"))
        if abs(float(cbm_json["accuracy"]) - cbm["accuracy"]) > 1e-9 or abs(float(cbm_json["macro_f1"]) - cbm["macro_f1"]) > 1e-9:
            raise ValueError("Fold {} stored LF+broad metrics disagree with predictions".format(fold))
        if fold == 1:
            checkpoint = str(paths["checkpoint"])
        else:
            metadata = json.loads(paths["checkpoint_provenance"].read_text(encoding="utf-8"))
            checkpoint = str(metadata["ast_checkpoint"])
        rows.append({
            "fold": fold,
            "test_fold": fold,
            "validation_fold": (fold % N_FOLDS) + 1,
            "test_examples": int(ast_labels.numel()),
            "ast_accuracy": ast["accuracy"],
            "ast_macro_f1": ast["macro_f1"],
            "lf_broad_accuracy": cbm["accuracy"],
            "lf_broad_macro_f1": cbm["macro_f1"],
            "ast_checkpoint": checkpoint,
            "lf_broad_metrics": str(paths["cbm_metrics"]),
            "vocabulary": str(vocab),
            "vocabulary_sha256": _sha256(vocab),
            "selection_split": "val",
        })

    def stats(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        return {"mean": float(statistics.mean(values)), "sd": float(statistics.stdev(values)), "min": min(values), "max": max(values)}

    return {
        "dataset": "esc50",
        "protocol": "official 5-fold CV; test fold f, validation fold f+1 cyclic; train excludes both",
        "variant": VARIANT,
        "fold1_reused": True,
        "new_ast_folds": [2, 3, 4, 5],
        "selection_split": "val",
        "test_examples_per_fold": 400,
        "vocabulary": str(vocab),
        "vocabulary_sha256": _sha256(vocab),
        "folds": rows,
        "aggregate": {
            "ast_accuracy": stats("ast_accuracy"),
            "ast_macro_f1": stats("ast_macro_f1"),
            "lf_broad_accuracy": stats("lf_broad_accuracy"),
            "lf_broad_macro_f1": stats("lf_broad_macro_f1"),
        },
    }


def write_outputs(repo_root: Path, summary: dict[str, Any]) -> None:
    out = repo_root / "results/revision/t4_cv"
    out.mkdir(parents=True, exist_ok=True)
    (out / "esc50_official_cv_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = summary["folds"]
    fields = ["fold", "test_fold", "validation_fold", "test_examples", "ast_accuracy", "ast_macro_f1", "lf_broad_accuracy", "lf_broad_macro_f1", "selection_split", "ast_checkpoint", "lf_broad_metrics", "vocabulary", "vocabulary_sha256"]
    with (out / "esc50_official_cv_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (out / "esc50_official_cv_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{rrrrrrr}\n\\toprule\n")
        handle.write("Test fold & Val. fold & $n$ & AST acc. & AST macro-F1 & LF+broad acc. & LF+broad macro-F1 \\\\\n\\midrule\n")
        for row in rows:
            handle.write("{} & {} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\n".format(row["test_fold"], row["validation_fold"], row["test_examples"], row["ast_accuracy"], row["ast_macro_f1"], row["lf_broad_accuracy"], row["lf_broad_macro_f1"]))
        agg = summary["aggregate"]
        handle.write("Mean $\\pm$ SD & -- & {} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} \\\\\n".format(rows[0]["test_examples"], agg["ast_accuracy"]["mean"], agg["ast_accuracy"]["sd"], agg["ast_macro_f1"]["mean"], agg["ast_macro_f1"]["sd"], agg["lf_broad_accuracy"]["mean"], agg["lf_broad_accuracy"]["sd"], agg["lf_broad_macro_f1"]["mean"], agg["lf_broad_macro_f1"]["sd"]))
        handle.write("\\bottomrule\n\\end{tabular}\n")
    readme = repo_root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    start = "<!-- ESC50_COMPRESSED_CV_START -->"
    end = "<!-- ESC50_COMPRESSED_CV_END -->"
    agg = summary["aggregate"]
    section = "{}\n## ESC-50 official-fold compressed CV\n\nPrincipal variant: LF+broad α-CBM; fold 1 reused the canonical compatible AST/CBM artifacts. Each test fold has 400 examples; validation is the next cyclic fold, and all selection is validation-only.\n\n| Metric | AST | LF+broad α-CBM |\n|---|---:|---:|\n| Accuracy mean ± SD | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} |\n| Macro-F1 mean ± SD | {:.4f} ± {:.4f} | {:.4f} ± {:.4f} |\n\nPer-fold results are in `results/revision/t4_cv/esc50_official_cv_summary.csv` and the LaTeX table is in `results/revision/t4_cv/esc50_official_cv_table.tex`.\n{}".format(start, agg["ast_accuracy"]["mean"], agg["ast_accuracy"]["sd"], agg["lf_broad_accuracy"]["mean"], agg["lf_broad_accuracy"]["sd"], agg["ast_macro_f1"]["mean"], agg["ast_macro_f1"]["sd"], agg["lf_broad_macro_f1"]["mean"], agg["lf_broad_macro_f1"]["sd"], end)
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        text = prefix + "\n\n" + section + ("\n" + suffix if suffix else "")
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    readme.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = aggregate(args.repo_root.resolve())
    write_outputs(args.repo_root.resolve(), summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
