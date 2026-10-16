#!/usr/bin/env python3
"""Aggregate completed official UrbanSound8K fold summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for fold in range(1, 11):
        marker = args.run_dir / ("fold{}/fold_complete.json".format(fold))
        if not marker.exists():
            raise FileNotFoundError("Missing Urban fold {}: {}".format(fold, marker))
        payload = json.loads(marker.read_text(encoding="utf-8"))
        ast = payload["ast_test_metrics"]
        cbm = payload["cbm_test_metrics"]
        rows.append({
            "fold": fold,
            "validation_fold": (fold % 10) + 1,
            "ast_accuracy": float(ast["accuracy"]),
            "ast_macro_f1": float(ast["macro_f1"]),
            "lf_broad_accuracy": float(cbm["accuracy"]),
            "lf_broad_macro_f1": float(cbm["macro_f1"]),
            "fold_complete": str(marker),
        })

    def stats(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in rows]
        return {"mean": float(statistics.mean(values)), "sd": float(statistics.stdev(values)), "min": min(values), "max": max(values)}

    summary = {
        "dataset": "urbansound8k",
        "protocol": "official 10-fold CV; test fold f, validation fold f+1 cyclic; selection validation-only",
        "folds": rows,
        "aggregate": {key: stats(key) for key in ("ast_accuracy", "ast_macro_f1", "lf_broad_accuracy", "lf_broad_macro_f1")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{rrrrrr}\\n\\toprule\\n")
        handle.write("Fold & Val. & AST acc. & AST macro-F1 & LF+broad acc. & LF+broad macro-F1 \\\\\n\\midrule\\n")
        for row in rows:
            handle.write("{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\\n".format(row["fold"], row["validation_fold"], row["ast_accuracy"], row["ast_macro_f1"], row["lf_broad_accuracy"], row["lf_broad_macro_f1"]))
        agg = summary["aggregate"]
        handle.write("Mean $\\pm$ SD & -- & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} \\\\\n".format(agg["ast_accuracy"]["mean"], agg["ast_accuracy"]["sd"], agg["ast_macro_f1"]["mean"], agg["ast_macro_f1"]["sd"], agg["lf_broad_accuracy"]["mean"], agg["lf_broad_accuracy"]["sd"], agg["lf_broad_macro_f1"]["mean"], agg["lf_broad_macro_f1"]["sd"]))
        handle.write("\\bottomrule\\n\\end{tabular}\\n")
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
