#!/usr/bin/env python3
"""Build a consolidated baseline/CBM ablation report from completed runs."""

import argparse
import csv
import json
from pathlib import Path


DATASET_NAMES = {
    "esc50": "ESC-50 (fold 1)",
    "urbansound8k": "UrbanSound8K (fold 10)",
    "cremad": "CREMA-D",
}
VARIANT_NAMES = {
    "lf": "LF",
    "lf_broad": "LF + Broad",
    "lf_contrastive": "LF + Contrastive",
    "full": "Full",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results/audio_concept_ablation")
    return parser.parse_args()


def _pct(value):
    return "{:.2f}".format(100.0 * value)


def main():
    args = parse_args()
    root = Path(args.results_root)
    baselines = json.loads((root / "baselines" / "summary.json").read_text(encoding="utf-8"))
    cbm_payload = json.loads((root / "cbm" / "summary.json").read_text(encoding="utf-8"))

    baseline_by_dataset = {row["dataset"]: row for row in baselines}
    runs = cbm_payload["runs"]
    rows = []
    for dataset in DATASET_NAMES:
        baseline = baseline_by_dataset.get(dataset)
        if baseline is None:
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": "Fine-tuned AST",
                "accuracy": baseline["accuracy"],
                "macro_f1": baseline["f1_macro"],
                "accuracy_delta_vs_ast": 0.0,
                "macro_f1_delta_vs_ast": 0.0,
                "input_concepts": "",
                "retained_concepts": "",
                "sparsity_fraction": "",
                "nonzero_weights": "",
                "total_weights": "",
                "avg_nonzero_per_class": "",
                "model_dir": baseline["model"],
            }
        )
        for run in runs:
            if run["dataset"] != dataset:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": "CBM: {}".format(VARIANT_NAMES[run["variant"]]),
                    "accuracy": run["test_accuracy"],
                    "macro_f1": run["test_f1_macro"],
                    "accuracy_delta_vs_ast": run["test_accuracy"] - baseline["accuracy"],
                    "macro_f1_delta_vs_ast": run["test_f1_macro"] - baseline["f1_macro"],
                    "input_concepts": run["input_concepts"],
                    "retained_concepts": run["retained_concepts"],
                    "sparsity_fraction": run["sparsity_fraction"],
                    "nonzero_weights": run["nonzero_weights"],
                    "total_weights": run["total_weights"],
                    "avg_nonzero_per_class": run["avg_nonzero_per_class"],
                    "model_dir": run["model_dir"],
                }
            )

    if len(runs) != 12:
        raise RuntimeError("Expected 12 completed CBM runs, found {}".format(len(runs)))

    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "comparison.json").write_text(
        json.dumps({"protocol": cbm_payload["protocol"], "results": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Audio concept bottleneck ablation",
        "",
        "All metrics are from the fixed held-out test split. Sparsity is the fraction of final",
        "CBM classifier weights with absolute value at most `1e-5`.",
        "",
        "| Dataset | Model | Accuracy (%) | Macro-F1 (%) | Delta acc. (pp) | Delta F1 (pp) | Concepts (kept/input) | Sparsity (%) | Nonzero/class |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        is_ast = row["model"] == "Fine-tuned AST"
        concepts = "--" if is_ast else "{}/{}".format(row["retained_concepts"], row["input_concepts"])
        sparsity = "--" if is_ast else _pct(row["sparsity_fraction"])
        nonzero = "--" if is_ast else "{:.1f}".format(row["avg_nonzero_per_class"])
        lines.append(
            "| {} | {} | {} | {} | {:+.2f} | {:+.2f} | {} | {} | {} |".format(
                DATASET_NAMES[row["dataset"]],
                row["model"],
                _pct(row["accuracy"]),
                _pct(row["macro_f1"]),
                100.0 * row["accuracy_delta_vs_ast"],
                100.0 * row["macro_f1_delta_vs_ast"],
                concepts,
                sparsity,
                nonzero,
            )
        )

    protocol = cbm_payload["protocol"]
    lines.extend(
        [
            "",
            "## Fixed CBM hyperparameters",
            "",
            "- Seed: `{seed}`; CLAP target: `laion/clap-htsat-unfused`; objective: `{similarity_objective}`".format(**protocol),
            "- Projection: up to `{proj_steps}` steps, batch `{proj_batch_size}`, interpretability cutoff `{interpretability_cutoff}`".format(**protocol),
            "- Grounding activation cutoff: `{concept_activation_cutoff}`".format(**protocol),
            "- Sparse classifier: SAGA `{n_iters}` iterations, batch `{saga_batch_size}`, lambda `{lam}`, elastic-net alpha `{elastic_alpha}`".format(**protocol),
            "",
            "The LLM only proposed candidates. CLAP/audio activation and projectability filters determined",
            "which concepts were retained before sparse classifier training.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote {}, {}, and {}".format(root / "report.md", root / "comparison.csv", root / "comparison.json"))


if __name__ == "__main__":
    main()
