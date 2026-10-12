#!/usr/bin/env python3
"""Compare Task 0 reproductions with the canonical reported tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLE2_ROWS = {
    ("esc50", "fine_tuned_ast"),
    ("esc50", "lf_broad"),
    ("urbansound8k", "fine_tuned_ast"),
    ("urbansound8k", "lf_broad"),
    ("cremad", "fine_tuned_ast"),
    ("cremad", "full"),
}
METRICS = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-results", type=Path, required=True)
    parser.add_argument("--segmented-results", type=Path, required=True)
    parser.add_argument("--environmental-segmented", type=Path, required=True)
    parser.add_argument("--targeted-segmented", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-pp", type=float, default=0.1)
    return parser.parse_args()


def check(actual: float, expected: float, tolerance_pp: float) -> dict:
    delta_pp = (actual - expected) * 100.0
    return {
        "actual": actual,
        "expected": expected,
        "delta_pp": delta_pp,
        "pass": abs(delta_pp) <= tolerance_pp,
    }


def expected_segmented(path: Path) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", [payload])
    out = {}
    for model in models:
        variant = model.get("variant", "fine_tuned_ast")
        if variant.startswith("targeted_"):
            variant = "full"
        out[(model["dataset"], variant)] = model
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_payload = json.loads(args.global_results.read_text(encoding="utf-8"))
    global_rows = {(row["dataset"], row["variant"]): row for row in global_payload["rows"]}

    table1_rows = []
    for dataset in ("esc50", "urbansound8k", "cremad"):
        ast = global_rows[(dataset, "fine_tuned_ast")]
        cbm_variant = "full" if dataset == "cremad" else "lf_broad"
        cbm = global_rows[(dataset, cbm_variant)]
        for row in (ast, cbm):
            table1_rows.append(
                {
                    "dataset": dataset,
                    "variant": row["variant"],
                    "accuracy": row["accuracy"],
                    "macro_f1": row["macro_f1"],
                    "sparsity": row["sparsity"],
                    "comparison": row["comparison"],
                }
            )

    segmented_payload = json.loads(args.segmented_results.read_text(encoding="utf-8"))
    segmented_rows = {
        (row["dataset"], row.get("variant", "fine_tuned_ast"), row["pool"]): row
        for row in segmented_payload["rows"]
    }
    environmental = expected_segmented(args.environmental_segmented)
    targeted = expected_segmented(args.targeted_segmented)
    table2_rows = []
    for dataset, variant in sorted(TABLE2_ROWS):
        expected_model = targeted[(dataset, variant)] if (dataset, variant) in targeted else environmental[(dataset, variant)]
        for pool in ("mean", "max", "lme"):
            actual = segmented_rows[(dataset, variant, pool)]
            expected_pool = expected_model["pools"][pool]
            comparison = {
                metric: check(actual[metric], expected_pool[metric], args.tolerance_pp)
                for metric in METRICS
            }
            table2_rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "pool": pool,
                    "actual": {metric: actual[metric] for metric in METRICS},
                    "expected": {metric: expected_pool[metric] for metric in METRICS},
                    "comparison": comparison,
                    "expected_source": str(
                        args.targeted_segmented.resolve()
                        if (dataset, variant) in targeted
                        else args.environmental_segmented.resolve()
                    ),
                }
            )

    checks = [
        comparison
        for row in table1_rows
        for comparison in row["comparison"].values()
    ]
    checks.extend(
        comparison
        for row in table2_rows
        for comparison in row["comparison"].values()
    )
    payload = {
        "tolerance_pp": args.tolerance_pp,
        "table1": table1_rows,
        "table2": table2_rows,
        "pass": all(item["pass"] for item in checks),
        "provenance_warning": (
            "The environmental segmented report contains superseded CREMA-D LF, LF+broad, "
            "and LF+contrastive checkpoint paths. CREMA-D Table 2 uses the release bundle's "
            "dedicated targeted full report; no stale rows were used for the gate."
        ),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table", "dataset", "variant", "pool", "metric", "actual", "expected", "delta_pp", "pass"])
        for table, rows in (("table1", table1_rows), ("table2", table2_rows)):
            for row in rows:
                pools = [row["pool"]] if table == "table2" else ["global"]
                for pool in pools:
                    comparisons = row["comparison"]
                    actual = row.get("actual", row)
                    for metric, result in comparisons.items():
                        writer.writerow(
                            [
                                table,
                                row["dataset"],
                                row["variant"],
                                pool,
                                metric,
                                actual.get(metric),
                                row.get("expected", {}).get(metric, result["expected"]),
                                result["delta_pp"],
                                result["pass"],
                            ]
                        )
    print(json.dumps({"pass": payload["pass"], "table1_rows": len(table1_rows), "table2_rows": len(table2_rows)}, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
