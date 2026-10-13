"""Consolidate completed Task 3 JSON outputs into CSV and TeX tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASETS = ("esc50", "urbansound8k", "cremad")
CAPACITIES = ("full", "nec25", "nec10", "nec5")
BASELINE_FILES = {
    "clap_zero_shot": ("metrics.json", "full"),
    "clap_linear_probe": ("metrics.json", "full"),
    "random_direction": ("metrics.json", "full"),
}

FIELDS = [
    "dataset",
    "method",
    "variant",
    "capacity",
    "accuracy",
    "macro_f1",
    "actual_mean_nnz_per_class",
    "target_nec",
    "bottleneck_dim",
    "concepts_input",
    "concepts_output",
    "lambda",
    "selection_split",
    "random_seed",
    "provenance_warning",
    "result_path",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(data: dict[str, Any], *, dataset: str, method: str, variant: str, capacity: str, path: Path) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "variant": variant,
        "capacity": capacity,
        "accuracy": data.get("accuracy"),
        "macro_f1": data.get("macro_f1"),
        "actual_mean_nnz_per_class": data.get("actual_mean_nnz_per_class"),
        "target_nec": data.get("target_nec"),
        "bottleneck_dim": data.get("bottleneck_dim", data.get("concepts_input")),
        "concepts_input": data.get("concepts_input"),
        "concepts_output": data.get("concepts_output"),
        "lambda": data.get("lambda"),
        "selection_split": data.get("selection_split"),
        "random_seed": data.get("random_seed"),
        "provenance_warning": (
            "CREMA-D original-vs-clean comparison provenance-confounded"
            if dataset == "cremad" and variant
            else ""
        ),
        "result_path": str(path),
    }


def collect_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_dir = run_dir / dataset
        if not dataset_dir.is_dir():
            continue
        for method, (filename, capacity) in BASELINE_FILES.items():
            path = dataset_dir / method / filename
            if path.exists():
                rows.append(_row(_load(path), dataset=dataset, method=method, variant="", capacity=capacity, path=path))
        for filename, capacity in (("validation_best.json", "validation_best"), ("reference_nnz.json", "reference_nnz")):
            path = dataset_dir / "sparse_ast_probe" / filename
            if path.exists():
                rows.append(_row(_load(path), dataset=dataset, method="sparse_ast_probe", variant="", capacity=capacity, path=path))
        for method in ("cbm_no_class_lf", "random_word"):
            method_dir = dataset_dir / method
            if not method_dir.is_dir():
                continue
            for variant_dir in sorted(item for item in method_dir.iterdir() if item.is_dir()):
                for capacity in CAPACITIES:
                    path = variant_dir / (capacity + ".json")
                    if path.exists():
                        rows.append(
                            _row(
                                _load(path),
                                dataset=dataset,
                                method=method,
                                variant=variant_dir.name,
                                capacity=capacity,
                                path=path,
                            )
                        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("_", r"\_")


def write_tex(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Task 3 held-out results. NEC rows report the actual mean nonzeros per class reached by validation-only path selection.}",
        r"\label{tab:task3-baselines}",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Dataset & Method & Variant & Capacity & Accuracy & Macro-F1 & Mean NNZ/class & Bottleneck \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _fmt(row["dataset"]),
                _fmt(row["method"]),
                _fmt(row["variant"]),
                _fmt(row["capacity"]),
                _fmt(row["accuracy"]),
                _fmt(row["macro_f1"]),
                _fmt(row["actual_mean_nnz_per_class"]),
                _fmt(row["bottleneck_dim"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"No Task 3 metric JSONs found under {args.run_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_csv(rows, args.output_dir / "table.csv")
    write_tex(rows, args.output_dir / "table.tex")
    print(f"wrote {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
