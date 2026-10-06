#!/usr/bin/env python3
"""Plot the cached quantitative concept-faithfulness results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results/concept_faithfulness/final/table_main.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results/concept_faithfulness/final"
DATASETS = ("ESC-50", "UrbanSound8K", "CREMA-D")
KS = (1, 3, 5, 10)
RATE_FIELDS = (
    "topk_recovery_rate",
    "topk_ci_low",
    "topk_ci_high",
    "random_recovery_mean",
    "random_ci_low",
    "random_ci_high",
    "delta_recovery",
    "delta_ci_low",
    "delta_ci_high",
)
COUNT_FIELDS = ("n_errors", "n_eligible", "n_excluded", "topk_recovered")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing figure/table outputs explicitly",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_table(path: Path) -> list[dict[str, Any]]:
    """Load and validate the source table used by both figures."""
    path = absolute(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_table(rows, path)
    normalized = []
    for row in rows:
        normalized_row = dict(row)
        normalized_row["k"] = int(row["k"])
        for field in RATE_FIELDS:
            normalized_row[field] = float(row[field])
        for field in COUNT_FIELDS:
            normalized_row[field] = int(row[field])
        normalized.append(normalized_row)
    return normalized


def validate_table(rows: list[dict[str, Any]], source: Path | None = None) -> None:
    """Reject incomplete or internally inconsistent aggregate results."""
    required = {
        "dataset",
        "variant",
        "k",
        "n_errors",
        "n_eligible",
        "n_excluded",
        "topk_recovered",
        "topk_recovery_rate",
        "topk_ci_low",
        "topk_ci_high",
        "random_recovery_mean",
        "random_ci_low",
        "random_ci_high",
        "delta_recovery",
        "delta_ci_low",
        "delta_ci_high",
    }
    if not rows:
        raise ValueError(f"No rows found in {source or 'the input table'}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")

    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        dataset = str(row["dataset"])
        try:
            k = int(row["k"])
            numeric = {
                key: float(row[key])
                for key in (
                    "topk_recovery_rate",
                    "topk_ci_low",
                    "topk_ci_high",
                    "random_recovery_mean",
                    "random_ci_low",
                    "random_ci_high",
                    "delta_recovery",
                    "delta_ci_low",
                    "delta_ci_high",
                )
            }
            counts = {
                key: int(row[key])
                for key in ("n_errors", "n_eligible", "n_excluded", "topk_recovered")
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric result row: {row}") from exc
        if dataset not in DATASETS or k not in KS:
            raise ValueError(f"Unexpected dataset/k pair: {dataset!r}, {k}")
        key = (dataset, k)
        if key in keyed:
            raise ValueError(f"Duplicate dataset/k row: {key}")
        if counts["n_errors"] < 0 or counts["n_eligible"] < 0 or counts["n_excluded"] < 0:
            raise ValueError(f"Negative sample count in row: {row}")
        if counts["n_eligible"] + counts["n_excluded"] != counts["n_errors"]:
            raise ValueError(f"Eligible plus excluded count does not equal errors: {row}")
        if not 0 <= counts["topk_recovered"] <= counts["n_eligible"]:
            raise ValueError(f"Invalid top-k recovered count: {row}")
        for name, value in numeric.items():
            if name.endswith("ci_low") or name.endswith("ci_high") or name in {
                "topk_recovery_rate",
                "random_recovery_mean",
            }:
                if not 0 <= value <= 1:
                    raise ValueError(f"Recovery value outside [0, 1] in {name}: {row}")
        if not numeric["topk_ci_low"] <= numeric["topk_recovery_rate"] <= numeric["topk_ci_high"]:
            raise ValueError(f"Top-k rate is outside its interval: {row}")
        if not numeric["random_ci_low"] <= numeric["random_recovery_mean"] <= numeric["random_ci_high"]:
            raise ValueError(f"Random rate is outside its interval: {row}")
        if abs(numeric["delta_recovery"] - (numeric["topk_recovery_rate"] - numeric["random_recovery_mean"])) > 1e-9:
            raise ValueError(f"Delta is inconsistent with top-k and random rates: {row}")
        keyed[key] = {**row, **numeric, **counts, "dataset": dataset, "k": k}

    expected = {(dataset, k) for dataset in DATASETS for k in KS}
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"Input table does not contain exactly 3 x 4 rows; missing={missing}, extra={extra}")


def rows_for_dataset(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    return sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: row["k"])


def style() -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def configure_axis(axis: Any) -> None:
    axis.set_xticks(KS)
    axis.set_xlim(0.5, 10.5)
    axis.set_ylim(0, 0.7)
    axis.set_yticks([value / 100 for value in range(0, 71, 10)])
    axis.set_yticklabels([f"{value}%" for value in range(0, 71, 10)])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(width=0.65, length=3)


def save_formats(fig: Any, output_dir: Path, stem: str, dpi: int, overwrite: bool) -> None:
    outputs = (output_dir / f"{stem}.pdf", output_dir / f"{stem}.png")
    if not overwrite:
        existing = [str(output) for output in outputs if output.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite {', '.join(existing)}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    for output in outputs:
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_recovery(rows: list[dict[str, Any]], output_dir: Path, dpi: int, overwrite: bool) -> None:
    top_color = "#1F4E79"
    random_color = "#7A7A7A"
    with plt.rc_context(style()):
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.15), sharey=True)
        for index, (axis, dataset) in enumerate(zip(axes, DATASETS)):
            subset = rows_for_dataset(rows, dataset)
            x = [row["k"] for row in subset]
            top = [row["topk_recovery_rate"] for row in subset]
            top_err = [
                [row["topk_recovery_rate"] - row["topk_ci_low"] for row in subset],
                [row["topk_ci_high"] - row["topk_recovery_rate"] for row in subset],
            ]
            random = [row["random_recovery_mean"] for row in subset]
            random_err = [
                [row["random_recovery_mean"] - row["random_ci_low"] for row in subset],
                [row["random_ci_high"] - row["random_recovery_mean"] for row in subset],
            ]
            axis.errorbar(
                x,
                top,
                yerr=top_err,
                color=top_color,
                marker="o",
                markersize=4.3,
                linewidth=1.7,
                capsize=2.8,
                capthick=0.8,
                label="Top positive contributors",
            )
            axis.errorbar(
                x,
                random,
                yerr=random_err,
                color=random_color,
                marker="o",
                markersize=4.0,
                linewidth=1.5,
                capsize=2.8,
                capthick=0.8,
                label="Matched random positive contributors",
            )
            configure_axis(axis)
            axis.set_title(dataset, pad=8, weight="bold")
            axis.set_xlabel("Concepts removed ($k$)")
            if index == 0:
                axis.set_ylabel("Original errors recovered")
            axis.text(
                0.02,
                0.92,
                f"({chr(ord('a') + index)})",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                weight="bold",
            )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.5,
        )
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.19, top=0.78, wspace=0.10)
        save_formats(fig, output_dir, "faithfulness_recovery", dpi, overwrite)


def plot_delta(rows: list[dict[str, Any]], output_dir: Path, dpi: int, overwrite: bool) -> None:
    colors = {"ESC-50": "#1F4E79", "UrbanSound8K": "#B45F06", "CREMA-D": "#38761D"}
    with plt.rc_context(style()):
        fig, axis = plt.subplots(figsize=(5.4, 3.25))
        for dataset in DATASETS:
            subset = rows_for_dataset(rows, dataset)
            axis.plot(
                [row["k"] for row in subset],
                [100 * row["delta_recovery"] for row in subset],
                color=colors[dataset],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                label=dataset,
            )
        configure_axis(axis)
        axis.set_ylim(-2, 36)
        axis.set_yticks(range(0, 36, 5))
        axis.set_yticklabels([f"{value} pp" for value in range(0, 36, 5)])
        axis.set_xlabel("Concepts removed ($k$)")
        axis.set_ylabel("Top-k recovery advantage")
        axis.axhline(0, color="#666666", linewidth=0.7, linestyle="--", zorder=0)
        axis.legend(frameon=False, loc="upper right")
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.96)
        save_formats(fig, output_dir, "delta_recovery", dpi, overwrite)


def write_latex_table(rows: list[dict[str, Any]], output_dir: Path, overwrite: bool) -> None:
    output = output_dir / "table_main.tex"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Variant & $k$ & Top-k & Random & $\Delta$ (pp) \\",
        r"\midrule",
    ]
    for row in rows:
        top_ci = f"[{100 * row['topk_ci_low']:.2f}, {100 * row['topk_ci_high']:.2f}]"
        random_ci = f"[{100 * row['random_ci_low']:.2f}, {100 * row['random_ci_high']:.2f}]"
        lines.append(
            f"{row['dataset']} & {row['variant']} & {row['k']} & "
            f"{100 * row['topk_recovery_rate']:.2f}\\% {top_ci} & "
            f"{100 * row['random_recovery_mean']:.2f}\\% {random_ci} & "
            f"{100 * row['delta_recovery']:+.2f} \\\\"  # noqa: W605
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("DPI must be positive")
    rows = load_table(args.input)
    output_dir = absolute(args.output_dir)
    plot_recovery(rows, output_dir, args.dpi, args.overwrite)
    plot_delta(rows, output_dir, args.dpi, args.overwrite)
    write_latex_table(rows, output_dir, args.overwrite)
    print(f"Validated {len(rows)} rows from {absolute(args.input)}")
    print(f"Wrote figures and table to {output_dir}")


if __name__ == "__main__":
    main()
