#!/usr/bin/env python3
"""Render manuscript figures from frozen Task 5 and Task 6 summaries.

This module is deliberately export-only: it reads existing JSON/CSV result
artifacts and writes publication figures. It never loads audio, checkpoints,
features, or a GPU device.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK5 = ROOT / "results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json"
TASK6 = ROOT / "results/revision/t6_urbansed/analysis/task6_final_metrics.json"
DEFAULT_OUTPUT = ROOT / "results/revision/final_audit/paper_assets_final/figures"

# Okabe–Ito palette.
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#666666"
METHOD_COLORS = [BLUE, ORANGE, GREEN, VERMILLION, PURPLE]
DATASET_COLORS = {"esc50": BLUE, "urbansound8k": ORANGE, "cremad": GREEN}
VARIANTS = ("full", "lf", "lf_broad", "lf_contrastive")
VARIANT_LABELS = {"full": "Full", "lf": "LF", "lf_broad": "LF + broad", "lf_contrastive": "LF + contrastive"}
DATASET_LABELS = {"esc50": "ESC-50", "urbansound8k": "UrbanSound8K", "cremad": "CREMA-D"}
METHODS = (
    "cbm_positive_contribution",
    "cbm_silence_baseline",
    "ast_window_logit",
    "ast_occlusion_drop",
    "ast_attention_rollout",
)
METHOD_LABELS = {
    "cbm_positive_contribution": r"$\alpha$-CBM",
    "cbm_silence_baseline": r"$\alpha$-CBM + silence",
    "ast_window_logit": "AST logits",
    "ast_occlusion_drop": "Occlusion",
    "ast_attention_rollout": "Attention rollout",
}
EDIT_LABELS = {
    "pitch_shift_minus_4_semitones": "Pitch −4",
    "pitch_shift_plus_4_semitones": "Pitch +4",
    "low_pass_2khz": "Low-pass",
    "gain_minus_20db": "Gain −20 dB",
    "synthetic_reverb_rt60_1s": "Reverb",
    "time_stretch_0p7x": "Time stretch",
}


def style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")
    axis.tick_params(width=0.65, length=3)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)


def save_figure(figure: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=400)
    figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)


def task5_associations(data: dict[str, Any], output: Path) -> None:
    rows = data["rows"]
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row.get("spearman_rho")
        if value is not None and np.isfinite(float(value)):
            values.setdefault((str(row["dataset"]), str(row["variant"])), []).append(float(value))
    datasets = ["esc50", "urbansound8k", "cremad"]
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    x = np.arange(len(VARIANTS))
    for index, (axis, dataset) in enumerate(zip(axes, datasets)):
        means = [float(np.mean(values[(dataset, variant)])) if (dataset, variant) in values else np.nan for variant in VARIANTS]
        axis.bar(x, means, color=[BLUE, ORANGE, GREEN, PURPLE], width=0.68, edgecolor="white", linewidth=0.5)
        axis.axhline(0, color=BLACK, linewidth=0.7)
        axis.set_title(DATASET_LABELS[dataset], weight="bold", pad=8)
        axis.set_xticks(x, [VARIANT_LABELS[v] for v in VARIANTS], rotation=35, ha="right")
        axis.set_xlabel("Model")
        if index == 0:
            axis.set_ylabel("Mean Spearman ρ")
        clean_axis(axis)
        axis.text(0.01, 0.98, f"({chr(97 + index)})", transform=axis.transAxes, va="top", weight="bold")
    figure.suptitle("Concept–descriptor association under controlled edits", y=0.98, weight="bold")
    figure.subplots_adjust(top=0.78, bottom=0.30, left=0.07, right=0.99, wspace=0.06)
    save_figure(figure, output / "task5_controlled_edit_associations")


def task5_heatmap(data: dict[str, Any], output: Path) -> None:
    rows = data["edit_summaries"]
    datasets = ["esc50", "urbansound8k", "cremad"]
    edits = list(EDIT_LABELS)
    matrix = np.full((len(datasets), len(edits)), np.nan)
    for row in rows:
        if row.get("matched_minus_unmatched") is not None:
            matrix[datasets.index(str(row["dataset"])), edits.index(str(row["edit"]))] = float(row["matched_minus_unmatched"])
    cmap = LinearSegmentedColormap.from_list("okabe_ito_diverging", [BLUE, "#F7F7F7", VERMILLION], N=256)
    vmax = float(np.nanmax(np.abs(matrix)))
    figure, axis = plt.subplots(figsize=(8.2, 2.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax), aspect="auto")
    axis.set_xticks(np.arange(len(edits)), [EDIT_LABELS[e] for e in edits], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7.5, color=BLACK)
    axis.set_title("Mapped-minus-null correlation under controlled edits", weight="bold", pad=10)
    figure.colorbar(image, ax=axis, label="Δ mean Spearman ρ", fraction=0.035, pad=0.02)
    save_figure(figure, output / "task5_controlled_edit_heatmap")


def task5_matched_null(data: dict[str, Any], output: Path) -> None:
    rows = data["edit_summaries"]
    for dataset in ("esc50", "urbansound8k", "cremad"):
        subset = [row for row in rows if row["dataset"] == dataset]
        y = np.arange(len(subset))
        figure, axis = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
        for offset, key, color, label in ((-0.12, "matched_mean_rho", DATASET_COLORS[dataset], "Mapped pairs"), (0.12, "unmatched_mean_rho", GREY, "Random unmatched pairs")):
            ci_key = "matched_ci" if key.startswith("matched") else "unmatched_ci"
            means = [float(row[key]) if row.get(key) is not None else np.nan for row in subset]
            lows = [float(row[ci_key][0]) if row.get(ci_key) else np.nan for row in subset]
            highs = [float(row[ci_key][1]) if row.get(ci_key) else np.nan for row in subset]
            finite = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
            means_array = np.asarray(means, dtype=float)
            lower = np.where(finite, means_array - np.asarray(lows, dtype=float), 0.0)
            upper = np.where(finite, np.asarray(highs, dtype=float) - means_array, 0.0)
            axis.errorbar(means_array, y + offset, xerr=[lower, upper], fmt="o", color=color, markersize=4.2, capsize=2.5, linewidth=1.3, label=label)
        axis.axvline(0, color=BLACK, linewidth=0.7)
        axis.set_yticks(y, [EDIT_LABELS[str(row["edit"])] for row in subset])
        axis.set_xlabel("Spearman ρ of edit changes")
        axis.set_title(f"{DATASET_LABELS[dataset]}: mapped versus null pairs", weight="bold", pad=8)
        clean_axis(axis)
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.65)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="lower right")
        save_figure(figure, output / f"task5_{dataset}_matched_vs_null")


def task6_quantitative(data: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.25), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.34
    for axis, (key, title) in zip(axes, (("auroc", "AUROC"), ("average_precision", "Average precision"), ("pointing_game", "Pointing game"))):
        for offset, hop, alpha, hatch in ((-width / 2, 0.5, 1.0, ""), (width / 2, 0.25, 0.48, "//")):
            vals = [float(data["by_hop"][str(hop)]["methods"][method]["macro"][key]) for method in METHODS]
            axis.bar(x + offset, vals, width, color=METHOD_COLORS, alpha=alpha, hatch=hatch, edgecolor="white", linewidth=0.5, label=f"hop={hop:g} s")
        axis.set_title(title, weight="bold", pad=8)
        axis.set_xticks(x, [r"$\alpha$-CBM", "CBM + silence", "AST", "Occlusion", "Rollout"], rotation=35, ha="right")
        axis.set_ylim(0, 1)
        clean_axis(axis)
        if axis is axes[0]: axis.set_ylabel("Macro score")
    axes[-1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    figure.suptitle("URBAN-SED temporal localization", y=0.98, weight="bold")
    figure.subplots_adjust(top=0.77, bottom=0.30, left=0.06, right=0.99, wspace=0.07)
    save_figure(figure, output / "task6_quantitative_comparison")


def task6_heatmap(data: dict[str, Any], output: Path) -> None:
    classes = data["class_names"]
    matrix = np.asarray([[data["by_hop"]["0.5"]["methods"][method]["by_class"][name]["auroc"] for name in classes] for method in METHODS], dtype=float)
    cmap = LinearSegmentedColormap.from_list("okabe_ito_sequential", ["#F7F7F7", SKY, BLUE], N=256)
    figure, axis = plt.subplots(figsize=(9.4, 3.1), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(classes)), [name.replace("_", " ") for name in classes], rotation=38, ha="right")
    axis.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[m] for m in METHODS])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=BLACK)
    axis.set_title("Task 6 AUROC by method and event class (hop = 0.5 s)", weight="bold", pad=10)
    figure.colorbar(image, ax=axis, label="AUROC", fraction=0.025, pad=0.015)
    save_figure(figure, output / "task6_concept_response_heatmap")


def task6_qualitative(data: dict[str, Any], output: Path) -> None:
    examples = data["qualitative_examples"]
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), sharey=False, constrained_layout=True)
    for axis, example in zip(axes.flat, examples):
        centers = (np.asarray(example["window_starts"]) + np.asarray(example["window_ends"])) / 2.0
        for method_index, method in enumerate(METHODS):
            label = METHOD_LABELS[method]
            axis.plot(centers, example["scores"][label], color=METHOD_COLORS[method_index], linewidth=1.35, label=label)
        for event in example.get("events", []):
            axis.axvspan(float(event["event_start"]), float(event["event_end"]), color=YELLOW, alpha=0.3, linewidth=0)
        axis.set_title(f"{example['target_class'].replace('_', ' ')} · {example['selection']}", loc="left", fontsize=8.5, weight="bold")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Score")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
        clean_axis(axis)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=5, frameon=False)
    figure.suptitle("Representative URBAN-SED localization traces", y=1.08, weight="bold")
    save_figure(figure, output / "task6_qualitative_examples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with plt.rc_context(style()):
        task5 = load_json(TASK5)
        task6 = load_json(TASK6)
        task5_associations(task5, args.output)
        task5_heatmap(task5, args.output)
        task5_matched_null(task5, args.output)
        task6_quantitative(task6, args.output)
        task6_heatmap(task6, args.output)
        task6_qualitative(task6, args.output)
    print(f"Wrote publication figures to {args.output}")


if __name__ == "__main__":
    main()
