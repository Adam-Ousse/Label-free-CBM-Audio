#!/usr/bin/env python3
"""Render the factual ESC-50 prediction example used in the project README."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "docs/assets/data/esc50_showcase.json"
DEFAULT_OUTPUT = ROOT / "docs/assets/images/esc50_prediction_example.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="1-24524-A-19")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def find_example(payload: dict, sample_id: str) -> dict:
    for group in payload["classes"]:
        for example in group["examples"]:
            if example["id"] == sample_id:
                return example
    raise ValueError(f"Sample not found: {sample_id}")


def display_label(value: str) -> str:
    return value.replace("_", " ")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8"))
    example = find_example(payload, args.sample_id)
    explanation = example["explanation"]
    concepts = explanation["top_concepts"]
    names = [item["concept"] for item in concepts][::-1]
    values = [float(item["contribution"]) for item in concepts][::-1]

    style = {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig = plt.figure(figsize=(9.2, 3.7))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.05, 1.75), wspace=0.22)
        info = fig.add_subplot(grid[0, 0])
        bars = fig.add_subplot(grid[0, 1])

        info.axis("off")
        info.text(0.0, 0.94, "LF + broad CBM", fontsize=15, weight="bold", va="top")
        info.text(0.0, 0.78, "ESC-50 · held-out fold 1", color="#555555", va="top")
        info.text(0.0, 0.57, "Ground truth", color="#666666", va="top")
        info.text(0.0, 0.48, display_label(explanation["gt_class"]), fontsize=14, weight="bold", va="top")
        info.text(0.0, 0.30, "Prediction", color="#666666", va="top")
        info.text(0.0, 0.21, display_label(explanation["pred_class"]), fontsize=14, weight="bold", color="#176B3A", va="top")
        info.text(0.0, 0.05, f'{100 * float(explanation["confidence"]):.2f}% confidence', color="#176B3A", va="bottom")

        colors = ["#4C78A8", "#5B86B5", "#6A94C2", "#79A2CF", "#88B0DC"]
        bars.barh(names, values, color=colors, height=0.62)
        bars.set_title("Top concept contributions", loc="left", weight="bold", pad=12)
        bars.set_xlabel("Contribution to predicted-class logit")
        bars.grid(axis="x", color="#DDDDDD", linewidth=0.7, alpha=0.8)
        bars.set_axisbelow(True)
        bars.spines[["top", "right", "left"]].set_visible(False)
        bars.tick_params(axis="y", length=0)
        bars.set_xlim(0, max(values) * 1.22)
        for index, value in enumerate(values):
            bars.text(value + max(values) * 0.025, index, f"+{value:.2f}", va="center", fontsize=9)

        fig.suptitle(f"Audio example {example['id']}", x=0.055, y=1.015, ha="left", fontsize=10, color="#666666")
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"Wrote {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
