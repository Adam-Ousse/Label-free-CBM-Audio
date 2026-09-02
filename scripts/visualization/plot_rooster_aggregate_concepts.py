#!/usr/bin/env python3
"""Create the NeurIPS-style aggregate temporal explanation for one ESC-50 clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results/audio_concept_ablation/cbm/esc50/lf_broad/models/esc50_cbm_2026_08_27_22_22"
TEMPORAL_PATH = ROOT / "results/audio_concept_ablation/segmented/esc50/lf_broad/test_temporal_concepts.pt"
FEATURE_DIR = ROOT / "saved_activations/audio_concept_ablation/esc50"
MANIFEST_PATH = ROOT / "data/esc50/manifests/fold1_test.jsonl"
CLASS_MAP_PATH = ROOT / "data/esc50/idx_to_label.json"
DEFAULT_SAMPLE_ID = "1-43382-A-1"
DEFAULT_OUTPUT = ROOT / "results/plots/rooster_1-43382-A-1_aggregate.pdf"


def load_tensor(path: Path, preserve_dtype: bool = False):
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    return value if preserve_dtype else value.float() if torch.is_tensor(value) else value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_tensor(TEMPORAL_PATH, preserve_dtype=True)
    sample_ids = [str(value) for value in bundle["sample_ids"]]
    if args.sample_id not in sample_ids:
        raise ValueError(f"Sample ID not found: {args.sample_id}")
    sample_index = sample_ids.index(args.sample_id)

    concepts = [str(value) for value in bundle["concepts"]]
    temporal = bundle["temporal_concepts"][sample_index].float()
    labels = bundle["labels"].long()
    times = bundle.get("segment_times_sec", bundle["times"]).float().numpy()
    window_sec = float(bundle["window_sec"])

    model_args = read_json(MODEL_DIR / "args.txt")
    feature_name = (
        f"{model_args['test_split']}_backbone_"
        f"{str(model_args['backbone']).replace('/', '_')}_"
        f"{model_args.get('feature_layer', 'layer4')}.pt"
    )
    features = load_tensor(FEATURE_DIR / feature_name)
    W_c = load_tensor(MODEL_DIR / "W_c.pt")
    W_g = load_tensor(MODEL_DIR / "W_g.pt")
    b_g = load_tensor(MODEL_DIR / "b_g.pt").reshape(-1)
    mean = load_tensor(MODEL_DIR / "proj_mean.pt").reshape(1, -1)
    std = load_tensor(MODEL_DIR / "proj_std.pt").reshape(1, -1).clamp_min(1e-6)

    with torch.no_grad():
        activations = (features @ W_c.T - mean) / std
        logits = activations @ W_g.T + b_g
        probabilities = torch.softmax(logits, dim=1)
        prediction = int(logits[sample_index].argmax())
        confidence = float(probabilities[sample_index, prediction])
        contribution = temporal * W_g[prediction].reshape(1, -1)

    strength = contribution.max(dim=0).values
    positive = [index for index, value in enumerate(strength) if float(value) > 0.0]
    top_indices = sorted(positive, key=lambda index: (-float(strength[index]), index))[:5]
    if len(top_indices) < 5:
        fallback = contribution.abs().max(dim=0).values.argsort(descending=True).tolist()
        top_indices.extend(index for index in fallback if index not in top_indices)
        top_indices = top_indices[:5]

    class_names = read_json(CLASS_MAP_PATH)
    label = int(labels[sample_index])
    centers = times + window_sec / 2.0
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    style = {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(7.2, 3.25))
        for color, concept_index in zip(palette, top_indices):
            axis.plot(
                centers,
                contribution[:, concept_index].numpy(),
                color=color,
                marker="o",
                markersize=3.2,
                markeredgewidth=0.4,
                linewidth=2.0,
                label=concepts[concept_index].capitalize(),
            )
        axis.axhline(0.0, color="#666666", linewidth=0.65, alpha=0.65, zorder=0)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Contribution to predicted class")
        axis.set_xticks(centers)
        axis.set_xticklabels([f"{value:.1f}" for value in centers])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
        axis.tick_params(width=0.65, length=3)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#555555")
        fig.savefig(args.output, format="pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {args.output}")
    print("Top concepts:")
    for index in top_indices:
        print(f"  {concepts[index]}: max positive contribution {float(strength[index]):.6f}")


if __name__ == "__main__":
    main()
