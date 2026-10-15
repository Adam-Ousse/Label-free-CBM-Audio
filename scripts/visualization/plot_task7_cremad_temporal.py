#!/usr/bin/env python3
"""Create Task-7 CREMA-D temporal contribution figures by inference only.

The script scores the speaker-disjoint test manifest with the finalized AST
checkpoint and targeted-full CBM head.  It uses the Task-1 selected -50 dBFS
content mask and masked mean pooling, then selects one highest-confidence
correct and one highest-confidence error for every true emotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_utils
from scripts.evaluation.run_segmented_audio_ablation import split_to_segments
from scripts.evaluation.silence_padding import (
    build_window_masks,
    masked_pool,
    segment_rms_dbfs,
    window_valid_fractions,
    window_valid_sample_counts,
)


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
DEFAULT_ROOT = ROOT / "results/revision/t7_cremad/speaker_disjoint"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--silence-threshold-dbfs", type=float, default=-50.0)
    parser.add_argument(
        "--ids-from",
        type=Path,
        default=None,
        help="Re-render only the already-selected IDs in an existing figure manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tensor(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_cbm(root: Path) -> tuple[dict[str, torch.Tensor], list[str], dict[str, Any]]:
    checkpoint = root / "cbm/full.pt"
    payload = load_tensor(checkpoint)
    tensors = {
        key: payload[key].float()
        for key in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")
    }
    vocab_path = root / "cbm/vocabulary/cremad_full.txt"
    vocab = [line.strip() for line in vocab_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if tensors["W_c"].shape[0] != tensors["W_g"].shape[1]:
        raise ValueError("CBM projection and head dimensions disagree")
    if len(vocab) != 701:
        raise ValueError(f"Expected 701 input vocabulary concepts, found {len(vocab)}")
    return tensors, vocab, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "vocabulary": str(vocab_path.resolve()),
        "vocabulary_sha256": sha256(vocab_path),
        "concepts_input": len(vocab),
        "concepts_output": int(tensors["W_c"].shape[0]),
    }


def recover_concept_labels(root: Path, tensors: dict[str, torch.Tensor], vocab: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Match retained learned directions to the nearest original CLAP concept.

    The lightweight Task-7 checkpoint stores tensors but not the post-filter
    concept strings.  The original 701-way CLAP matrix and validation AST
    features are available, so each retained projection direction is matched
    to its highest validation-set cubed cosine similarity among the input
    concepts.  This recovers labels without fitting or changing the model.
    """
    val_ast = load_tensor(root / "features/val_backbone_ast_task7_speaker_disjoint_layer4.pt").float()
    from utils import compute_concept_matrix_from_activations

    train_targets = compute_concept_matrix_from_activations(
        str(root / "features/train_clap_audio.pt"), str(root / "features/full_clap_text.pt")
    ).float()
    val_targets = compute_concept_matrix_from_activations(
        str(root / "features/val_clap_audio.pt"), str(root / "features/full_clap_text.pt")
    ).float()
    top_k = min(5, train_targets.shape[0])
    keep = train_targets.topk(top_k, dim=0).values.mean(dim=0) > 0.25
    if int(keep.sum()) != 681:
        raise ValueError(f"Task-7 activation filter reconstructed {int(keep.sum())} concepts, expected 681")
    candidates = val_targets[:, keep]
    projected = F.linear(val_ast, tensors["W_c"])

    def normalize_cubed(values: torch.Tensor) -> torch.Tensor:
        centered = values.float() - values.float().mean(dim=0, keepdim=True)
        cubed = centered ** 3
        return cubed / cubed.norm(dim=0, keepdim=True).clamp_min(1e-12)

    similarities = normalize_cubed(projected).T @ normalize_cubed(candidates)
    candidate_names = [name for name, flag in zip(vocab, keep.tolist()) if flag]
    # The checkpoint does not retain concept strings.  Use a deterministic
    # greedy one-to-one assignment so two plotted directions do not silently
    # receive the same descriptor label.
    assigned: list[int | None] = [None] * similarities.shape[0]
    used: set[int] = set()
    pairs = sorted(
        ((float(similarities[row, col]), row, col) for row in range(similarities.shape[0]) for col in range(similarities.shape[1])),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    for _, row, col in pairs:
        if assigned[row] is None and col not in used:
            assigned[row] = col
            used.add(col)
    if any(index is None for index in assigned):
        raise RuntimeError("Failed to assign unique labels to every CBM direction")
    candidate_indices = torch.tensor([int(index) for index in assigned], dtype=torch.long)
    names = [candidate_names[index] for index in candidate_indices.tolist()]
    mapping = {
        "method": "greedy one-to-one assignment to input CLAP concepts by validation-set cubed cosine similarity",
        "activation_cutoff": 0.25,
        "activation_kept": int(keep.sum()),
        "concept_labels": [
            {
                "direction": i,
                "concept": names[i],
                "similarity": float(similarities[i, int(candidate_indices[i])]),
            }
            for i in range(len(names))
        ],
    }
    return names, mapping


def style() -> dict[str, Any]:
    return {
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


def top_concepts(contribution: torch.Tensor) -> list[int]:
    strength = contribution.max(dim=0).values
    positive = [i for i, value in enumerate(strength.tolist()) if value > 0.0]
    selected = sorted(positive, key=lambda i: (-float(strength[i]), i))[:5]
    if len(selected) < 5:
        fallback = contribution.abs().max(dim=0).values.argsort(descending=True).tolist()
        selected.extend(i for i in fallback if i not in selected)
    return selected[:5]


def save_figure(trace: dict[str, Any], concepts: list[str], output_stem: Path, threshold: float) -> None:
    temporal = trace["temporal"]
    predicted = int(trace["predicted"])
    contribution = temporal * trace["W_g"][predicted].reshape(1, -1)
    selected = top_concepts(contribution)
    centers = trace["times"] + trace["window_sec"] / 2.0
    windows = trace["windows"]
    raw_mask = trace["raw_mask"]
    true_name = CLASSES[int(trace["true"])]
    pred_name = CLASSES[predicted]
    outcome = "correct" if predicted == int(trace["true"]) else "error"
    title = (
        f"{true_name} ({outcome}) → {pred_name} | "
        f"confidence={trace['confidence']:.3f} | Task-1 masked mean"
    )
    with plt.rc_context(style()):
        figure, axis = plt.subplots(figsize=(8.8, 3.6))
        for index, (color, concept_index) in enumerate(zip(PALETTE, selected)):
            axis.plot(
                centers,
                contribution[:, concept_index],
                color=color,
                marker="o",
                markersize=3.2,
                markeredgewidth=0.4,
                linewidth=2.0,
                label=concepts[concept_index].capitalize(),
            )
        for start_end, retained in zip(windows, raw_mask):
            if not retained:
                axis.axvspan(float(start_end[0]), float(start_end[1]), color="#B0B0B0", alpha=0.28, zorder=0)
        axis.axhline(0.0, color="#666666", linewidth=0.65, alpha=0.65, zorder=0)
        axis.set_title(title, loc="left", pad=12)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Contribution to predicted class")
        axis.set_xticks(centers)
        axis.set_xticklabels([f"{value:.1f}" for value in centers])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.55)
        axis.tick_params(width=0.65, length=3)
        handles, labels = axis.get_legend_handles_labels()
        handles.append(Patch(facecolor="#B0B0B0", edgecolor="none", alpha=0.28, label=f"masked (< {threshold:g} dBFS / padding)"))
        labels.append(f"masked (< {threshold:g} dBFS / padding)")
        axis.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=1, frameon=False)
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color("#555555")
        figure.savefig(output_stem.with_suffix(".pdf"), format="pdf", dpi=400, bbox_inches="tight")
        figure.savefig(output_stem.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def infer_test(args: argparse.Namespace, root: Path, tensors: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = root / "manifests/test.jsonl"
    dataset = data_utils.get_audio_dataset("cremad", split="test", manifest_path=str(manifest))
    selected_ids = getattr(args, "selected_ids", None)
    if selected_ids is not None:
        dataset.samples = [sample for sample in dataset.samples if str(sample["id"]) in selected_ids]
        if len(dataset.samples) != len(selected_ids):
            raise ValueError("--ids-from contains an ID absent from the Task-7 test manifest")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        collate_fn=data_utils.collate_audio_batch,
    )
    checkpoint = root / "ast/test/best_model"
    target_name = "ast_local__" + str(checkpoint.relative_to(ROOT)).replace("/", "__")
    backbone, _ = data_utils.get_target_model(target_name, args.device)
    backbone.eval()
    device = torch.device(args.device)
    for key in tensors:
        tensors[key] = tensors[key].to(device)

    traces: list[dict[str, Any]] = []
    sample_rate = 16000
    duration_sec = 4.0
    times_ref = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="Task-7 temporal inference"):
            segments, times = split_to_segments(batch["audio"], sample_rate, args.window_sec, args.hop_sec)
            true_duration = batch["duration_sec"].float().clamp(0.0, duration_sec)
            valid_counts = window_valid_sample_counts(true_duration, times, sample_rate, args.window_sec)
            valid_fractions = window_valid_fractions(true_duration, times, args.window_sec)
            rms = segment_rms_dbfs(segments, valid_counts)
            raw_mask = build_window_masks(valid_fractions, rms, args.silence_threshold_dbfs)["content_mask"]
            flat = segments.reshape(-1, segments.shape[-1])
            chunks = []
            for start in range(0, flat.shape[0], args.segment_batch_size):
                current = flat[start : start + args.segment_batch_size]
                sr_batch = torch.full((current.shape[0],), sample_rate, dtype=torch.long)
                chunks.append(backbone(current, sample_rates=sr_batch).detach())
            features = torch.cat(chunks).reshape(segments.shape[0], segments.shape[1], -1)
            temporal = (F.linear(features, tensors["W_c"]) - tensors["proj_mean"].reshape(1, 1, -1)) / tensors["proj_std"].reshape(1, 1, -1).clamp_min(1e-6)
            pooled, effective_mask, fallback = masked_pool(temporal, "mean", raw_mask.to(device), rms.to(device))
            logits = F.linear(pooled, tensors["W_g"], tensors["b_g"])
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)
            confidence = probs.gather(1, preds[:, None]).squeeze(1)
            if times_ref is None:
                times_ref = times
            for i in range(len(batch["id"])):
                traces.append({
                    "sample_id": str(batch["id"][i]),
                    "audio_path": str(batch["path"][i]),
                    "true": int(batch["target"][i]),
                    "predicted": int(preds[i]),
                    "confidence": float(confidence[i]),
                    "temporal": temporal[i].cpu(),
                    "W_g": tensors["W_g"].cpu(),
                    "times": times.cpu(),
                    "windows": torch.stack([times, times + args.window_sec], dim=1).cpu(),
                    "raw_mask": raw_mask[i].cpu().tolist(),
                    "effective_mask": effective_mask[i].cpu().tolist(),
                    "fallback": bool(fallback[i]),
                    "window_sec": args.window_sec,
                })
    del backbone
    return traces, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "pooling": "masked mean over effective Task-1 content windows",
        "masking_mode": "content_mask = padding_mask AND RMS_dBFS >= -50; all-masked rows retain highest-RMS window",
        "silence_threshold_dbfs": args.silence_threshold_dbfs,
        "num_test_examples": len(traces),
    }


def select_examples(traces: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    selected = []
    for label, emotion in enumerate(CLASSES):
        candidates = [(index, item) for index, item in enumerate(traces) if item["true"] == label and item["predicted"] == label]
        if not candidates:
            raise RuntimeError(f"No correct test example found for {emotion}")
        _, correct = max(candidates, key=lambda pair: (pair[1]["confidence"], -pair[0]))
        errors = [(index, item) for index, item in enumerate(traces) if item["true"] == label and item["predicted"] != label]
        if not errors:
            raise RuntimeError(f"No test error found for {emotion}")
        _, error = max(errors, key=lambda pair: (pair[1]["confidence"], -pair[0]))
        selected.extend([(emotion, "correct", correct), (emotion, "error", error)])
    return selected


def actor_id(sample_id: str) -> str:
    return sample_id.split("_", 1)[0]


def main() -> None:
    args = parse_args()
    root = absolute(args.root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tensors, vocab, cbm_meta = load_cbm(root)
    concepts, concept_meta = recover_concept_labels(root, tensors, vocab)
    selected_rows = None
    if args.ids_from is not None:
        with args.ids_from.open(newline="", encoding="utf-8") as handle:
            selected_rows = list(csv.DictReader(handle))
        args.selected_ids = {Path(row["audio_filename"]).stem for row in selected_rows}
        if len(args.selected_ids) != 12:
            raise ValueError("--ids-from must contain the 12-row Task-7 figure manifest")
    traces, inference_meta = infer_test(args, root, tensors)
    if selected_rows is None:
        selected = select_examples(traces)
    else:
        by_id = {trace["sample_id"]: trace for trace in traces}
        selected = [(row["emotion"], row["outcome"], by_id[Path(row["audio_filename"]).stem]) for row in selected_rows]
        inference_meta["selection_source"] = str(args.ids_from.resolve())

    rows = []
    for emotion, outcome, trace in selected:
        stem = figures / f"{emotion}_{outcome}"
        if not args.overwrite and (stem.with_suffix(".pdf").exists() or stem.with_suffix(".png").exists()):
            raise FileExistsError(f"Refusing to overwrite existing figure for {emotion}/{outcome}: {stem}")
        save_figure(trace, concepts, stem, args.silence_threshold_dbfs)

    manifest_by_id = {str(json.loads(line)["id"]): json.loads(line) for line in (root / "manifests/test.jsonl").read_text().splitlines() if line.strip()}
    for emotion, outcome, trace in selected:
        row = manifest_by_id[trace["sample_id"]]
        rows.append({
            "emotion": emotion,
            "outcome": outcome,
            "audio_filename": Path(trace["audio_path"]).name,
            "actor_id": str(row.get("actor_id", actor_id(trace["sample_id"]))),
            "true_class": CLASSES[trace["true"]],
            "predicted_class": CLASSES[trace["predicted"]],
            "confidence": trace["confidence"],
            "checkpoint_model": inference_meta["checkpoint"],
            "cbm_checkpoint": cbm_meta["checkpoint"],
            "masking_mode": inference_meta["masking_mode"],
            "pooling": inference_meta["pooling"],
            "masked_window_indices": [i for i, retained in enumerate(trace["raw_mask"]) if not retained],
            "figure_pdf": str((figures / f"{emotion}_{outcome}.pdf").relative_to(ROOT)),
            "figure_png": str((figures / f"{emotion}_{outcome}.png").relative_to(ROOT)),
        })

    with (figures / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (figures / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (figures / "concept_label_mapping.json").write_text(json.dumps(concept_meta, indent=2) + "\n", encoding="utf-8")
    (figures / "inference_metadata.json").write_text(json.dumps({"cbm": cbm_meta, "inference": inference_meta, "concept_labels": concept_meta["method"]}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(rows), "manifest": str(figures / "manifest.csv"), "checkpoint": inference_meta["checkpoint"], "masking": inference_meta["masking_mode"]}, indent=2))


if __name__ == "__main__":
    main()
