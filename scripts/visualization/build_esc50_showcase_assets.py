#!/usr/bin/env python3
"""Export an interactive LF + broad ESC-50 showcase for static GitHub Pages."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
AUDIO_OUT = DOCS / "assets" / "audio"
DATA_OUT = DOCS / "assets" / "data"
JSON_OUT = DATA_OUT / "esc50_showcase.json"
JS_OUT = DATA_OUT / "esc50_showcase.js"
RUN_SUMMARY = ROOT / "results/audio_concept_ablation/cbm/esc50/lf_broad/run_summary.json"
TEMPORAL = ROOT / "results/audio_concept_ablation/segmented/esc50/lf_broad/test_temporal_concepts.pt"
SEGMENT_METRICS = TEMPORAL.with_name("segmented_metrics.json")
BASELINE_METRICS = ROOT / "results/audio_concept_ablation/baselines/esc50.json"
TEST_MANIFEST = ROOT / "data/esc50/manifests/fold1_test.jsonl"
CLASS_MAP = ROOT / "data/esc50/idx_to_label.json"

EMOJI = {
    "dog": "🐶", "rooster": "🐓", "pig": "🐷", "cow": "🐄", "frog": "🐸",
    "cat": "🐱", "hen": "🐔", "insects": "🐞", "sheep": "🐑", "crow": "🐦",
    "rain": "🌧️", "sea_waves": "🌊", "crackling_fire": "🔥", "crickets": "🦗",
    "chirping_birds": "🐤", "water_drops": "💧", "wind": "🌬️", "pouring_water": "🚿",
    "toilet_flush": "🚽", "thunderstorm": "⛈️", "crying_baby": "👶", "sneezing": "🤧",
    "clapping": "👏", "breathing": "😮", "coughing": "😷", "footsteps": "👣",
    "laughing": "😂", "brushing_teeth": "🪥", "snoring": "😴", "drinking_sipping": "🥤",
    "door_wood_knock": "🚪", "mouse_click": "🖱️", "keyboard_typing": "⌨️",
    "door_wood_creaks": "🚪", "can_opening": "🥫", "washing_machine": "🧺",
    "vacuum_cleaner": "🧹", "clock_alarm": "⏰", "clock_tick": "🕒",
    "glass_breaking": "🥛", "helicopter": "🚁", "chainsaw": "🪚", "siren": "🚨",
    "car_horn": "🚗", "engine": "⚙️", "train": "🚆", "church_bells": "🔔",
    "airplane": "✈️", "fireworks": "🎆", "hand_saw": "🪚",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--max-concepts", type=int, default=5)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--temporal-concepts", type=Path, default=TEMPORAL)
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_tensor(path: Path, preserve_dtype: bool = False) -> Any:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if torch.is_tensor(value) and not preserve_dtype:
        value = value.float()
    return value


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def numbers(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in tensor.reshape(-1)]


def choose_examples(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    count: int,
) -> dict[int, list[tuple[int, str]]]:
    """Prefer one confident correct and one confident error per true class."""
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped[int(label)].append(index)
    output: dict[int, list[tuple[int, str]]] = {}
    for label, indices in grouped.items():
        correct = sorted(
            (i for i in indices if int(predictions[i]) == label),
            key=lambda i: (-float(confidence[i]), i),
        )
        wrong = sorted(
            (i for i in indices if int(predictions[i]) != label),
            key=lambda i: (-float(confidence[i]), i),
        )
        picks: list[tuple[int, str]] = []
        if correct:
            picks.append((correct[0], "highest-confidence correct"))
        if wrong and len(picks) < count:
            picks.append((wrong[0], "most-confident incorrect"))
        used = {index for index, _ in picks}
        rest = sorted((i for i in indices if i not in used), key=lambda i: (-float(confidence[i]), i))
        for index in rest:
            if len(picks) >= count:
                break
            outcome = "correct fallback" if int(predictions[index]) == label else "incorrect fallback"
            picks.append((index, outcome))
        if len(picks) != count:
            raise RuntimeError(f"Class {label} yielded only {len(picks)} examples")
        output[label] = picks
    return output


def export_temporal(
    values: torch.Tensor,
    predicted_weights: torch.Tensor,
    concepts: list[str],
    times: torch.Tensor,
    window: float,
) -> dict[str, Any]:
    """Mirror plot_temporal_concept_explanations.py's two panels."""
    contribution = (values.float() * predicted_weights.reshape(1, -1)).detach()
    strength = contribution.max(dim=0).values
    positive = [i for i in range(len(concepts)) if float(strength[i]) > 0]
    top = sorted(positive, key=lambda i: (-float(strength[i]), i))[:5]
    if len(top) < 5:
        fallback = contribution.abs().max(dim=0).values.argsort(descending=True).tolist()
        top.extend(i for i in fallback if i not in top)
        top = top[:5]
    local_values, local_indices = torch.topk(contribution, k=5, dim=1)
    return {
        "centers_sec": numbers(times.float() + window / 2, 3),
        "series": [
            {"index": i, "concept": concepts[i], "values": numbers(contribution[:, i])}
            for i in top
        ],
        "local": [
            [
                {
                    "index": int(local_indices[segment, rank]),
                    "concept": concepts[int(local_indices[segment, rank])],
                    "contribution": round(float(local_values[segment, rank]), 6),
                }
                for rank in range(5)
            ]
            for segment in range(contribution.shape[0])
        ],
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0 or args.max_concepts <= 0:
        raise ValueError("Sample and concept counts must be positive")
    summary = read_json(RUN_SUMMARY)
    model_dir = absolute(args.model_dir or Path(summary["model_dir"]))
    model_args = read_json(model_dir / "args.txt")
    records = read_jsonl(absolute(args.manifest))
    class_map = read_json(CLASS_MAP)
    class_names = [str(class_map[str(i)]) for i in range(len(class_map))]
    concepts = [line for line in (model_dir / "concepts.txt").read_text(encoding="utf-8").splitlines() if line]

    W_c = load_tensor(model_dir / "W_c.pt")
    W_g = load_tensor(model_dir / "W_g.pt")
    b_g = load_tensor(model_dir / "b_g.pt").reshape(-1)
    mean = load_tensor(model_dir / "proj_mean.pt").reshape(1, -1)
    std = load_tensor(model_dir / "proj_std.pt").reshape(1, -1).clamp_min(1e-6)
    feature_file = (
        f"{model_args['test_split']}_backbone_"
        f"{str(model_args['backbone']).replace('/', '_')}_{model_args.get('feature_layer', 'layer4')}.pt"
    )
    features = load_tensor(absolute(Path(model_args["activation_dir"])) / feature_file)
    bundle = load_tensor(absolute(args.temporal_concepts), preserve_dtype=True)
    temporal = bundle["temporal_concepts"].float()
    labels = bundle["labels"].long()

    if [row["id"] for row in records] != [str(value) for value in bundle["sample_ids"]]:
        raise ValueError("Temporal samples and test manifest order differ")
    if not torch.equal(labels, torch.tensor([int(row["label_idx"]) for row in records])):
        raise ValueError("Temporal labels and test manifest labels differ")
    if concepts != [str(value) for value in bundle["concepts"]]:
        raise ValueError("Temporal and checkpoint concept orders differ")

    with torch.no_grad():
        activations = (features @ W_c.T - mean) / std
        logits = activations @ W_g.T + b_g
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)
        confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)

    selected = choose_examples(labels, predictions, confidence, args.samples_per_class)
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    for old in AUDIO_OUT.glob("*.wav"):
        old.unlink()

    payload_classes = []
    correct_count = 0
    wrong_count = 0
    times = bundle.get("segment_times_sec", bundle["times"])
    window = float(bundle["window_sec"])
    for label_index, label in enumerate(class_names):
        examples = []
        for sample_index, reason in selected[label_index]:
            row = records[sample_index]
            source = absolute(Path(row["audio_path"]))
            if not source.exists():
                raise FileNotFoundError(source)
            audio_name = f"{row['id']}.wav"
            shutil.copy2(source, AUDIO_OUT / audio_name)
            predicted = int(predictions[sample_index])
            is_correct = predicted == label_index
            correct_count += int(is_correct)
            wrong_count += int(not is_correct)

            effects = activations[sample_index].reshape(1, -1) * W_g
            predicted_effect = effects[predicted]
            top = predicted_effect.abs().argsort(descending=True)[: args.max_concepts].tolist()
            examples.append({
                "id": str(row["id"]),
                "fold": int(row["fold"]),
                "duration_sec": float(row["duration"]),
                "audio": f"assets/audio/{audio_name}",
                "sample_rate": int(row["sample_rate"]),
                "selection_reason": reason,
                "explanation": {
                    "gt_class": label,
                    "gt_index": label_index,
                    "pred_class": class_names[predicted],
                    "pred_index": predicted,
                    "correct": is_correct,
                    "confidence": round(float(confidence[sample_index]), 8),
                    "base_logits": numbers(logits[sample_index]),
                    "top_concepts": [
                        {
                            "index": concept_index,
                            "concept": concepts[concept_index],
                            "activation": round(float(activations[sample_index, concept_index]), 6),
                            "contribution": round(float(predicted_effect[concept_index]), 6),
                            "class_effects": numbers(effects[:, concept_index]),
                        }
                        for concept_index in top
                    ],
                },
                "temporal": export_temporal(
                    temporal[sample_index], W_g[predicted], concepts, times, window
                ),
            })
        payload_classes.append({
            "label": label,
            "label_idx": label_index,
            "emoji": EMOJI.get(label, "🔊"),
            "examples": examples,
        })

    baseline = read_json(BASELINE_METRICS)
    segment_metrics = read_json(SEGMENT_METRICS)
    payload = {
        "schema_version": 2,
        "dataset": "esc50",
        "split": str(bundle["split"]),
        "variant": "lf_broad",
        "variant_label": "LF + broad",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples_per_class": args.samples_per_class,
        "max_concepts": args.max_concepts,
        "num_classes": len(class_names),
        "model_dir": str(model_dir.relative_to(ROOT)),
        "window_sec": window,
        "hop_sec": float(bundle["hop_sec"]),
        "showcase_counts": {"correct": correct_count, "incorrect": wrong_count},
        "metrics": {
            "baseline": {"accuracy": baseline["accuracy"], "macro_f1": baseline["f1_macro"]},
            "cbm": {
                "accuracy": summary["test_accuracy"],
                "macro_f1": summary["test_f1_macro"],
                "sparsity": summary["sparsity_fraction"],
                "retained_concepts": summary["retained_concepts"],
            },
            "segmented": segment_metrics["pools"],
        },
        "classes": payload_classes,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    JSON_OUT.write_text(serialized + "\n", encoding="utf-8")
    JS_OUT.write_text(f"window.ESC50_SHOWCASE = {serialized};\n", encoding="utf-8")
    print(f"Using LF + broad model: {model_dir.relative_to(ROOT)}")
    print(f"Exported {correct_count} correct and {wrong_count} incorrect test examples")
    print(f"Wrote {JSON_OUT.relative_to(ROOT)} and {JS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
