#!/usr/bin/env python3
"""Build static ESC-50 showcase assets for GitHub Pages.

This script selects 2 deterministic examples per ESC-50 class, copies audio clips
into docs/assets/audio, and precomputes top concept contribution bars (CBM) so the
website can render notebook-like explanations without backend inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cbm
import data_utils

ESC50_CLASSES_PATH = ROOT / "data" / "esc50_classes.txt"
ESC50_MANIFEST_PATH = ROOT / "data" / "esc50" / "manifests" / "all.jsonl"
DOCS_DIR = ROOT / "docs"
AUDIO_OUT_DIR = DOCS_DIR / "assets" / "audio"
IMAGE_OUT_DIR = DOCS_DIR / "assets" / "images" / "spectrograms"
DATA_OUT_DIR = DOCS_DIR / "assets" / "data"
MANIFEST_OUT_PATH = DATA_OUT_DIR / "esc50_showcase.json"
MANIFEST_JS_OUT_PATH = DATA_OUT_DIR / "esc50_showcase.js"


EMOJI_MAP = {
    "dog": "🐶",
    "rooster": "🐓",
    "pig": "🐷",
    "cow": "🐄",
    "frog": "🐸",
    "cat": "🐱",
    "hen": "🐔",
    "insects": "🐞",
    "sheep": "🐑",
    "crow": "🐦",
    "rain": "🌧️",
    "sea_waves": "🌊",
    "crackling_fire": "🔥",
    "crickets": "🦗",
    "chirping_birds": "🐤",
    "water_drops": "💧",
    "wind": "🌬️",
    "pouring_water": "🚿",
    "toilet_flush": "🚽",
    "thunderstorm": "⛈️",
    "crying_baby": "👶",
    "sneezing": "🤧",
    "clapping": "👏",
    "breathing": "😮",
    "coughing": "😷",
    "footsteps": "👣",
    "laughing": "😂",
    "brushing_teeth": "🪥",
    "snoring": "😴",
    "drinking_sipping": "🥤",
    "door_wood_knock": "🚪",
    "mouse_click": "🖱️",
    "keyboard_typing": "⌨️",
    "door_wood_creaks": "🚪",
    "can_opening": "🥫",
    "washing_machine": "🧺",
    "vacuum_cleaner": "🧹",
    "clock_alarm": "⏰",
    "clock_tick": "🕒",
    "glass_breaking": "🥛",
    "helicopter": "🚁",
    "chainsaw": "🪚",
    "siren": "🚨",
    "car_horn": "🚗",
    "engine": "⚙️",
    "train": "🚆",
    "church_bells": "🔔",
    "airplane": "✈️",
    "fireworks": "🎆",
    "hand_saw": "🪚",
}


@dataclass(frozen=True)
class EscSample:
    sample_id: str
    label: str
    label_idx: int
    fold: int
    sample_rate: int
    duration: float
    audio_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESC-50 static showcase assets")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of examples per class (default: 2)",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=10,
        help="Number of top concepts to keep per sample (default: 10)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to a specific esc50_cbm_* directory; defaults to latest by mtime",
    )
    return parser.parse_args()


def load_classes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing class file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_samples(path: Path) -> list[EscSample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest file: {path}")

    samples: list[EscSample] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            samples.append(
                EscSample(
                    sample_id=rec["id"],
                    label=rec["label"],
                    label_idx=int(rec["label_idx"]),
                    fold=int(rec["fold"]),
                    sample_rate=int(rec["sample_rate"]),
                    duration=float(rec["duration"]),
                    audio_path=ROOT / rec["audio_path"],
                )
            )
    return samples


def find_latest_model_dir() -> Path:
    root = ROOT / "saved_models"
    candidates = [p for p in root.glob("esc50_cbm_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No ESC-50 CBM checkpoints found under saved_models/esc50_cbm_*")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_concepts(model_dir: Path) -> list[str]:
    concept_path = model_dir / "concepts.txt"
    if not concept_path.exists():
        raise FileNotFoundError(f"Missing concepts file: {concept_path}")
    return [line.strip() for line in concept_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_examples_per_class(samples: list[EscSample], samples_per_class: int) -> dict[str, list[EscSample]]:
    grouped: dict[str, list[EscSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    selected: dict[str, list[EscSample]] = {}

    for label, group in grouped.items():
        group_sorted = sorted(group, key=lambda s: (s.fold, s.sample_id))

        by_fold: dict[int, list[EscSample]] = defaultdict(list)
        for sample in group_sorted:
            by_fold[sample.fold].append(sample)

        picked: list[EscSample] = []
        used_ids: set[str] = set()

        for fold in sorted(by_fold.keys()):
            cand = by_fold[fold][0]
            if cand.sample_id not in used_ids:
                picked.append(cand)
                used_ids.add(cand.sample_id)
            if len(picked) == samples_per_class:
                break

        if len(picked) < samples_per_class:
            for sample in group_sorted:
                if sample.sample_id in used_ids:
                    continue
                picked.append(sample)
                used_ids.add(sample.sample_id)
                if len(picked) == samples_per_class:
                    break

        if len(picked) != samples_per_class:
            raise RuntimeError(
                f"Could not select {samples_per_class} samples for class '{label}'. Found only {len(picked)}."
            )

        selected[label] = picked

    return selected


def prepare_audio(audio_path: Path, sample_rate: int = 16000, duration_sec: float = 5.0) -> tuple[torch.Tensor, int]:
    audio, sr = data_utils._load_wav_audio(audio_path, target_sample_rate=sample_rate, mono=True)
    audio = data_utils._pad_or_truncate(audio, sr, duration_sec)
    return audio.float(), int(sr)


def compute_explanation(
    model: torch.nn.Module,
    concepts: list[str],
    idx_to_class: dict[int, str],
    audio: torch.Tensor,
    gt_idx: int,
    device: str,
    max_concepts: int,
) -> dict:
    with torch.no_grad():
        logits, concept_act = model(audio.unsqueeze(0).to(device))

    logits = logits[0].detach().cpu()
    concept_act = concept_act[0].detach().cpu()
    probs = torch.softmax(logits, dim=0)
    pred_idx = int(torch.argmax(probs).item())

    contrib = (concept_act * model.final.weight[pred_idx].detach().cpu()).numpy()
    feature_names = [(("NOT " if concept_act[i].item() < 0 else "") + concepts[i]) for i in range(len(concepts))]

    order = np.argsort(np.abs(contrib))[::-1][:max_concepts].tolist()
    top_concepts = [
        {
            "concept": feature_names[i],
            "score": float(contrib[i]),
        }
        for i in order
    ]

    return {
        "gt_class": idx_to_class[int(gt_idx)],
        "pred_class": idx_to_class[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "top_concepts": top_concepts,
    }


def main() -> None:
    args = parse_args()

    classes = load_classes(ESC50_CLASSES_PATH)
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    model_dir = Path(args.model_dir) if args.model_dir else find_latest_model_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cbm.load_cbm(str(model_dir), device).to(device).eval()
    concepts = load_concepts(model_dir)

    all_samples = load_samples(ESC50_MANIFEST_PATH)
    selected = select_examples_per_class(all_samples, args.samples_per_class)

    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old generated assets for deterministic rebuilds.
    for p in AUDIO_OUT_DIR.glob("*.wav"):
        p.unlink()

    # Spectrograms are no longer used in the showcase. Clean stale files if present.
    if IMAGE_OUT_DIR.exists():
        for p in IMAGE_OUT_DIR.glob("*.png"):
            p.unlink()

    payload_classes = []

    for label_idx, label in enumerate(classes):
        if label not in selected:
            raise RuntimeError(f"No selected samples for class: {label}")

        examples_json = []
        for sample in selected[label]:
            if not sample.audio_path.exists():
                raise FileNotFoundError(f"Audio file missing: {sample.audio_path}")

            out_audio_name = f"{sample.sample_id}.wav"
            out_audio_path = AUDIO_OUT_DIR / out_audio_name
            shutil.copy2(sample.audio_path, out_audio_path)

            audio, sr = prepare_audio(sample.audio_path, sample_rate=16000, duration_sec=5.0)
            explanation = compute_explanation(
                model=model,
                concepts=concepts,
                idx_to_class=idx_to_class,
                audio=audio,
                gt_idx=sample.label_idx,
                device=device,
                max_concepts=args.max_concepts,
            )

            examples_json.append(
                {
                    "id": sample.sample_id,
                    "fold": sample.fold,
                    "duration_sec": sample.duration,
                    "audio": f"assets/audio/{out_audio_name}",
                    "sample_rate": sr,
                    "explanation": explanation,
                }
            )

        payload_classes.append(
            {
                "label": label,
                "label_idx": label_idx,
                "emoji": EMOJI_MAP.get(label, "🔊"),
                "examples": examples_json,
            }
        )

    manifest = {
        "dataset": "esc50",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "samples_per_class": args.samples_per_class,
        "max_concepts": args.max_concepts,
        "num_classes": len(payload_classes),
        "model_dir": str(model_dir),
        "classes": payload_classes,
    }

    MANIFEST_OUT_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    MANIFEST_JS_OUT_PATH.write_text(
        "window.ESC50_SHOWCASE = " + json.dumps(manifest, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

    print(f"Using model: {model_dir}")
    print(f"Wrote showcase manifest: {MANIFEST_OUT_PATH}")
    print(f"Wrote showcase JS manifest: {MANIFEST_JS_OUT_PATH}")
    print(f"Copied audio files to: {AUDIO_OUT_DIR}")


if __name__ == "__main__":
    main()
