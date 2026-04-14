#!/usr/bin/env python3
"""Prepare AudioSet metadata files for HF-backed loading.

Audio samples are no longer reconstructed from YouTube or CSV metadata.
This utility only manages local class/mapping files used by training code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CLASSES = Path("data/audioset_classes.txt")
DEFAULT_OUT_ROOT = Path("data/audioset")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _load_classes(classes_txt: Path) -> list[str]:
    if not classes_txt.exists():
        raise FileNotFoundError(f"Missing AudioSet class file: {classes_txt}")
    with classes_txt.open("r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    if not classes:
        raise ValueError(f"No class names found in: {classes_txt}")
    return classes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AudioSet mapping metadata for HF-backed pipeline")
    parser.add_argument("--classes_txt", type=Path, default=DEFAULT_CLASSES, help="Class names file (one per line)")
    parser.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root for mapping files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = _load_classes(args.classes_txt)

    idx_to_label = {str(i): name for i, name in enumerate(classes)}
    label_to_idx = {name: i for i, name in enumerate(classes)}

    # MID-level metadata is not required in HF mode, but these compatibility files
    # preserve the shape expected by previous utilities.
    idx_to_mid = {str(i): f"/m/hf_{i:04d}" for i in range(len(classes))}
    mid_to_idx = {mid: int(idx) for idx, mid in idx_to_mid.items()}

    _write_json(args.out_root / "idx_to_display_name.json", idx_to_label)
    _write_json(args.out_root / "idx_to_mid.json", idx_to_mid)
    _write_json(args.out_root / "mid_to_idx.json", mid_to_idx)

    summary = {
        "num_classes": len(classes),
        "class_file": args.classes_txt.as_posix(),
        "source": "HF agkphysics/AudioSet",
        "note": "Audio clips are loaded lazily via datasets.load_dataset at runtime.",
    }
    _write_json(args.out_root / "summary.json", summary)

    print(f"Prepared AudioSet HF mapping files in: {args.out_root}")
    print(f"Number of classes: {len(classes)}")


if __name__ == "__main__":
    main()
