#!/usr/bin/env python3
"""Prepare ESC-50 manifests and label mappings.

This script reads the official ESC-50 metadata CSV and creates:
- class list file
- label_to_idx / idx_to_label mappings
- per-fold train/val/test manifests (JSONL)
- optional default split manifests

Expected ESC-50 layout:
  <esc50_root>/meta/esc50.csv
  <esc50_root>/audio/*.wav
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Dict, List


def _read_wav_info(audio_path: Path) -> tuple[int, float]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
    duration = float(n_frames) / float(sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, duration


def _as_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_manifests(
    esc50_root: Path,
    out_root: Path,
    repo_root: Path,
    val_fold_offset: int,
    write_default_split: bool,
    default_test_fold: int,
) -> None:
    meta_csv = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"

    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing ESC-50 metadata file: {meta_csv}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing ESC-50 audio directory: {audio_dir}")

    rows = []
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"filename", "fold", "target", "category", "esc10", "src_file", "take"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"esc50.csv missing required columns: {sorted(missing)}")

        for row in reader:
            rows.append(row)

    target_to_label = {}
    for row in rows:
        target = int(row["target"])
        label = row["category"].strip()
        if target in target_to_label and target_to_label[target] != label:
            raise ValueError(
                f"Inconsistent target mapping for target={target}: {target_to_label[target]} vs {label}"
            )
        target_to_label[target] = label

    class_names = [target_to_label[i] for i in sorted(target_to_label.keys())]
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_label = {str(i): name for name, i in label_to_idx.items()}

    manifests_root = out_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    all_samples: List[Dict] = []
    for row in rows:
        filename = row["filename"].strip()
        fold = int(row["fold"])
        label = row["category"].strip()
        audio_path = (audio_dir / filename).resolve()

        if not audio_path.exists():
            raise FileNotFoundError(f"ESC-50 audio file not found: {audio_path}")

        sample_rate, duration = _read_wav_info(audio_path)
        sample_id = audio_path.stem

        sample = {
            "id": sample_id,
            "audio_path": _as_repo_relative(audio_path, repo_root),
            "label": label,
            "label_idx": int(label_to_idx[label]),
            "fold": fold,
            "sample_rate": sample_rate,
            "duration": round(duration, 6),
            "dataset": "esc50",
        }
        all_samples.append(sample)

    _write_jsonl(manifests_root / "all.jsonl", all_samples)

    for test_fold in range(1, 6):
        val_fold = ((test_fold + val_fold_offset - 1) % 5) + 1
        train_rows = [s for s in all_samples if s["fold"] not in {test_fold, val_fold}]
        val_rows = [s for s in all_samples if s["fold"] == val_fold]
        test_rows = [s for s in all_samples if s["fold"] == test_fold]

        _write_jsonl(manifests_root / f"fold{test_fold}_train.jsonl", train_rows)
        _write_jsonl(manifests_root / f"fold{test_fold}_val.jsonl", val_rows)
        _write_jsonl(manifests_root / f"fold{test_fold}_test.jsonl", test_rows)

    if write_default_split:
        _write_jsonl(manifests_root / "train.jsonl", [s for s in all_samples if s["fold"] not in {default_test_fold, ((default_test_fold + val_fold_offset - 1) % 5) + 1}])
        _write_jsonl(manifests_root / "val.jsonl", [s for s in all_samples if s["fold"] == ((default_test_fold + val_fold_offset - 1) % 5) + 1])
        _write_jsonl(manifests_root / "test.jsonl", [s for s in all_samples if s["fold"] == default_test_fold])

    classes_txt_path = out_root.parent / "esc50_classes.txt"
    classes_txt_path.parent.mkdir(parents=True, exist_ok=True)
    classes_txt_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    _write_json(out_root / "label_to_idx.json", label_to_idx)
    _write_json(out_root / "idx_to_label.json", idx_to_label)

    summary = {
        "num_samples": len(all_samples),
        "num_classes": len(class_names),
        "default_test_fold": default_test_fold,
        "val_fold_offset": val_fold_offset,
    }
    _write_json(out_root / "summary.json", summary)

    print(f"Wrote ESC-50 metadata to: {out_root}")
    print(f"Class list: {classes_txt_path}")
    print(f"Total samples: {len(all_samples)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ESC-50 manifests and mappings")
    parser.add_argument(
        "--esc50_root",
        type=Path,
        required=True,
        help="Path to ESC-50 root containing meta/esc50.csv and audio/",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/esc50"),
        help="Output directory for ESC-50 metadata artifacts",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root used to write repo-relative audio paths when possible",
    )
    parser.add_argument(
        "--val_fold_offset",
        type=int,
        default=1,
        help="Validation fold is (test_fold + val_fold_offset) mod 5",
    )
    parser.add_argument(
        "--default_test_fold",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Fold used for top-level train/val/test manifests",
    )
    parser.add_argument(
        "--no_default_split",
        action="store_true",
        help="Do not write train.jsonl/val.jsonl/test.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifests(
        esc50_root=args.esc50_root,
        out_root=args.out_root,
        repo_root=args.repo_root,
        val_fold_offset=args.val_fold_offset,
        write_default_split=not args.no_default_split,
        default_test_fold=args.default_test_fold,
    )


if __name__ == "__main__":
    main()
