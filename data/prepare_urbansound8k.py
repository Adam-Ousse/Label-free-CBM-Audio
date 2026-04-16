#!/usr/bin/env python3
"""Prepare UrbanSound8K manifests and label mappings."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Dict, List


FIXED_TEST_FOLD = 10
FIXED_VAL_FOLD = 9
TRAIN_FOLDS = {1, 2, 3, 4, 5, 6, 7, 8}


def _read_wav_info(audio_path: Path) -> tuple[int, float]:
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
    except Exception:
        try:
            import soundfile as sf

            info = sf.info(str(audio_path))
            sample_rate = int(info.samplerate)
            n_frames = int(info.frames)
        except Exception:
            from scipy.io import wavfile

            sample_rate, audio = wavfile.read(str(audio_path))
            n_frames = int(audio.shape[0])
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


def _has_fold_dirs(base_dir: Path) -> bool:
    for fold in range(1, 11):
        if not (base_dir / "fold{}".format(fold)).exists():
            return False
    return True


def _resolve_layout(urbansound8k_root: Path) -> tuple[Path, Path]:
    direct = urbansound8k_root
    nested = urbansound8k_root / "UrbanSound8K"

    for candidate in [direct, nested]:
        # layout a: metadata/UrbanSound8K.csv + audio/fold1..fold10
        meta_nested = candidate / "metadata" / "UrbanSound8K.csv"
        audio_nested = candidate / "audio"
        if meta_nested.exists() and audio_nested.exists() and _has_fold_dirs(audio_nested):
            return meta_nested, audio_nested

        # layout b (kaggle flat): UrbanSound8K.csv + fold1..fold10 at root
        meta_flat = candidate / "UrbanSound8K.csv"
        if meta_flat.exists() and _has_fold_dirs(candidate):
            return meta_flat, candidate

    raise FileNotFoundError(
        "Could not find UrbanSound8K layout under {}. Expected either metadata/UrbanSound8K.csv + audio/fold1..fold10 or UrbanSound8K.csv + fold1..fold10".format(
            urbansound8k_root
        )
    )


def build_manifests(urbansound8k_root: Path, out_root: Path, repo_root: Path, write_default_split: bool) -> None:
    meta_csv, audio_dir = _resolve_layout(urbansound8k_root)

    rows = []
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"slice_file_name", "fold", "classID", "class"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError("UrbanSound8K.csv missing required columns: {}".format(sorted(missing)))

        for row in reader:
            rows.append(row)

    class_id_to_label = {}
    for row in rows:
        class_id = int(row["classID"])
        label = row["class"].strip()
        if class_id in class_id_to_label and class_id_to_label[class_id] != label:
            raise ValueError(
                "Inconsistent class mapping for classID={}: {} vs {}".format(
                    class_id,
                    class_id_to_label[class_id],
                    label,
                )
            )
        class_id_to_label[class_id] = label

    ordered_class_ids = sorted(class_id_to_label.keys())
    class_names = [class_id_to_label[idx] for idx in ordered_class_ids]
    label_to_idx = {label: i for i, label in enumerate(class_names)}
    idx_to_label = {str(i): label for label, i in label_to_idx.items()}

    all_samples: List[Dict] = []
    for row in rows:
        fold = int(row["fold"])
        filename = row["slice_file_name"].strip()
        label = row["class"].strip()
        class_id = int(row["classID"])

        fold_dir = audio_dir / "fold{}".format(fold)
        audio_path = (fold_dir / filename).resolve()
        if not audio_path.exists():
            raise FileNotFoundError("UrbanSound8K audio file not found: {}".format(audio_path))

        sample_rate, duration = _read_wav_info(audio_path)
        sample = {
            "id": audio_path.stem,
            "audio_path": _as_repo_relative(audio_path, repo_root),
            "label": label,
            "label_idx": int(label_to_idx[label]),
            "class_id": class_id,
            "fold": fold,
            "sample_rate": sample_rate,
            "duration": round(duration, 6),
            "dataset": "urbansound8k",
        }
        all_samples.append(sample)

    manifests_root = out_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    train_rows = [s for s in all_samples if int(s["fold"]) in TRAIN_FOLDS]
    val_rows = [s for s in all_samples if int(s["fold"]) == FIXED_VAL_FOLD]
    test_rows = [s for s in all_samples if int(s["fold"]) == FIXED_TEST_FOLD]

    _write_jsonl(manifests_root / "all.jsonl", all_samples)
    _write_jsonl(manifests_root / "fold10_train.jsonl", train_rows)
    _write_jsonl(manifests_root / "fold10_val.jsonl", val_rows)
    _write_jsonl(manifests_root / "fold10_test.jsonl", test_rows)

    if write_default_split:
        _write_jsonl(manifests_root / "train.jsonl", train_rows)
        _write_jsonl(manifests_root / "val.jsonl", val_rows)
        _write_jsonl(manifests_root / "test.jsonl", test_rows)

    classes_txt_path = out_root.parent / "urbansound8k_classes.txt"
    classes_txt_path.parent.mkdir(parents=True, exist_ok=True)
    classes_txt_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    _write_json(out_root / "label_to_idx.json", label_to_idx)
    _write_json(out_root / "idx_to_label.json", idx_to_label)
    _write_json(
        out_root / "summary.json",
        {
            "num_samples": len(all_samples),
            "num_classes": len(class_names),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_folds": sorted(TRAIN_FOLDS),
            "val_fold": FIXED_VAL_FOLD,
            "test_fold": FIXED_TEST_FOLD,
        },
    )

    print("Wrote UrbanSound8K metadata to:", out_root)
    print("Class list:", classes_txt_path)
    print("Total samples:", len(all_samples))
    print("Split sizes -> train: {} val: {} test: {}".format(len(train_rows), len(val_rows), len(test_rows)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UrbanSound8K manifests and mappings")
    parser.add_argument(
        "--urbansound8k_root",
        type=Path,
        default=Path("data/urbansound8k/raw"),
        help="Path to UrbanSound8K root or parent containing UrbanSound8K/",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/urbansound8k"),
        help="Output directory for UrbanSound8K metadata artifacts",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root used to write repo-relative audio paths",
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
        urbansound8k_root=args.urbansound8k_root,
        out_root=args.out_root,
        repo_root=args.repo_root,
        write_default_split=not args.no_default_split,
    )


if __name__ == "__main__":
    main()
