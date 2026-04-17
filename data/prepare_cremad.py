#!/usr/bin/env python3
"""Prepare CREMA-D manifests and label mappings."""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from sklearn.model_selection import train_test_split


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


def _resolve_layout(cremad_root: Path) -> tuple[Path, Path, Path]:
    direct = cremad_root
    nested = cremad_root / "CREMA-D"

    for candidate in [direct, nested]:
        train_csv = candidate / "train.csv"
        test_csv = candidate / "test.csv"
        audio_dir = candidate / "audios"
        if train_csv.exists() and test_csv.exists() and audio_dir.exists():
            return train_csv, test_csv, audio_dir

    raise FileNotFoundError(
        "Could not find CREMA-D layout under {}. Expected train.csv, test.csv, and audios/.".format(cremad_root)
    )


def _looks_like_header(cell_a: str, cell_b: str) -> bool:
    a = str(cell_a).strip().lower()
    b = str(cell_b).strip().lower()
    path_aliases = {"path", "audio_path", "audio", "file", "filename"}
    label_aliases = {"classname", "class", "label", "emotion", "category"}
    return a in path_aliases and b in label_aliases


def _read_split_csv(split_csv: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if len(r) >= 2]

    if len(raw_rows) == 0:
        raise ValueError("No rows found in split file: {}".format(split_csv))

    start = 1 if _looks_like_header(raw_rows[0][0], raw_rows[0][1]) else 0

    for i in range(start, len(raw_rows)):
        rel_audio = str(raw_rows[i][0]).strip()
        label = str(raw_rows[i][1]).strip().lower()
        if not rel_audio or not label:
            continue
        rows.append((rel_audio.replace("\\", "/"), label))

    if len(rows) == 0:
        raise ValueError("No valid rows parsed from split file: {}".format(split_csv))
    return rows


def _resolve_audio_path(root_dir: Path, audio_dir: Path, rel_audio: str) -> Path:
    rel = Path(rel_audio)
    candidates = [
        (root_dir / rel).resolve(),
        (audio_dir / rel).resolve(),
        (audio_dir / rel.name).resolve(),
        (root_dir / rel.name).resolve(),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("CREMA-D audio file not found for '{}' under {}".format(rel_audio, root_dir))


def _split_train_val(
    rows: Sequence[Tuple[str, str]],
    val_fraction: float,
    split_seed: int,
) -> tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1), got {}".format(val_fraction))

    indices = list(range(len(rows)))
    labels = [rows[i][1] for i in indices]

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=float(val_fraction),
            random_state=int(split_seed),
            shuffle=True,
            stratify=labels,
        )
    except ValueError:
        # fallback if some classes are too rare for strict stratification
        train_idx, val_idx = train_test_split(
            indices,
            test_size=float(val_fraction),
            random_state=int(split_seed),
            shuffle=True,
            stratify=None,
        )

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    return train_rows, val_rows


def _to_manifest_rows(
    split_rows: Sequence[Tuple[str, str]],
    split_name: str,
    class_to_idx: Dict[str, int],
    root_dir: Path,
    audio_dir: Path,
    repo_root: Path,
) -> List[Dict]:
    out_rows: List[Dict] = []
    for rel_audio, label in split_rows:
        if label not in class_to_idx:
            raise ValueError("Unknown label '{}' in split {}".format(label, split_name))

        audio_path = _resolve_audio_path(root_dir, audio_dir, rel_audio)
        sample_rate, duration = _read_wav_info(audio_path)

        out_rows.append(
            {
                "id": audio_path.stem,
                "audio_path": _as_repo_relative(audio_path, repo_root),
                "label": label,
                "label_idx": int(class_to_idx[label]),
                "split": split_name,
                "sample_rate": sample_rate,
                "duration": round(duration, 6),
                "dataset": "cremad",
            }
        )
    return out_rows


def _count_labels(rows: Sequence[Dict], idx_to_label: Dict[str, str]) -> Dict[str, int]:
    counts = {label: 0 for label in idx_to_label.values()}
    for row in rows:
        label = str(row["label"])
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_manifests(
    cremad_root: Path,
    out_root: Path,
    repo_root: Path,
    val_fraction: float,
    split_seed: int,
) -> None:
    train_csv, test_csv, audio_dir = _resolve_layout(cremad_root)
    root_dir = train_csv.parent

    official_train_rows = _read_split_csv(train_csv)
    official_test_rows = _read_split_csv(test_csv)

    if len(official_train_rows) < 2:
        raise ValueError("CREMA-D train split needs at least 2 samples to build a validation split")

    train_rows_raw, val_rows_raw = _split_train_val(
        official_train_rows,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    all_labels = sorted({label for _, label in (official_train_rows + official_test_rows)})
    if len(all_labels) == 0:
        raise ValueError("No labels found in CREMA-D metadata")

    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    idx_to_label = {str(i): label for label, i in label_to_idx.items()}

    train_rows = _to_manifest_rows(train_rows_raw, "train", label_to_idx, root_dir, audio_dir, repo_root)
    val_rows = _to_manifest_rows(val_rows_raw, "val", label_to_idx, root_dir, audio_dir, repo_root)
    test_rows = _to_manifest_rows(official_test_rows, "test", label_to_idx, root_dir, audio_dir, repo_root)

    all_rows = train_rows + val_rows + test_rows

    manifests_root = out_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    _write_jsonl(manifests_root / "all.jsonl", all_rows)
    _write_jsonl(manifests_root / "train.jsonl", train_rows)
    _write_jsonl(manifests_root / "val.jsonl", val_rows)
    _write_jsonl(manifests_root / "test.jsonl", test_rows)

    classes_txt_path = out_root.parent / "cremad_classes.txt"
    classes_txt_path.parent.mkdir(parents=True, exist_ok=True)
    classes_txt_path.write_text("\n".join(all_labels) + "\n", encoding="utf-8")

    _write_json(out_root / "label_to_idx.json", label_to_idx)
    _write_json(out_root / "idx_to_label.json", idx_to_label)
    _write_json(
        out_root / "summary.json",
        {
            "num_samples": len(all_rows),
            "num_classes": len(all_labels),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "official_train_rows": len(official_train_rows),
            "official_test_rows": len(official_test_rows),
            "val_fraction": float(val_fraction),
            "split_seed": int(split_seed),
            "class_counts_train": _count_labels(train_rows, idx_to_label),
            "class_counts_val": _count_labels(val_rows, idx_to_label),
            "class_counts_test": _count_labels(test_rows, idx_to_label),
        },
    )

    print("Wrote CREMA-D metadata to:", out_root)
    print("Class list:", classes_txt_path)
    print("Total samples:", len(all_rows))
    print("Split sizes -> train: {} val: {} test: {}".format(len(train_rows), len(val_rows), len(test_rows)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CREMA-D manifests and mappings")
    parser.add_argument(
        "--cremad_root",
        type=Path,
        default=Path("data/cremad/raw"),
        help="Path to CREMA-D root or parent containing CREMA-D/",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/cremad"),
        help="Output directory for CREMA-D metadata artifacts",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path("."),
        help="Repository root used to write repo-relative audio paths",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Validation fraction sampled from official train split",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifests(
        cremad_root=args.cremad_root,
        out_root=args.out_root,
        repo_root=args.repo_root,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
