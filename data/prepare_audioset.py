#!/usr/bin/env python3
"""Prepare AudioSet manifests and label mappings.

This script supports balanced_train_segments and eval_segments CSVs from AudioSet.
It writes:
- class list and MID mappings
- balanced_train / eval manifests (JSONL)
- summary metrics including missing clip counts

The script is resilient to partial local clip availability: missing clips are skipped
unless --fail_on_missing is enabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _as_repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_clip_id(youtube_id: str, start_sec: float, end_sec: float) -> str:
    def _fmt(value: float) -> str:
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return str(value).replace(".", "p")

    return f"{youtube_id}_{_fmt(start_sec)}_{_fmt(end_sec)}"


def _read_wav_info(audio_path: Path) -> tuple[int, float]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
    duration = float(n_frames) / float(sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, duration


def _parse_labels_field(value: str) -> List[str]:
    raw = value.strip().strip('"')
    if not raw:
        return []
    labels = [token.strip() for token in raw.split(",") if token.strip()]
    return labels


def _find_local_clip(clips_root: Path, clip_id: str, youtube_id: str, extensions: Sequence[str]) -> Optional[Path]:
    for ext in extensions:
        candidate = clips_root / f"{clip_id}.{ext}"
        if candidate.exists():
            return candidate
    for ext in extensions:
        candidate = clips_root / f"{youtube_id}.{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_class_labels(class_labels_csv: Path) -> tuple[List[str], Dict[str, int], Dict[str, str]]:
    with class_labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "mid", "display_name"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"class_labels_indices.csv missing required columns: {sorted(missing)}")

        rows = []
        for row in reader:
            rows.append(row)

    rows = sorted(rows, key=lambda x: int(x["index"]))

    mids = [row["mid"].strip() for row in rows]
    display_names = [row["display_name"].strip() for row in rows]
    mid_to_idx = {mid: idx for idx, mid in enumerate(mids)}
    idx_to_mid = {str(idx): mid for mid, idx in mid_to_idx.items()}
    idx_to_display_name = {str(idx): display_names[idx] for idx in range(len(display_names))}
    return display_names, mid_to_idx, idx_to_display_name


def _parse_segment_csv(segment_csv: Path) -> List[Dict]:
    if not segment_csv.exists():
        raise FileNotFoundError(f"Missing segment CSV: {segment_csv}")

    parsed_rows: List[Dict] = []
    with segment_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue

            first = row[0].strip()
            if first == "YTID":
                continue

            if len(row) < 4:
                continue

            youtube_id = row[0].strip()
            try:
                start_sec = float(row[1].strip())
                end_sec = float(row[2].strip())
            except ValueError:
                continue

            labels_mid = _parse_labels_field(row[3])
            parsed_rows.append(
                {
                    "youtube_id": youtube_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "labels_mid": labels_mid,
                }
            )

    return parsed_rows


def _build_manifest_rows(
    split_name: str,
    rows: List[Dict],
    clips_root: Path,
    repo_root: Path,
    mid_to_idx: Dict[str, int],
    fail_on_missing: bool,
    extensions: Sequence[str],
) -> tuple[List[Dict], Dict[str, int]]:
    manifest_rows: List[Dict] = []
    stats = {
        "total_rows": len(rows),
        "missing_audio": 0,
        "missing_labels": 0,
        "written_rows": 0,
    }

    for row in rows:
        youtube_id = row["youtube_id"]
        start_sec = row["start_sec"]
        end_sec = row["end_sec"]
        labels_mid = [mid for mid in row["labels_mid"] if mid in mid_to_idx]
        if not labels_mid:
            stats["missing_labels"] += 1
            continue

        clip_id = _safe_clip_id(youtube_id, start_sec, end_sec)
        local_clip = _find_local_clip(clips_root, clip_id, youtube_id, extensions)

        if local_clip is None:
            stats["missing_audio"] += 1
            if fail_on_missing:
                raise FileNotFoundError(
                    f"Audio clip not found for {split_name} entry {clip_id} in {clips_root}"
                )
            continue

        sample_rate, duration = _read_wav_info(local_clip)
        label_idx = sorted({mid_to_idx[mid] for mid in labels_mid})

        manifest_rows.append(
            {
                "id": clip_id,
                "youtube_id": youtube_id,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "audio_path": _as_repo_relative(local_clip, repo_root),
                "labels_mid": labels_mid,
                "label_idx": label_idx,
                "sample_rate": sample_rate,
                "duration": round(duration, 6),
                "dataset": "audioset",
            }
        )

    stats["written_rows"] = len(manifest_rows)
    return manifest_rows, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AudioSet manifests and mappings")
    parser.add_argument("--class_labels_csv", type=Path, required=True, help="Path to class_labels_indices.csv")
    parser.add_argument("--balanced_csv", type=Path, required=True, help="Path to balanced_train_segments.csv")
    parser.add_argument("--eval_csv", type=Path, required=True, help="Path to eval_segments.csv")
    parser.add_argument("--clips_root", type=Path, required=True, help="Directory containing local audio clips")
    parser.add_argument("--out_root", type=Path, default=Path("data/audioset"), help="Output root for AudioSet metadata")
    parser.add_argument("--repo_root", type=Path, default=Path("."), help="Repository root for relative paths")
    parser.add_argument(
        "--extensions",
        type=str,
        default="wav,flac,mp3,m4a",
        help="Comma-separated list of clip extensions to search",
    )
    parser.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Fail immediately if any referenced clip is missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    display_names, mid_to_idx, idx_to_display_name = _parse_class_labels(args.class_labels_csv)
    idx_to_mid = {str(v): k for k, v in mid_to_idx.items()}

    classes_txt_path = out_root.parent / "audioset_classes.txt"
    classes_txt_path.write_text("\n".join(display_names) + "\n", encoding="utf-8")

    _write_json(out_root / "mid_to_idx.json", mid_to_idx)
    _write_json(out_root / "idx_to_mid.json", idx_to_mid)
    _write_json(out_root / "idx_to_display_name.json", idx_to_display_name)

    extensions = [ext.strip().lstrip(".") for ext in args.extensions.split(",") if ext.strip()]

    balanced_rows = _parse_segment_csv(args.balanced_csv)
    eval_rows = _parse_segment_csv(args.eval_csv)

    manifests_root = out_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    balanced_manifest, balanced_stats = _build_manifest_rows(
        split_name="balanced_train",
        rows=balanced_rows,
        clips_root=args.clips_root,
        repo_root=args.repo_root,
        mid_to_idx=mid_to_idx,
        fail_on_missing=args.fail_on_missing,
        extensions=extensions,
    )
    eval_manifest, eval_stats = _build_manifest_rows(
        split_name="eval",
        rows=eval_rows,
        clips_root=args.clips_root,
        repo_root=args.repo_root,
        mid_to_idx=mid_to_idx,
        fail_on_missing=args.fail_on_missing,
        extensions=extensions,
    )

    _write_jsonl(manifests_root / "balanced_train.jsonl", balanced_manifest)
    _write_jsonl(manifests_root / "eval.jsonl", eval_manifest)

    summary = {
        "num_classes": len(display_names),
        "balanced_train": balanced_stats,
        "eval": eval_stats,
        "clips_root": args.clips_root.resolve().as_posix(),
        "extensions": extensions,
    }
    _write_json(out_root / "summary.json", summary)

    print(f"Wrote AudioSet metadata to: {out_root}")
    print(f"Class list: {classes_txt_path}")
    print(f"Balanced rows written: {balanced_stats['written_rows']}/{balanced_stats['total_rows']}")
    print(f"Eval rows written: {eval_stats['written_rows']}/{eval_stats['total_rows']}")


if __name__ == "__main__":
    main()
