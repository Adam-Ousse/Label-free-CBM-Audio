#!/usr/bin/env python3
"""Load AudioSet directly from Hugging Face Datasets.

This replaces the old yt-dlp/ffmpeg reconstruction pipeline. Audio and labels are
read from `agkphysics/AudioSet`, which already provides decoded waveforms.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Iterable

# Disable TorchCodec audio decoding (unavailable on cluster due to missing CUDA deps)
os.environ.setdefault("HF_DATASETS_DISABLE_TORCHCODEC", "1")

from datasets import Audio, load_dataset

HF_DATASET_ID = "agkphysics/AudioSet"
SUBSET_SPLITS = {
    "balanced": {"train", "test"},
    "unbalanced": {"train", "test"},
    "full": {"bal_train", "unbal_train", "eval"},
}

DATA_GLOB_BY_SUBSET_SPLIT = {
    ("balanced", "train"): "data/bal_train/*.parquet",
    ("balanced", "test"): "data/eval/*.parquet",
    ("unbalanced", "train"): "data/unbal_train/*.parquet",
    ("unbalanced", "test"): "data/eval/*.parquet",
    ("full", "bal_train"): "data/bal_train/*.parquet",
    ("full", "unbal_train"): "data/unbal_train/*.parquet",
    ("full", "eval"): "data/eval/*.parquet",
}

TRAIN_LIKE_SPLITS = {"train", "bal_train", "unbal_train"}


def resolve_subset_split(split: str, subset: str | None = None) -> tuple[str, str]:
    split_key = split.strip().lower()
    subset_key = subset.strip().lower() if subset is not None else None

    if subset_key is None and ":" in split_key:
        subset_key, split_key = split_key.split(":", 1)

    if subset_key is None:
        legacy = {
            "balanced": ("balanced", "train"),
            "balanced_train": ("balanced", "train"),
            "unbalanced": ("unbalanced", "train"),
            "unbalanced_train": ("unbalanced", "train"),
            "eval": ("full", "eval"),
            "validation": ("full", "eval"),
            "valid": ("full", "eval"),
            "bal_train": ("full", "bal_train"),
            "unbal_train": ("full", "unbal_train"),
            "train": ("balanced", "train"),
            "test": ("balanced", "test"),
        }
        if split_key in legacy:
            return legacy[split_key]
        subset_key = "balanced"

    if subset_key not in SUBSET_SPLITS:
        raise ValueError("Unknown subset '{}'. Use balanced, unbalanced, or full".format(subset_key))

    if subset_key in {"balanced", "unbalanced"} and split_key in {"eval", "validation", "valid"}:
        split_key = "test"
    elif subset_key == "full" and split_key in {"test", "validation", "valid"}:
        split_key = "eval"

    if split_key not in SUBSET_SPLITS[subset_key]:
        valid = ", ".join(sorted(SUBSET_SPLITS[subset_key]))
        raise ValueError(
            "Invalid split '{}' for subset '{}'. Valid splits: {}".format(split_key, subset_key, valid)
        )

    return subset_key, split_key


def iter_examples(ds: Iterable, max_items: int | None):
    count = 0
    for example in ds:
        yield example
        count += 1
        if max_items is not None and count >= max_items:
            break


def sanitize_filename(value: str) -> str:
    keep = []
    for c in value:
        if c.isalnum() or c in {"-", "_", "."}:
            keep.append(c)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "sample"


def write_audio_bytes(path: Path, audio_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio_bytes)


def resolve_data_glob(subset: str, split: str) -> str:
    key = (subset, split)
    if key not in DATA_GLOB_BY_SUBSET_SPLIT:
        raise ValueError("No parquet mapping for subset/split: {}/{}".format(subset, split))
    return DATA_GLOB_BY_SUBSET_SPLIT[key]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/cache AudioSet from Hugging Face datasets")
    parser.add_argument("--subset", type=str, default=None, help="Subset: balanced, unbalanced, or full")
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Split inside subset. Examples: balanced/test, unbalanced/test, full/eval",
    )
    parser.add_argument("--streaming", action="store_true", help="Enable Hugging Face streaming mode")
    parser.add_argument(
        "--decode_audio",
        action="store_true",
        help="Decode waveform arrays from the audio column (requires TorchCodec runtime dependencies)",
    )
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache directory")
    parser.add_argument("--max_items", type=int, default=64, help="How many samples to inspect")
    parser.add_argument(
        "--export_dir",
        type=Path,
        default=None,
        help="Optional directory to save inspected examples as WAV files",
    )
    parser.add_argument(
        "--export_limit",
        type=int,
        default=None,
        help="Maximum number of WAV files to export (defaults to max_items)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite WAV files if they already exist")
    parser.add_argument(
        "--allow_train_splits",
        action="store_true",
        help="Allow train-like splits (train, bal_train, unbal_train). Default blocks them to save disk.",
    )
    parser.add_argument(
        "--summary_out",
        type=Path,
        default=Path("data/audioset/summary_hf.json"),
        help="Where to write a small JSON summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset, split = resolve_subset_split(args.split, subset=args.subset)
    if split in TRAIN_LIKE_SPLITS and not args.allow_train_splits:
        raise ValueError(
            "Blocked split '{}'. This command is configured for eval/test only to save space. "
            "Use --allow_train_splits to override.".format(split)
        )

    data_glob = resolve_data_glob(subset, split)
    data_files = {"requested": "hf://datasets/{}/{}".format(HF_DATASET_ID, data_glob)}

    export_requested = args.export_dir is not None
    should_decode = bool(args.decode_audio)
    export_limit = args.max_items if args.export_limit is None else max(0, args.export_limit)

    ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="requested",
        streaming=args.streaming,
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
    )

    # Keep decoding disabled by default to avoid TorchCodec runtime failures on systems
    # without matching FFmpeg/CUDA shared libraries.
    if not should_decode:
        ds = ds.cast_column("audio", Audio(decode=False))

    inspected = 0
    sample_rates = {}
    min_len = None
    max_len = None
    first_ids = []
    decoded_audio = bool(should_decode)
    exported = 0
    skipped_export_no_audio = 0
    exported_extensions = {}

    if export_requested:
        args.export_dir.mkdir(parents=True, exist_ok=True)

    try:
        for item in iter_examples(ds, args.max_items):
            audio = item.get("audio", {})
            arr = audio.get("array", []) if isinstance(audio, dict) else []
            sr = int(audio.get("sampling_rate", 0)) if isinstance(audio, dict) else 0

            # If decode fails unexpectedly or was disabled, keep summary generation working.
            if isinstance(audio, dict) and "array" not in audio:
                decoded_audio = False

            inspected += 1
            sample_rates[str(sr)] = sample_rates.get(str(sr), 0) + 1

            n = len(arr) if hasattr(arr, "__len__") else 0
            min_len = n if min_len is None else min(min_len, n)
            max_len = n if max_len is None else max(max_len, n)

            if len(first_ids) < 10:
                first_ids.append(str(item.get("video_id", f"idx_{inspected - 1}")))

            if export_requested and exported < export_limit:
                audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
                source_path = str(audio.get("path", "")) if isinstance(audio, dict) else ""
                suffix = Path(source_path).suffix or ".flac"

                if not audio_bytes:
                    skipped_export_no_audio += 1
                else:
                    video_id = str(item.get("video_id", f"idx_{inspected - 1}"))
                    stem = f"{inspected - 1:06d}_{sanitize_filename(video_id)}"
                    out_path = args.export_dir / f"{stem}{suffix}"
                    if out_path.exists() and not args.overwrite:
                        continue
                    write_audio_bytes(out_path, audio_bytes)
                    exported += 1
                    exported_extensions[suffix] = exported_extensions.get(suffix, 0) + 1
    finally:
        # Suppress known HF streaming cleanup warnings when closing the dataset.
        # Export is already complete and saved to disk; this is cosmetic cleanup noise.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Bad file descriptor.*")

        payload = {
            "dataset": HF_DATASET_ID,
            "subset": subset,
            "split": split,
            "data_glob": data_glob,
            "streaming": bool(args.streaming),
            "decode_audio": bool(args.decode_audio),
            "effective_decode_audio": bool(should_decode),
            "decoded_audio_observed": decoded_audio,
            "inspected_examples": inspected,
            "sample_rates": sample_rates,
            "min_num_samples": min_len,
            "max_num_samples": max_len,
            "example_video_ids": first_ids,
            "schema": {
                "audio": {"array": "waveform", "sampling_rate": "int"},
                "labels": "list[int]",
                "human_labels": "list[str]",
                "video_id": "str",
            },
            "export": {
                "requested": export_requested,
                "dir": str(args.export_dir) if args.export_dir is not None else None,
                "export_limit": export_limit if export_requested else None,
                "exported_audio_files": exported,
                "skipped_missing_audio": skipped_export_no_audio,
                "exported_extensions": exported_extensions,
                "overwrite": bool(args.overwrite),
            },
        }

        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print("AudioSet Hugging Face summary")
        print(f"  dataset: {HF_DATASET_ID}")
        print(f"  subset: {subset}")
        print(f"  split: {split}")
        print(f"  data_glob: {data_glob}")
        print(f"  streaming: {args.streaming}")
        print(f"  inspected_examples: {inspected}")
        print(f"  sample_rates: {sample_rates}")
        if export_requested:
            print(f"  export_dir: {args.export_dir}")
            print(f"  exported_audio_files: {exported}")
            print(f"  skipped_missing_audio: {skipped_export_no_audio}")
        print(f"  wrote_summary: {args.summary_out}")


if __name__ == "__main__":
    main()
