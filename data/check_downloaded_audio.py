#!/usr/bin/env python3
"""Smoke-test downloaded audio directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_utils import read_wav_info, validate_wav_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a directory of downloaded WAV clips")
    parser.add_argument("--audio_dir", type=Path, required=True, help="Directory containing downloaded WAV files")
    parser.add_argument("--sample", type=int, default=5, help="Number of files to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {args.audio_dir}")

    files = sorted(args.audio_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found in {args.audio_dir}")

    print(f"Found {len(files)} WAV files in {args.audio_dir}")
    for path in files[: args.sample]:
        if not validate_wav_file(path):
            raise RuntimeError(f"Unreadable WAV file: {path}")
        sample_rate, duration = read_wav_info(path)
        print(f"{path.name}: sr={sample_rate}, duration={duration:.3f}s")

    print("Downloaded audio validation completed successfully.")


if __name__ == "__main__":
    main()
