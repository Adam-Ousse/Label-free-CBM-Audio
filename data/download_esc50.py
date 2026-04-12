#!/usr/bin/env python3
"""Download or validate ESC-50.

The official ESC-50 distribution is published as a GitHub archive zip. This script
can download and extract it into a local directory, or simply validate a manually
provided dataset tree.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from download_utils import ensure_directory, is_nonempty_file, safe_remove, stream_download

ESC50_SOURCE_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


def find_esc50_root(base_dir: Path) -> Path | None:
    """Find a directory that contains the expected ESC-50 layout."""
    if (base_dir / "meta" / "esc50.csv").exists() and (base_dir / "audio").exists():
        return base_dir

    for child in base_dir.iterdir() if base_dir.exists() else []:
        if child.is_dir() and (child / "meta" / "esc50.csv").exists() and (child / "audio").exists():
            return child
    return None


def validate_esc50_root(root: Path) -> tuple[bool, str]:
    meta_csv = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    if not meta_csv.exists():
        return False, f"missing metadata file: {meta_csv}"
    if not audio_dir.exists():
        return False, f"missing audio directory: {audio_dir}"
    if not any(audio_dir.glob("*.wav")):
        return False, f"no WAV files found under: {audio_dir}"
    if not is_nonempty_file(meta_csv):
        return False, f"empty metadata file: {meta_csv}"
    return True, f"validated ESC-50 root at {root}"


def download_and_extract(url: str, output_dir: Path, force: bool = False) -> Path:
    ensure_directory(output_dir)

    existing_root = find_esc50_root(output_dir)
    if existing_root is not None:
        valid, reason = validate_esc50_root(existing_root)
        if valid and not force:
            print(f"ESC-50 already present: {existing_root}")
            print(reason)
            return existing_root
        if force:
            safe_remove(existing_root)

    archive_path = output_dir / "esc50_master.zip"
    print(f"Downloading ESC-50 from {url}")
    stream_download(url, archive_path)

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(output_dir)

    if archive_path.exists():
        archive_path.unlink()

    extracted_root = find_esc50_root(output_dir)
    if extracted_root is None:
        raise RuntimeError(
            f"ESC-50 download finished but no valid dataset root was found under {output_dir}"
        )

    valid, reason = validate_esc50_root(extracted_root)
    if not valid:
        raise RuntimeError(f"ESC-50 extraction validation failed: {reason}")

    print(f"ESC-50 downloaded and extracted to {extracted_root}")
    return extracted_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or validate the ESC-50 dataset")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the ESC-50 archive should be extracted or validated",
    )
    parser.add_argument(
        "--source_url",
        type=str,
        default=ESC50_SOURCE_URL,
        help="Download URL for the official ESC-50 archive zip",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite any existing extracted dataset")
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Exit early if a valid ESC-50 tree is already present",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate an existing ESC-50 directory tree; do not download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_esc50_root(args.output_dir)

    if args.validate_only:
        if root is None:
            raise FileNotFoundError(
                f"Could not find an ESC-50 directory tree under {args.output_dir}.\n"
                "Expected meta/esc50.csv and audio/*.wav."
            )
        valid, reason = validate_esc50_root(root)
        if not valid:
            raise RuntimeError(reason)
        print(reason)
        return

    if args.skip_if_exists and root is not None:
        valid, reason = validate_esc50_root(root)
        if valid and not args.force:
            print(f"ESC-50 already exists at {root}")
            print(reason)
            return

    extracted_root = download_and_extract(args.source_url, args.output_dir, force=args.force)
    valid, reason = validate_esc50_root(extracted_root)
    print(reason)


if __name__ == "__main__":
    main()
