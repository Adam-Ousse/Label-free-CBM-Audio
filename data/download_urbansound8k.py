#!/usr/bin/env python3
"""Download or validate UrbanSound8K using curl and a Kaggle dataset URL."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from download_utils import ensure_directory, first_existing, is_nonempty_file, require_command, run_command, safe_remove

URBANSOUND8K_SOURCE_URL = "https://www.kaggle.com/api/v1/datasets/download/chrisfilo/urbansound8k"


def _has_fold_dirs(base_dir: Path) -> bool:
    for fold in range(1, 11):
        if not (base_dir / "fold{}".format(fold)).exists():
            return False
    return True


def _resolve_layout(root: Path) -> tuple[Path, Path]:
    # layout a: UrbanSound8K/metadata/UrbanSound8K.csv + UrbanSound8K/audio/fold1..fold10
    meta_nested = root / "metadata" / "UrbanSound8K.csv"
    audio_nested = root / "audio"
    if meta_nested.exists() and audio_nested.exists() and _has_fold_dirs(audio_nested):
        return meta_nested, audio_nested

    # layout b (kaggle flat): UrbanSound8K.csv + fold1..fold10 directly under root
    meta_flat = root / "UrbanSound8K.csv"
    if meta_flat.exists() and _has_fold_dirs(root):
        return meta_flat, root

    raise FileNotFoundError("UrbanSound8K layout not found under {}".format(root))


def find_urbansound8k_root(base_dir: Path) -> Path | None:
    candidates = [
        base_dir,
        base_dir / "UrbanSound8K",
    ]

    for child in base_dir.iterdir() if base_dir.exists() else []:
        if child.is_dir():
            candidates.append(child)
            candidates.append(child / "UrbanSound8K")

    for candidate in candidates:
        try:
            _resolve_layout(candidate)
            return candidate
        except FileNotFoundError:
            continue

    return None


def validate_urbansound8k_root(root: Path) -> tuple[bool, str]:
    try:
        meta_csv, audio_dir = _resolve_layout(root)
    except FileNotFoundError:
        return (
            False,
            "missing expected UrbanSound8K layout under {} (either metadata/UrbanSound8K.csv + audio/fold* or UrbanSound8K.csv + fold*)".format(
                root
            ),
        )

    if not is_nonempty_file(meta_csv):
        return False, "empty metadata file: {}".format(meta_csv)

    missing_folds = []
    wav_count = 0
    for fold in range(1, 11):
        fold_dir = audio_dir / "fold{}".format(fold)
        if not fold_dir.exists():
            missing_folds.append(str(fold_dir))
            continue
        wav_count += len(list(fold_dir.glob("*.wav")))

    if missing_folds:
        return False, "missing fold directories: {}".format(", ".join(missing_folds))
    if wav_count == 0:
        return False, "no wav files found under: {}".format(audio_dir)

    return True, "validated UrbanSound8K root at {} (wav files: {})".format(root, wav_count)


def download_archive(url: str, archive_path: Path, curl_bin: str) -> None:
    ensure_directory(archive_path.parent)
    if archive_path.exists():
        archive_path.unlink()

    curl_exec = require_command(curl_bin, "curl")
    cmd = [curl_exec, "-L", "-o", str(archive_path), url]
    result = run_command(cmd)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        raise RuntimeError(
            "UrbanSound8K download failed (exit {}).\nstdout:\n{}\nstderr:\n{}\n"
            "Tip: make sure Kaggle credentials are configured (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).".format(
                result.returncode,
                stdout,
                stderr,
            )
        )

    if not archive_path.exists() or archive_path.stat().st_size == 0:
        raise RuntimeError("UrbanSound8K archive download produced an empty file: {}".format(archive_path))


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(output_dir)


def download_and_extract(url: str, output_dir: Path, curl_bin: str, force: bool = False) -> Path:
    ensure_directory(output_dir)

    existing_root = find_urbansound8k_root(output_dir)
    if existing_root is not None:
        valid, reason = validate_urbansound8k_root(existing_root)
        if valid and not force:
            print("UrbanSound8K already present:", existing_root)
            print(reason)
            return existing_root
        if force:
            safe_remove(existing_root)

    archive_candidates = [
        output_dir / "urbansound8k.zip",
        output_dir / "UrbanSound8K.zip",
    ]
    archive_path = first_existing(archive_candidates) or archive_candidates[0]

    print("Downloading UrbanSound8K from", url)
    download_archive(url, archive_path, curl_bin=curl_bin)
    print("Extracting archive to", output_dir)
    extract_archive(archive_path, output_dir)
    if archive_path.exists():
        archive_path.unlink()

    extracted_root = find_urbansound8k_root(output_dir)
    if extracted_root is None:
        raise RuntimeError("UrbanSound8K extracted but expected layout was not found under {}".format(output_dir))

    valid, reason = validate_urbansound8k_root(extracted_root)
    if not valid:
        raise RuntimeError("UrbanSound8K extraction validation failed: {}".format(reason))

    print(reason)
    return extracted_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or validate UrbanSound8K dataset")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/urbansound8k/raw"),
        help="Directory where UrbanSound8K archive should be extracted or validated",
    )
    parser.add_argument(
        "--source_url",
        type=str,
        default=URBANSOUND8K_SOURCE_URL,
        help="Kaggle dataset download URL",
    )
    parser.add_argument(
        "--curl_bin",
        type=str,
        default="curl",
        help="curl executable or full path",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing extracted dataset")
    parser.add_argument("--skip_if_exists", action="store_true", help="Exit early if a valid dataset tree exists")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing dataset tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_urbansound8k_root(args.output_dir)

    if args.validate_only:
        if root is None:
            raise FileNotFoundError(
                "Could not find UrbanSound8K under {}. Expected metadata/UrbanSound8K.csv and audio/fold1..fold10".format(
                    args.output_dir
                )
            )
        valid, reason = validate_urbansound8k_root(root)
        if not valid:
            raise RuntimeError(reason)
        print(reason)
        return

    if args.skip_if_exists and root is not None:
        valid, reason = validate_urbansound8k_root(root)
        if valid and not args.force:
            print("UrbanSound8K already exists at", root)
            print(reason)
            return

    download_and_extract(args.source_url, args.output_dir, curl_bin=args.curl_bin, force=args.force)


if __name__ == "__main__":
    main()
