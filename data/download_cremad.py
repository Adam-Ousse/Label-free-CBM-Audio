#!/usr/bin/env python3
"""Download or validate CREMA-D from Hugging Face datasets hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from download_utils import ensure_directory, is_nonempty_file, safe_remove

CREMAD_REPO_ID = "MahiA/CREMA-D"


def _resolve_layout(root: Path) -> tuple[Path, Path, Path]:
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    audio_dir = root / "audios"
    if train_csv.exists() and test_csv.exists() and audio_dir.exists():
        return train_csv, test_csv, audio_dir
    raise FileNotFoundError("CREMA-D layout not found under {}".format(root))


def find_cremad_root(base_dir: Path) -> Path | None:
    candidates = [
        base_dir,
        base_dir / "CREMA-D",
    ]

    for child in base_dir.iterdir() if base_dir.exists() else []:
        if child.is_dir():
            candidates.append(child)
            candidates.append(child / "CREMA-D")

    for candidate in candidates:
        try:
            _resolve_layout(candidate)
            return candidate
        except FileNotFoundError:
            continue

    return None


def validate_cremad_root(root: Path) -> tuple[bool, str]:
    try:
        train_csv, test_csv, audio_dir = _resolve_layout(root)
    except FileNotFoundError:
        return False, "missing expected CREMA-D layout under {} (train.csv, test.csv, audios/)".format(root)

    if not is_nonempty_file(train_csv):
        return False, "empty or missing train split file: {}".format(train_csv)
    if not is_nonempty_file(test_csv):
        return False, "empty or missing test split file: {}".format(test_csv)

    wav_count = len(list(audio_dir.glob("*.wav"))) + len(list(audio_dir.glob("*.WAV")))
    if wav_count == 0:
        return False, "no wav files found under {}".format(audio_dir)

    return True, "validated CREMA-D root at {} (wav files: {})".format(root, wav_count)


def download_snapshot(repo_id: str, output_dir: Path, revision: str | None, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for CREMA-D download. Install with: pip install huggingface_hub"
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_dir),
        revision=revision,
        token=token,
    )


def download_and_validate(
    repo_id: str,
    output_dir: Path,
    force: bool = False,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    ensure_directory(output_dir)

    existing_root = find_cremad_root(output_dir)
    if existing_root is not None:
        valid, reason = validate_cremad_root(existing_root)
        if valid and not force:
            print("CREMA-D already present:", existing_root)
            print(reason)
            return existing_root
        if force:
            safe_remove(existing_root)
            ensure_directory(output_dir)

    print("Downloading CREMA-D dataset from", repo_id)
    download_snapshot(repo_id=repo_id, output_dir=output_dir, revision=revision, token=token)

    extracted_root = find_cremad_root(output_dir)
    if extracted_root is None:
        raise RuntimeError("CREMA-D downloaded but expected layout was not found under {}".format(output_dir))

    valid, reason = validate_cremad_root(extracted_root)
    if not valid:
        raise RuntimeError("CREMA-D validation failed: {}".format(reason))

    print(reason)
    return extracted_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or validate CREMA-D dataset")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/cremad/raw"),
        help="Directory where CREMA-D should be downloaded or validated",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=CREMAD_REPO_ID,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional repo revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token",
    )
    parser.add_argument("--force", action="store_true", help="Redownload and overwrite existing extracted dataset")
    parser.add_argument("--skip_if_exists", action="store_true", help="Exit early if a valid dataset tree exists")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing dataset tree")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_cremad_root(args.output_dir)

    if args.validate_only:
        if root is None:
            raise FileNotFoundError(
                "Could not find CREMA-D under {}. Expected train.csv, test.csv, and audios/.".format(args.output_dir)
            )
        valid, reason = validate_cremad_root(root)
        if not valid:
            raise RuntimeError(reason)
        print(reason)
        return

    if args.skip_if_exists and root is not None:
        valid, reason = validate_cremad_root(root)
        if valid and not args.force:
            print("CREMA-D already exists at", root)
            print(reason)
            return

    download_and_validate(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        force=args.force,
        revision=args.revision,
        token=args.token,
    )


if __name__ == "__main__":
    main()
