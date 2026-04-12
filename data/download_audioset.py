#!/usr/bin/env python3
"""Download and segment AudioSet clips locally using yt-dlp and ffmpeg.

The script reads official AudioSet CSV metadata and reconstructs local waveform clips.
It is best-effort: unavailable YouTube videos are skipped or reported as failures.
"""

from __future__ import annotations

import argparse
import csv
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from download_utils import (
    AUDIOSET_DEFAULT_SAMPLE_RATE,
    audioset_clip_filename,
    audioset_clip_stem,
    ensure_directory,
    read_wav_info,
    require_command,
    run_command,
    safe_remove,
    validate_wav_file,
)

LOGGER = logging.getLogger("download_audioset")
MAX_ITEMS_CAP = 2000


@dataclass(frozen=True)
class AudioSetRow:
    youtube_id: str
    start_sec: float
    end_sec: float
    labels_mid: List[str]


def parse_segment_csv(csv_path: Path) -> List[AudioSetRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"AudioSet CSV not found: {csv_path}")

    rows: List[AudioSetRow] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0].strip() == "YTID":
                continue
            if len(row) < 4:
                continue
            try:
                youtube_id = row[0].strip()
                start_sec = float(row[1].strip())
                end_sec = float(row[2].strip())
                labels_mid = [token.strip() for token in row[3].strip().strip('"').split(",") if token.strip()]
            except ValueError:
                continue
            rows.append(AudioSetRow(youtube_id, start_sec, end_sec, labels_mid))
    return rows


def infer_split_name(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if "balanced" in stem:
        return "balanced_train"
    if "eval" in stem:
        return "eval"
    return stem


def _download_source_audio(
    youtube_id: str,
    temp_dir: Path,
    yt_dlp_path: str,
    retries: int,
) -> Path:
    output_template = str(temp_dir / f"{youtube_id}.%(ext)s")
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    args = [
        yt_dlp_path,
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--retries",
        str(retries),
        "--fragment-retries",
        str(retries),
        "-o",
        output_template,
        url,
    ]
    result = run_command(args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "yt-dlp download failed")

    candidates = list(temp_dir.glob(f"{youtube_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp completed but no source file was found for {youtube_id}")
    return candidates[0]


def _extract_segment(
    source_audio: Path,
    destination: Path,
    ffmpeg_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int,
    audio_format: str,
) -> None:
    ensure_directory(destination.parent)
    duration = max(0.0, float(end_sec) - float(start_sec))
    if duration <= 0:
        raise ValueError(f"Invalid segment duration for {destination.name}: {start_sec} to {end_sec}")

    if audio_format != "wav":
        raise ValueError(f"Unsupported audio_format={audio_format}. This downloader currently writes WAV clips only.")

    temp_out = destination.with_suffix(".tmp.wav")
    args = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_audio),
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(temp_out),
    ]
    result = run_command(args)
    if result.returncode != 0:
        safe_remove(temp_out)
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "ffmpeg extraction failed")

    temp_out.replace(destination)


def download_one(
    row: AudioSetRow,
    output_dir: Path,
    yt_dlp_path: str,
    ffmpeg_path: str,
    audio_format: str,
    sample_rate: int,
    skip_existing: bool,
    force_redownload: bool,
    keep_temp: bool,
    retries: int,
) -> Dict:
    clip_name = audioset_clip_filename(row.youtube_id, row.start_sec, row.end_sec, audio_format=audio_format)
    final_path = output_dir / clip_name

    if final_path.exists() and validate_wav_file(final_path) and not force_redownload:
        return {"status": "skipped", "path": str(final_path), "youtube_id": row.youtube_id}

    if final_path.exists() and force_redownload:
        safe_remove(final_path)

    temp_root = Path(tempfile.mkdtemp(prefix="audioset_dl_", dir=str(output_dir)))
    try:
        source_audio = _download_source_audio(row.youtube_id, temp_root, yt_dlp_path, retries=retries)
        _extract_segment(
            source_audio=source_audio,
            destination=final_path,
            ffmpeg_path=ffmpeg_path,
            start_sec=row.start_sec,
            end_sec=row.end_sec,
            sample_rate=sample_rate,
            audio_format=audio_format,
        )
        if not validate_wav_file(final_path):
            raise RuntimeError(f"Extracted clip is not a valid WAV file: {final_path}")
        return {"status": "downloaded", "path": str(final_path), "youtube_id": row.youtube_id}
    finally:
        if keep_temp:
            LOGGER.info("Keeping temp directory: %s", temp_root)
        else:
            safe_remove(temp_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AudioSet clips from local CSV metadata")
    parser.add_argument("--csv", type=Path, required=True, help="AudioSet segment CSV (balanced_train or eval)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store reconstructed clips")
    parser.add_argument("--audio_format", type=str, default="wav", help="Output audio format (wav only supported)")
    parser.add_argument("--jobs", type=int, default=4, help="Number of parallel download workers")
    parser.add_argument(
        "--max_items",
        type=int,
        default=MAX_ITEMS_CAP,
        help="Limit how many rows to process (hard-capped at 2000)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip clips that already exist and validate")
    parser.add_argument("--force_redownload", action="store_true", help="Overwrite existing clips")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately on the first failure")
    parser.add_argument("--keep_temp", action="store_true", help="Keep intermediate temp download files")
    parser.add_argument("--yt_dlp_path", type=str, default="yt-dlp", help="yt-dlp executable path")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="ffmpeg executable path")
    parser.add_argument("--sample_rate", type=int, default=AUDIOSET_DEFAULT_SAMPLE_RATE, help="Output sample rate")
    parser.add_argument("--retries", type=int, default=3, help="Retries for yt-dlp downloads")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    yt_dlp_path = require_command(args.yt_dlp_path, "yt-dlp")
    ffmpeg_path = require_command(args.ffmpeg_path, "ffmpeg")

    rows = parse_segment_csv(args.csv)
    requested_max_items = args.max_items if args.max_items is not None else MAX_ITEMS_CAP
    if requested_max_items > MAX_ITEMS_CAP:
        LOGGER.warning(
            "Requested --max_items=%s exceeds hard cap; using %s",
            requested_max_items,
            MAX_ITEMS_CAP,
        )
    effective_max_items = min(requested_max_items, MAX_ITEMS_CAP)
    rows = rows[:effective_max_items]

    split_name = infer_split_name(args.csv)
    ensure_directory(args.output_dir)

    LOGGER.info("Processing %s rows from %s", len(rows), split_name)
    LOGGER.info("Output directory: %s", args.output_dir)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []

    if args.jobs <= 1:
        iterator = enumerate(rows, start=1)
        for index, row in iterator:
            try:
                result = download_one(
                    row=row,
                    output_dir=args.output_dir,
                    yt_dlp_path=yt_dlp_path,
                    ffmpeg_path=ffmpeg_path,
                    audio_format=args.audio_format,
                    sample_rate=args.sample_rate,
                    skip_existing=args.skip_existing,
                    force_redownload=args.force_redownload,
                    keep_temp=args.keep_temp,
                    retries=args.retries,
                )
                counts[result["status"]] += 1
                LOGGER.info("[%s/%s] %s %s", index, len(rows), result["status"], result["path"])
            except Exception as exc:
                counts["failed"] += 1
                message = f"{row.youtube_id} {row.start_sec}-{row.end_sec}: {exc}"
                failures.append(message)
                LOGGER.warning(message)
                if args.fail_fast:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    download_one,
                    row,
                    args.output_dir,
                    yt_dlp_path,
                    ffmpeg_path,
                    args.audio_format,
                    args.sample_rate,
                    args.skip_existing,
                    args.force_redownload,
                    args.keep_temp,
                    args.retries,
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                    counts[result["status"]] += 1
                    LOGGER.info("%s %s", result["status"], result["path"])
                except Exception as exc:
                    counts["failed"] += 1
                    message = f"{row.youtube_id} {row.start_sec}-{row.end_sec}: {exc}"
                    failures.append(message)
                    LOGGER.warning(message)
                    if args.fail_fast:
                        for pending in futures:
                            pending.cancel()
                        raise

    print("AudioSet download summary")
    print(f"  split: {split_name}")
    print(f"  processed: {len(rows)}")
    print(f"  downloaded: {counts['downloaded']}")
    print(f"  skipped_existing: {counts['skipped']}")
    print(f"  failures: {counts['failed']}")
    if failures:
        print("  sample failures:")
        for failure in failures[:10]:
            print(f"    - {failure}")


if __name__ == "__main__":
    main()
