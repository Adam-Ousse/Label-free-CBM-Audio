#!/usr/bin/env python3
"""Shared helpers for audio dataset download scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Iterable, Optional


AUDIOSET_DEFAULT_SAMPLE_RATE = 16000


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_command(command_or_path: str, friendly_name: str) -> str:
    """Return an executable path or raise a clear error if it cannot be found."""
    candidate = Path(command_or_path)
    if candidate.exists():
        return str(candidate)

    resolved = shutil.which(command_or_path)
    if resolved:
        return resolved

    raise FileNotFoundError(
        f"Could not find {friendly_name} ({command_or_path}). Install it or pass an explicit path."
    )


def run_command(args: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture output for debugging."""
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def stream_download(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a remote file to destination using urllib without extra dependencies."""
    from urllib.request import urlopen

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output.write(chunk)


def validate_wav_file(path: Path) -> bool:
    """Return True if the file exists and can be opened as a WAV."""
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            wav_file.getframerate()
            wav_file.getnframes()
    except Exception:
        return False
    return True


def read_wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
    duration = float(n_frames) / float(sample_rate) if sample_rate > 0 else 0.0
    return sample_rate, duration


def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def audioset_clip_stem(youtube_id: str, start_sec: float, end_sec: float) -> str:
    """Deterministic AudioSet clip filename stem using millisecond timestamps."""
    start_ms = int(round(float(start_sec) * 1000.0))
    end_ms = int(round(float(end_sec) * 1000.0))
    return f"{youtube_id}_{start_ms}_{end_ms}"


def audioset_clip_filename(youtube_id: str, start_sec: float, end_sec: float, audio_format: str = "wav") -> str:
    return f"{audioset_clip_stem(youtube_id, start_sec, end_sec)}.{audio_format}"


def safe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None
