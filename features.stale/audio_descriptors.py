"""Deterministic CPU audio descriptors used by the revision preparation.

The scalar descriptors are aggregated over librosa frames.  ``harmonic_ratio``
is the fraction of waveform energy retained by ``librosa.effects.harmonic``;
this definition is recorded in the cache metadata so it is not confused with
a voicing probability or a harmonic-to-noise ratio from a different package.

OpenSMILE is optional at import time.  ``compute_opensmile_egemaps`` raises a
clear dependency error if it is requested without the ``opensmile`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

DESCRIPTOR_NAMES = (
    "rms_db",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "spectral_rolloff_hz",
    "zcr",
    "onset_mean",
    "onset_peak",
    "f0_median_hz",
    "f0_range_hz",
    "harmonic_ratio",
)

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512


def _librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised in dependency smoke tests
        raise RuntimeError("Task 5 requires librosa; install it with `uv pip install librosa`.") from exc
    return librosa


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("audio must be a one- or two-dimensional array")
    if array.ndim == 2:
        # Accept either (channels, samples) or (samples, channels).
        axis = 1 if array.shape[0] <= array.shape[1] else 0
        array = np.mean(array, axis=axis)
    if array.ndim != 1:
        raise ValueError("audio must be a one- or two-dimensional array")
    if array.size == 0:
        raise ValueError("audio must contain at least one sample")
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _f0_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.max(values) - np.min(values)) if values.size else float("nan")


def compute_librosa_descriptors(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    fmin: float = 65.0,
    fmax: float = 2093.0,
) -> dict[str, float]:
    """Compute the requested scalar librosa descriptors for one audio segment."""

    if int(sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    y = _mono_float32(audio)
    librosa = _librosa()

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    rms_db = float(np.mean(librosa.amplitude_to_db(rms, ref=1.0)))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sample_rate, hop_length=hop_length)

    try:
        f0, _, voiced_prob = librosa.pyin(
            y,
            fmin=fmin,
            fmax=min(fmax, (float(sample_rate) / 2.0) - 1.0),
            sr=sample_rate,
            frame_length=n_fft,
            hop_length=hop_length,
        )
    except (ValueError, RuntimeError):
        f0 = np.full(1, np.nan, dtype=np.float32)

    harmonic = librosa.effects.harmonic(y)
    total_energy = float(np.sum(np.square(y), dtype=np.float64))
    harmonic_ratio = (
        float(np.sum(np.square(harmonic), dtype=np.float64) / total_energy)
        if total_energy > 0.0
        else 0.0
    )

    # Keep pyin's voiced probability in the local scope intentionally: it is
    # useful for debugging the all-unvoiced case, but not part of the requested
    # descriptor set.  This avoids silently treating unvoiced frames as zero Hz.
    del voiced_prob
    return {
        "rms_db": rms_db,
        "spectral_centroid_hz": float(np.mean(centroid)),
        "spectral_bandwidth_hz": float(np.mean(bandwidth)),
        "spectral_flatness": float(np.mean(flatness)),
        "spectral_rolloff_hz": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
        "onset_mean": float(np.mean(onset)) if onset.size else 0.0,
        "onset_peak": float(np.max(onset)) if onset.size else 0.0,
        "f0_median_hz": _finite_median(f0),
        "f0_range_hz": _f0_range(f0),
        "harmonic_ratio": float(np.clip(harmonic_ratio, 0.0, 1.0)),
    }


def compute_opensmile_egemaps(
    audio: np.ndarray,
    sample_rate: int,
    *,
    feature_set: str = "eGeMAPSv02",
) -> dict[str, float]:
    """Compute OpenSMILE eGeMAPS functionals for one audio segment."""

    try:
        import opensmile
    except ImportError as exc:  # pragma: no cover - dependency-gated
        raise RuntimeError(
            "CREMA-D eGeMAPS requires opensmile; install it with `uv pip install opensmile`."
        ) from exc

    try:
        feature_set_value = getattr(opensmile.FeatureSet, feature_set)
    except AttributeError as exc:
        raise ValueError(f"Unsupported OpenSMILE feature set: {feature_set}") from exc

    smile = opensmile.Smile(
        feature_set=feature_set_value,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    frame = smile.process_signal(_mono_float32(audio), int(sample_rate))
    if frame.shape[0] != 1:
        raise ValueError(f"Expected one OpenSMILE functional row, got shape {frame.shape}")
    return {str(key): float(value) for key, value in frame.iloc[0].items()}


def load_audio(audio_path: str | Path, *, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono audio via soundfile, optionally resampling with librosa."""

    import soundfile as sf

    y, sr = sf.read(str(audio_path), always_2d=False, dtype="float32")
    y = _mono_float32(y)
    if sample_rate is not None and int(sr) != int(sample_rate):
        librosa = _librosa()
        y = librosa.resample(y, orig_sr=int(sr), target_sr=int(sample_rate))
        sr = int(sample_rate)
    return y, int(sr)


def extract_audio_descriptors(
    audio_path: str | Path,
    *,
    include_egemaps: bool = False,
    sample_rate: int | None = None,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> dict[str, float]:
    """Load a file and return librosa descriptors, optionally plus eGeMAPS."""

    y, sr = load_audio(audio_path, sample_rate=sample_rate)
    values: dict[str, float] = compute_librosa_descriptors(
        y, sr, n_fft=n_fft, hop_length=hop_length
    )
    if include_egemaps:
        values.update(compute_opensmile_egemaps(y, sr))
    return values


def iter_fixed_windows(
    audio: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    pad_end: bool = True,
):
    """Yield ``(start_seconds, end_seconds, samples)`` fixed-size windows."""

    if window_seconds <= 0.0 or hop_seconds <= 0.0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    y = _mono_float32(audio)
    window_samples = max(1, int(round(window_seconds * int(sample_rate))))
    hop_samples = max(1, int(round(hop_seconds * int(sample_rate))))
    if pad_end:
        starts = range(0, max(len(y), 1), hop_samples)
    else:
        starts = range(0, max(len(y) - window_samples + 1, 0), hop_samples)
    for start in starts:
        stop = start + window_samples
        if start >= len(y):
            break
        segment = y[start:stop]
        if len(segment) < window_samples:
            if not pad_end:
                continue
            segment = np.pad(segment, (0, window_samples - len(segment)))
        yield start / float(sample_rate), stop / float(sample_rate), segment


def descriptor_vector(values: Mapping[str, Any], names: tuple[str, ...] = DESCRIPTOR_NAMES) -> np.ndarray:
    """Convert a descriptor mapping into a stable float32 vector."""

    return np.asarray([values.get(name, np.nan) for name in names], dtype=np.float32)
