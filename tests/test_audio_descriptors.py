import numpy as np

from features.audio_descriptors import (
    DESCRIPTOR_NAMES,
    compute_librosa_descriptors,
    descriptor_vector,
    iter_fixed_windows,
)


def test_librosa_descriptors_have_requested_names_and_finite_core_values():
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.25 * np.sin(2.0 * np.pi * 440.0 * time)

    values = compute_librosa_descriptors(audio, sample_rate, n_fft=1024, hop_length=256)

    assert tuple(values) == DESCRIPTOR_NAMES
    assert np.isfinite(values["rms_db"])
    assert 400.0 < values["spectral_centroid_hz"] < 480.0
    assert values["harmonic_ratio"] == np.clip(values["harmonic_ratio"], 0.0, 1.0)
    assert 430.0 < values["f0_median_hz"] < 450.0
    assert values["f0_range_hz"] >= 0.0
    assert descriptor_vector(values).shape == (len(DESCRIPTOR_NAMES),)


def test_silent_audio_does_not_turn_unvoiced_f0_into_zero():
    values = compute_librosa_descriptors(np.zeros(16000, dtype=np.float32), 16000)

    assert np.isfinite(values["rms_db"])
    assert np.isnan(values["f0_median_hz"])
    assert np.isnan(values["f0_range_hz"])
    assert values["harmonic_ratio"] == 0.0


def test_fixed_windows_pads_final_window_and_preserves_times():
    windows = list(iter_fixed_windows(np.arange(10, dtype=np.float32), 10, window_seconds=1.0, hop_seconds=0.5))

    assert len(windows) == 2
    assert windows[0][0:2] == (0.0, 1.0)
    assert windows[1][0:2] == (0.5, 1.5)
    assert windows[1][2].shape == (10,)
