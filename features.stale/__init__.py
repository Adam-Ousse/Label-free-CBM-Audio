"""CPU audio feature extraction utilities."""

from .audio_descriptors import (
    DESCRIPTOR_NAMES,
    compute_librosa_descriptors,
    compute_opensmile_egemaps,
    extract_audio_descriptors,
)

__all__ = [
    "DESCRIPTOR_NAMES",
    "compute_librosa_descriptors",
    "compute_opensmile_egemaps",
    "extract_audio_descriptors",
]
