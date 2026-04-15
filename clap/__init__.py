from .core import (
    DEFAULT_CLAP_MODEL,
    DEFAULT_CLAP_SAMPLE_RATE,
    _extract_embedding_tensor,
    compute_audio_text_similarity,
    encode_audio,
    encode_text,
    load_clap_model,
)

__all__ = [
    "DEFAULT_CLAP_MODEL",
    "DEFAULT_CLAP_SAMPLE_RATE",
    "_extract_embedding_tensor",
    "compute_audio_text_similarity",
    "encode_audio",
    "encode_text",
    "load_clap_model",
]
