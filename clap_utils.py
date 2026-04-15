import warnings

from clap import (  # noqa: F401
    DEFAULT_CLAP_MODEL,
    DEFAULT_CLAP_SAMPLE_RATE,
    _extract_embedding_tensor,
    compute_audio_text_similarity,
    encode_audio,
    encode_text,
    load_clap_model,
)


warnings.warn(
    "clap_utils.py is deprecated. Import from the clap package instead.",
    DeprecationWarning,
    stacklevel=2,
)
