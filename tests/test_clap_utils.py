import pytest
import torch

from clap import core as clap_utils


@pytest.fixture(scope="module")
def real_clap_bundle():
    # real model load from hf
    return clap_utils.load_clap_model(device="cpu")


def test_extract_embedding_tensor_from_tuple():
    tensor = torch.randn(2, 3)
    out = (tensor,)
    result = clap_utils._extract_embedding_tensor(out, preferred_keys=["text_embeds"])
    assert result is tensor


def test_clap_real_text_and_audio_embeddings(real_clap_bundle):
    text_embs = clap_utils.encode_text(["dog bark", "rain sound"], clap_bundle=real_clap_bundle, batch_size=2)
    audio = torch.randn(2, 1, clap_utils.DEFAULT_CLAP_SAMPLE_RATE)
    audio_embs = clap_utils.encode_audio(
        audio_or_paths=audio,
        clap_bundle=real_clap_bundle,
        sample_rates=[clap_utils.DEFAULT_CLAP_SAMPLE_RATE, clap_utils.DEFAULT_CLAP_SAMPLE_RATE],
        batch_size=2,
        target_sample_rate=clap_utils.DEFAULT_CLAP_SAMPLE_RATE,
    )
    sims = clap_utils.compute_audio_text_similarity(audio_embs, text_embs)

    assert text_embs.dim() == 2
    assert audio_embs.dim() == 2
    assert text_embs.shape[0] == 2
    assert audio_embs.shape[0] == 2
    assert text_embs.shape[1] == audio_embs.shape[1]
    assert sims.shape == (2, 2)


def test_encode_audio_requires_sample_rates_for_tensors():
    real_clap_bundle = clap_utils.load_clap_model(device="cpu")

    with pytest.raises(ValueError, match="sample_rates is required"):
        clap_utils.encode_audio(audio_or_paths=torch.randn(1, 16000), clap_bundle=real_clap_bundle)
