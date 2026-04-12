import pytest
import torch

from models import ast_backbone


@pytest.fixture(scope="module")
def real_ast_backbone():
    # real model load from hf
    return ast_backbone.build_ast_backbone("ast_audioset", "cpu")


def test_resolve_model_id_alias_and_custom():
    assert ast_backbone._resolve_model_id("ast_audioset") == "MIT/ast-finetuned-audioset-10-10-0.4593"
    assert ast_backbone._resolve_model_id("ast_hf__foo__bar") == "foo/bar"


def test_ast_real_forward_output_shape(real_ast_backbone):
    audio = torch.randn(2, 1, 16000)
    out = real_ast_backbone(audio, sample_rates=torch.tensor([16000, 16000]))

    assert out.shape[0] == 2
    assert out.shape[1] == real_ast_backbone.hidden_size
    assert out.dtype == torch.float32


def test_ast_forward_raises_on_mixed_sample_rates(real_ast_backbone):
    audio = torch.randn(2, 16000)

    with pytest.raises(ValueError, match="Mixed sample rates"):
        real_ast_backbone(audio, sample_rates=[16000, 22050])
