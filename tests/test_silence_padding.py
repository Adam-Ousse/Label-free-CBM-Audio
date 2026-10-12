import math

import torch

from scripts.evaluation.silence_padding import (
    build_window_masks,
    energy_weights,
    ensure_highest_rms_fallback,
    masked_pool,
    replace_masked_with_baseline,
    segment_rms_dbfs,
    window_valid_fractions,
    window_valid_sample_counts,
)


def test_window_validity_uses_true_duration():
    durations = torch.tensor([1.25, 0.0])
    starts = torch.tensor([0.0, 0.5, 1.0])
    counts = window_valid_sample_counts(durations, starts, sample_rate=10, window_sec=1.0)
    fractions = window_valid_fractions(durations, starts, window_sec=1.0)
    assert counts.tolist() == [[10, 8, 2], [0, 0, 0]]
    assert torch.allclose(fractions, torch.tensor([[1.0, 0.75, 0.25], [0.0, 0.0, 0.0]]))


def test_segment_rms_ignores_padded_samples_and_reports_silence():
    segments = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    rms_dbfs = segment_rms_dbfs(segments, torch.tensor([[2, 0]]))
    assert torch.allclose(rms_dbfs[0, 0], torch.tensor(0.0))
    assert math.isinf(float(rms_dbfs[0, 1])) and rms_dbfs[0, 1] < 0


def test_build_window_masks_separates_padding_and_silence():
    masks = build_window_masks(
        torch.tensor([[1.0, 0.5, 0.0]]),
        torch.tensor([[-40.0, -70.0, -math.inf]]),
        silence_threshold_dbfs=-60.0,
    )
    assert masks["padding_mask"].tolist() == [[True, True, False]]
    assert masks["silence_mask"].tolist() == [[False, True, True]]
    assert masks["content_mask"].tolist() == [[True, False, False]]


def test_highest_rms_fallback_selects_one_window_only_when_needed():
    mask = torch.tensor([[False, False], [True, False]])
    effective, fallback = ensure_highest_rms_fallback(mask, torch.tensor([[-80.0, -40.0], [-30.0, -20.0]]))
    assert fallback.tolist() == [True, False]
    assert effective.tolist() == [[False, True], [True, False]]


def test_energy_weights_are_duration_corrected_and_normalized():
    weights = energy_weights(
        torch.tensor([[-10.0, -20.0]]),
        torch.tensor([[1.0, 0.5]]),
        torch.tensor([[True, True]]),
    )
    expected = torch.tensor([[0.1, 0.5 * 0.01]])
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(weights, expected)


def test_masked_pool_supports_mean_max_lme_and_energy_weighted():
    values = torch.tensor([[[1.0], [3.0], [9.0]]])
    mask = torch.tensor([[True, False, True]])
    rms = torch.tensor([[-10.0, -20.0, -30.0]])
    mean, _, fallback = masked_pool(values, "mean", mask, rms)
    maximum, _, _ = masked_pool(values, "max", mask, rms)
    lme, _, _ = masked_pool(values, "lme", mask, rms, alpha=5.0)
    weighted, _, _ = masked_pool(
        values,
        "energy_weighted",
        mask,
        rms,
        weights=energy_weights(rms, torch.ones_like(rms), mask),
    )
    assert mean.item() == 5.0
    assert maximum.item() == 9.0
    assert lme.item() > 8.8
    assert weighted.item() < 5.0
    assert not fallback.any()

    batched, _, _ = masked_pool(
        values.repeat(2, 1, 1),
        "energy_weighted",
        mask.repeat(2, 1),
        rms.repeat(2, 1),
        weights=energy_weights(rms.repeat(2, 1), torch.ones((2, 3)), mask.repeat(2, 1)),
    )
    assert batched.shape == (2, 1)


def test_masked_pool_fallback_prevents_nan_for_all_masked_rows():
    values = torch.tensor([[[1.0], [4.0]]])
    pooled, effective, fallback = masked_pool(
        values,
        "mean",
        torch.zeros((1, 2), dtype=torch.bool),
        torch.tensor([[-50.0, -40.0]]),
    )
    assert pooled.item() == 4.0
    assert effective.tolist() == [[False, True]]
    assert fallback.tolist() == [True]


def test_replace_masked_with_baseline_preserves_fallback_rows():
    values = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    replaced, pool_mask = replace_masked_with_baseline(
        values,
        torch.tensor([0.0]),
        torch.tensor([[True, False], [False, False]]),
        torch.tensor([[True, False], [False, True]]),
        torch.tensor([False, True]),
    )
    assert replaced.squeeze(-1).tolist() == [[1.0, 0.0], [3.0, 4.0]]
    assert pool_mask.tolist() == [[True, True], [False, True]]
