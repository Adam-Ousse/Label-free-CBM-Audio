"""Pure masking and pooling primitives for segmented audio inference."""

from __future__ import annotations

import math

import torch


def window_valid_sample_counts(
    durations_sec: torch.Tensor,
    starts_sec: torch.Tensor,
    sample_rate: int,
    window_sec: float,
) -> torch.Tensor:
    """Return the number of non-padding samples in each segmented window."""
    durations = durations_sec.float().unsqueeze(-1)
    starts = starts_sec.float()
    overlap_sec = (torch.minimum(durations, starts + float(window_sec)) - starts).clamp_min(0.0)
    return torch.round(overlap_sec * float(sample_rate)).long()


def window_valid_fractions(
    durations_sec: torch.Tensor,
    starts_sec: torch.Tensor,
    window_sec: float,
) -> torch.Tensor:
    """Return each window's fraction covered by the true clip duration."""
    durations = durations_sec.float().unsqueeze(-1)
    starts = starts_sec.float()
    overlap_sec = (torch.minimum(durations, starts + float(window_sec)) - starts).clamp_min(0.0)
    return (overlap_sec / float(window_sec)).clamp(0.0, 1.0)


def segment_rms_dbfs(segments: torch.Tensor, valid_sample_counts: torch.Tensor) -> torch.Tensor:
    """Compute RMS in dBFS using only true, non-padding samples.

    ``segments`` has shape ``[batch, windows, samples]``.  A zero-length
    window has RMS ``-inf`` so it is always treated as silence.
    """
    if segments.ndim != 3:
        raise ValueError("segments must have shape [batch, windows, samples]")
    if valid_sample_counts.shape != segments.shape[:2]:
        raise ValueError("valid_sample_counts must match the first two segment dimensions")
    sample_index = torch.arange(segments.shape[-1], device=segments.device)
    valid = sample_index.view(1, 1, -1) < valid_sample_counts.unsqueeze(-1)
    squared = segments.float().square().masked_fill(~valid, 0.0)
    counts = valid_sample_counts.clamp_min(1).float()
    rms = torch.sqrt(squared.sum(dim=-1) / counts)
    return torch.where(rms > 0.0, 20.0 * torch.log10(rms), torch.full_like(rms, -math.inf))


def build_window_masks(
    valid_fractions: torch.Tensor,
    rms_dbfs: torch.Tensor,
    silence_threshold_dbfs: float,
) -> dict[str, torch.Tensor]:
    """Build padding, silence, and combined content masks."""
    if valid_fractions.shape != rms_dbfs.shape:
        raise ValueError("valid_fractions and rms_dbfs must have the same shape")
    padding_mask = valid_fractions > 0.0
    silence_mask = rms_dbfs < float(silence_threshold_dbfs)
    content_mask = padding_mask & ~silence_mask
    return {
        "padding_mask": padding_mask,
        "silence_mask": silence_mask,
        "content_mask": content_mask,
    }


def ensure_highest_rms_fallback(
    mask: torch.Tensor,
    rms_dbfs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure every sample retains its highest-RMS window when fully masked."""
    if mask.shape != rms_dbfs.shape or mask.ndim != 2:
        raise ValueError("mask and rms_dbfs must have shape [batch, windows]")
    fallback_rows = ~mask.any(dim=1)
    if not fallback_rows.any():
        return mask, fallback_rows
    selected = torch.argmax(rms_dbfs, dim=1)
    effective = mask.clone()
    rows = torch.nonzero(fallback_rows, as_tuple=False).flatten()
    effective[rows, selected[rows]] = True
    return effective, fallback_rows


def energy_weights(
    rms_dbfs: torch.Tensor,
    valid_fractions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return normalized RMS-energy weights with true-duration correction."""
    if rms_dbfs.shape != valid_fractions.shape:
        raise ValueError("rms_dbfs and valid_fractions must have the same shape")
    weights = torch.where(torch.isfinite(rms_dbfs), 10.0 ** (rms_dbfs / 10.0), torch.zeros_like(rms_dbfs))
    weights = weights * valid_fractions.float().clamp_min(0.0)
    if mask is not None:
        weights = weights.masked_fill(~mask, 0.0)
    total = weights.sum(dim=1, keepdim=True)
    return weights / total.clamp_min(torch.finfo(weights.dtype).eps)


def masked_pool(
    values: torch.Tensor,
    pool: str,
    mask: torch.Tensor,
    rms_dbfs: torch.Tensor,
    alpha: float = 5.0,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool ``[batch, windows, features]`` with masking and fallback.

    Returns pooled values, the effective mask, and rows that used the
    highest-RMS fallback.  The LME denominator counts retained windows, which
    keeps its scale comparable after masking.
    """
    if values.ndim != 3:
        raise ValueError("values must have shape [batch, windows, features]")
    if mask.shape != values.shape[:2] or rms_dbfs.shape != mask.shape:
        raise ValueError("mask and rms_dbfs must match values' first two dimensions")
    effective_mask, fallback_rows = ensure_highest_rms_fallback(mask, rms_dbfs)
    selected = effective_mask.unsqueeze(-1)

    if pool == "mean":
        pooled = (values * selected).sum(dim=1) / effective_mask.sum(dim=1, keepdim=True).float()
    elif pool == "max":
        pooled = values.masked_fill(~selected, -torch.inf).max(dim=1).values
    elif pool == "lme":
        count = effective_mask.sum(dim=1).float().clamp_min(1.0)
        pooled = torch.logsumexp(values.masked_fill(~selected, -torch.inf) * float(alpha), dim=1)
        pooled = pooled / float(alpha) - torch.log(count).unsqueeze(-1) / float(alpha)
    elif pool == "energy_weighted":
        if weights is None:
            raise ValueError("energy_weighted pooling requires weights")
        weights = weights.masked_fill(~effective_mask, 0.0)
        rows_without_weight = weights.sum(dim=1) <= torch.finfo(weights.dtype).eps
        if rows_without_weight.any():
            fallback_index = torch.argmax(rms_dbfs, dim=1)
            rows = torch.nonzero(rows_without_weight, as_tuple=False).flatten()
            weights[rows] = 0.0
            weights[rows, fallback_index[rows]] = 1.0
        pooled = (values * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
    else:
        raise ValueError("unknown pool: {}".format(pool))
    return pooled, effective_mask, fallback_rows


def replace_masked_with_baseline(
    values: torch.Tensor,
    baseline: torch.Tensor,
    raw_mask: torch.Tensor,
    effective_mask: torch.Tensor,
    fallback_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace masked windows by a baseline, retaining fallback rows unchanged."""
    if values.ndim != 3 or raw_mask.shape != values.shape[:2]:
        raise ValueError("values and raw_mask have incompatible shapes")
    baseline = baseline.reshape(1, 1, -1).to(values)
    replaced = torch.where(raw_mask.unsqueeze(-1), values, baseline)
    all_masked = fallback_rows.unsqueeze(-1).unsqueeze(-1)
    replaced = torch.where(all_masked, values, replaced)
    pool_mask = torch.ones_like(raw_mask)
    if fallback_rows.any():
        pool_mask[fallback_rows] = effective_mask[fallback_rows]
    return replaced, pool_mask
