"""CPU labels, metrics, scorers, and plots for Task 6 URBAN-SED analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def load_event_window_rows(path: str | Path, *, hop_seconds: float) -> list[dict[str, Any]]:
    """Load prepared event windows and verify the requested fixed hop."""

    if hop_seconds not in (0.5, 0.25):
        raise ValueError("Task 6 supports only 0.5-second and 0.25-second hops")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"event-window file is empty: {path}")
    for row in rows:
        if abs(float(row["hop_seconds"]) - float(hop_seconds)) > 1e-8:
            raise ValueError("event-window file contains a different hop than requested")
        if abs(float(row["window_seconds"]) - 1.0) > 1e-8:
            raise ValueError("Task 6 event windows must be one second")
    return rows


def _event_interval(event: Mapping[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(event["event_start"])
        end = float(event["event_end"])
    except (KeyError, TypeError, ValueError):
        try:
            start = float(event["time"])
            end = start + float(event["duration"])
        except (KeyError, TypeError, ValueError):
            return None
    return (start, end) if end > start else None


def event_active_window_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    class_names: Sequence[str] | None = None,
    minimum_overlap_seconds: float = 0.0,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build binary event-active labels directly from onset/offset spans.

    The prepared rows contain the overlap-expanded event records.  This
    function recomputes activity from their event start/end values and does
    not use a test-tuned threshold.
    """

    if minimum_overlap_seconds < 0.0:
        raise ValueError("minimum_overlap_seconds must be non-negative")
    labels = sorted(
        {
            str(event["label"])
            for row in rows
            for event in row.get("events", [])
            if event.get("label") is not None and _event_interval(event) is not None
        }
    )
    if class_names is not None:
        labels = [str(value) for value in class_names]
    class_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(rows), len(labels)), dtype=np.int8)
    soundscape_ids: list[str] = []
    for row_index, row in enumerate(rows):
        soundscape_ids.append(str(row["soundscape_id"]))
        window_start = float(row["window_start"])
        window_end = float(row["window_end"])
        for event in row.get("events", []):
            label = event.get("label")
            interval = _event_interval(event)
            if label is None or interval is None or str(label) not in class_index:
                continue
            overlap = max(0.0, min(window_end, interval[1]) - max(window_start, interval[0]))
            if overlap > float(minimum_overlap_seconds):
                matrix[row_index, class_index[str(label)]] = 1
    return matrix, labels, soundscape_ids


def _validate_window_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 2 or labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError(
            "scores and labels must have equal shape [windows, classes], "
            f"got {scores.shape} and {labels.shape}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values")
    return scores, labels


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute AUROC via the Mann--Whitney statistic with tie averaging."""

    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal shape")
    positives = y == 1
    negatives = y == 0
    if not positives.any() or not negatives.any():
        return None
    ranks = _average_ranks(s)
    positive_rank_sum = float(ranks[positives].sum())
    n_positive = int(positives.sum())
    n_negative = int(negatives.sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute threshold-free average precision for one binary class."""

    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal shape")
    positive_count = int((y == 1).sum())
    if positive_count == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    cumulative = np.cumsum(sorted_y == 1)
    ranks = np.arange(1, y.size + 1)
    return float((cumulative[sorted_y == 1] / ranks[sorted_y == 1]).sum() / positive_count)


def pointing_game(
    labels: np.ndarray,
    scores: np.ndarray,
    soundscape_ids: Sequence[str],
) -> dict[str, float | int | None]:
    """Score whether each soundscape's highest window points into its event."""

    scores, labels = _validate_window_scores(scores, labels)
    groups = np.asarray([str(value) for value in soundscape_ids])
    if groups.shape != (labels.shape[0],):
        raise ValueError("soundscape_ids must have one value per window")
    hits = 0
    eligible = 0
    for group in dict.fromkeys(groups.tolist()):
        indices = np.flatnonzero(groups == group)
        class_positive = labels[indices].any(axis=0)
        if not class_positive.any():
            continue
        eligible += 1
        best_index = indices[int(np.argmax(scores[indices].max(axis=1)))]
        hits += int(labels[best_index].any())
    return {"hits": hits, "eligible_soundscapes": eligible, "accuracy": hits / eligible if eligible else None}


def _bootstrap_groups(
    labels: np.ndarray,
    scores: np.ndarray,
    soundscape_ids: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray, Sequence[str]], float | None],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float] | None:
    groups = np.asarray([str(value) for value in soundscape_ids])
    unique = np.asarray(list(dict.fromkeys(groups.tolist())))
    if draws <= 0 or unique.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    for _ in range(int(draws)):
        sampled = rng.integers(0, unique.size, size=unique.size)
        selected_groups = unique[sampled]
        selected = np.concatenate([np.flatnonzero(groups == group) for group in selected_groups])
        value = metric(labels[selected], scores[selected], groups[selected])
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _pointing_accuracy(labels: np.ndarray, scores: np.ndarray, groups: Sequence[str]) -> float | None:
    result = pointing_game(labels, scores, groups)
    return result["accuracy"]


def _bootstrap_binary_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    kind: str,
    *,
    seed: int,
    draws: int,
) -> tuple[float, float] | None:
    """Efficient cluster bootstrap using group multiplicities as sample weights."""

    unique, inverse = np.unique(groups, return_inverse=True)
    if draws <= 0 or unique.size == 0:
        return None
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, unique.size, size=(int(draws), unique.size))
    counts = np.zeros_like(sampled, dtype=np.float64)
    for row_index, values in enumerate(sampled):
        counts[row_index] = np.bincount(values, minlength=unique.size)
    if kind == "pointing":
        hits = np.zeros(unique.size, dtype=np.float64)
        eligible = np.zeros(unique.size, dtype=np.float64)
        for group_index in range(unique.size):
            rows = np.flatnonzero(inverse == group_index)
            if not labels[rows].any():
                continue
            eligible[group_index] = 1.0
            hits[group_index] = float(labels[rows[np.argmax(scores[rows])]])
        denominators = counts @ eligible
        values = np.divide(counts @ hits, denominators, out=np.full(int(draws), np.nan), where=denominators > 0.0)
    else:
        order = np.argsort(scores, kind="mergesort") if kind == "auroc" else np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order].astype(np.float64)
        sorted_scores = scores[order]
        sorted_inverse = inverse[order]
        values = np.full(int(draws), np.nan, dtype=np.float64)
        tie_group = np.r_[0, np.cumsum(sorted_scores[1:] != sorted_scores[:-1])]
        for draw_index, group_counts in enumerate(counts):
            weights = group_counts[sorted_inverse]
            positive = weights * sorted_labels
            if kind == "auroc":
                negative = weights * (1.0 - sorted_labels)
                total_positive = float(positive.sum())
                total_negative = float(negative.sum())
                if total_positive == 0.0 or total_negative == 0.0:
                    continue
                negative_by_tie = np.bincount(tie_group, weights=negative)
                negative_before_tie = np.cumsum(negative_by_tie) - negative_by_tie
                u = float(np.sum(positive * (negative_before_tie[tie_group] + 0.5 * negative)))
                values[draw_index] = u / (total_positive * total_negative)
            elif kind == "average_precision":
                total_positive = float(positive.sum())
                if total_positive == 0.0:
                    continue
                cumulative_positive = np.cumsum(positive)
                cumulative_weight = np.cumsum(weights)
                precision = np.divide(
                    cumulative_positive,
                    cumulative_weight,
                    out=np.zeros_like(cumulative_positive),
                    where=cumulative_weight > 0.0,
                )
                values[draw_index] = float(np.sum(precision * positive) / total_positive)
            else:
                raise ValueError(f"unknown bootstrap metric: {kind}")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def compute_task6_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: Sequence[str],
    soundscape_ids: Sequence[str],
    *,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Compute per-class AUROC/AP and class-specific pointing-game CIs."""

    scores, labels = _validate_window_scores(scores, labels)
    if scores.shape[1] != len(class_names):
        raise ValueError("class_names width does not match scores")
    groups = np.asarray([str(value) for value in soundscape_ids])
    if groups.shape != (scores.shape[0],):
        raise ValueError("soundscape_ids must have one value per window")
    results: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        class_labels = labels[:, class_index]
        class_scores = scores[:, class_index]
        auroc = binary_auroc(class_labels, class_scores)
        average_precision = binary_average_precision(class_labels, class_scores)
        eligible_groups = [
            group for group in dict.fromkeys(groups.tolist()) if labels[groups == group, class_index].any()
        ]

        def class_pointing(y: np.ndarray, s: np.ndarray, sampled_groups: Sequence[str]) -> float | None:
            y2 = y[:, class_index : class_index + 1]
            s2 = s[:, class_index : class_index + 1]
            return _pointing_accuracy(y2, s2, sampled_groups)

        auroc_ci = _bootstrap_binary_ci(
            class_labels,
            class_scores,
            groups,
            "auroc",
            seed=seed + class_index * 11,
            draws=bootstrap_resamples,
        )
        ap_ci = _bootstrap_binary_ci(
            class_labels,
            class_scores,
            groups,
            "average_precision",
            seed=seed + class_index * 11 + 1,
            draws=bootstrap_resamples,
        )
        pointing_ci = _bootstrap_binary_ci(
            class_labels,
            class_scores,
            groups,
            "pointing",
            seed=seed + class_index * 11 + 2,
            draws=bootstrap_resamples,
        )
        results[str(class_name)] = {
            "positive_windows": int(class_labels.sum()),
            "positive_soundscapes": len(eligible_groups),
            "auroc": auroc,
            "auroc_ci": list(auroc_ci) if auroc_ci else None,
            "average_precision": average_precision,
            "average_precision_ci": list(ap_ci) if ap_ci else None,
            "pointing_game": class_pointing(labels, scores, groups),
            "pointing_game_ci": list(pointing_ci) if pointing_ci else None,
        }
    return {
        "classes": [str(value) for value in class_names],
        "n_windows": int(labels.shape[0]),
        "n_soundscapes": int(np.unique(groups).size),
        "bootstrap_resamples": int(bootstrap_resamples),
        "by_class": results,
    }


def _target_classes(target_classes: Sequence[int], sample_count: int) -> np.ndarray:
    targets = np.asarray(target_classes, dtype=np.int64).reshape(-1)
    if targets.shape != (sample_count,) or (targets < 0).any():
        raise ValueError("target_classes must contain one non-negative class index per sample")
    return targets


def cbm_summed_positive_contribution(contributions: np.ndarray, target_classes: Sequence[int]) -> np.ndarray:
    """Score windows by summed positive CBM concept contributions."""

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("CBM contributions must have shape [samples, windows, classes, concepts]")
    targets = _target_classes(target_classes, values.shape[0])
    selected = values[np.arange(values.shape[0]), :, targets, :]
    return np.maximum(selected, 0.0).sum(axis=-1)


def cbm_silence_baseline_contribution(
    contributions: np.ndarray,
    silence_baseline_contributions: np.ndarray,
    target_classes: Sequence[int],
) -> np.ndarray:
    """Score positive CBM evidence above the fixed Task-1 silence baseline."""

    values = np.asarray(contributions, dtype=np.float64)
    baseline = np.asarray(silence_baseline_contributions, dtype=np.float64)
    if values.shape != baseline.shape:
        raise ValueError("CBM and silence-baseline contributions must have equal shape")
    centered = values - baseline
    return cbm_summed_positive_contribution(centered, target_classes)


def ast_per_window_logit(window_logits: np.ndarray, target_classes: Sequence[int]) -> np.ndarray:
    """Gather the target class AST logit for every temporal window."""

    logits = np.asarray(window_logits, dtype=np.float64)
    if logits.ndim != 3:
        raise ValueError("window_logits must have shape [samples, windows, classes]")
    targets = _target_classes(target_classes, logits.shape[0])
    return logits[np.arange(logits.shape[0])[:, None], np.arange(logits.shape[1])[None, :], targets[:, None]]


def ast_occlusion_class_logit_drop(
    full_clip_logits: np.ndarray,
    occluded_window_logits: np.ndarray,
    target_classes: Sequence[int],
) -> np.ndarray:
    """Return full-clip target logit minus each window-silenced target logit."""

    full = np.asarray(full_clip_logits, dtype=np.float64)
    occluded = np.asarray(occluded_window_logits, dtype=np.float64)
    if (
        full.ndim != 2
        or occluded.ndim != 3
        or full.shape[0] != occluded.shape[0]
        or full.shape[1] != occluded.shape[2]
    ):
        raise ValueError("full logits must be [samples, classes] and occluded logits [samples, windows, classes]")
    targets = _target_classes(target_classes, full.shape[0])
    full_target = full[np.arange(full.shape[0]), targets]
    occluded_target = occluded[
        np.arange(occluded.shape[0])[:, None],
        np.arange(occluded.shape[1])[None, :],
        targets[:, None],
    ]
    return full_target[:, None] - occluded_target


def aggregate_attention_rollout_to_windows(
    rollout: np.ndarray,
    token_intervals: np.ndarray,
    window_intervals: np.ndarray,
) -> np.ndarray:
    """Split token attention by temporal overlap and sum it into windows."""

    attention = np.asarray(rollout, dtype=np.float64)
    tokens = np.asarray(token_intervals, dtype=np.float64)
    windows = np.asarray(window_intervals, dtype=np.float64)
    if attention.ndim != 2 or tokens.shape != (attention.shape[1], 2):
        raise ValueError("rollout must be [samples, tokens] and token_intervals [tokens, 2]")
    if windows.ndim == 2:
        windows = np.broadcast_to(windows, (attention.shape[0],) + windows.shape)
    if windows.ndim != 3 or windows.shape[0] != attention.shape[0] or windows.shape[2] != 2:
        raise ValueError("window_intervals must be [windows, 2] or [samples, windows, 2]")
    result = np.zeros((attention.shape[0], windows.shape[1]), dtype=np.float64)
    token_durations = np.maximum(tokens[:, 1] - tokens[:, 0], 0.0)
    for sample_index in range(attention.shape[0]):
        for window_index, (window_start, window_end) in enumerate(windows[sample_index]):
            overlap = np.maximum(
                0.0,
                np.minimum(window_end, tokens[:, 1]) - np.maximum(window_start, tokens[:, 0]),
            )
            fractions = np.divide(overlap, token_durations, out=np.zeros_like(overlap), where=token_durations > 0.0)
            result[sample_index, window_index] = float(np.sum(attention[sample_index] * fractions))
    return result


def score_windows(kind: str, *, target_classes: Sequence[int], **payload: Any) -> np.ndarray:
    """Unified scorer interface used by all Task-6 attribution sources."""

    if kind == "cbm_positive_contribution":
        return cbm_summed_positive_contribution(payload["contributions"], target_classes)
    if kind == "cbm_silence_baseline":
        return cbm_silence_baseline_contribution(
            payload["contributions"], payload["silence_baseline_contributions"], target_classes
        )
    if kind == "ast_window_logit":
        return ast_per_window_logit(payload["window_logits"], target_classes)
    if kind == "ast_occlusion_drop":
        return ast_occlusion_class_logit_drop(
            payload["full_clip_logits"], payload["occluded_window_logits"], target_classes
        )
    if kind == "ast_attention_rollout":
        return aggregate_attention_rollout_to_windows(
            payload["rollout"], payload["token_intervals"], payload["window_intervals"]
        )
    raise ValueError(f"unknown Task-6 scorer: {kind}")


def plot_qualitative_examples(
    examples: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    max_examples: int = 6,
) -> None:
    """Plot up to six score timelines with ground-truth spans shaded."""

    import matplotlib.pyplot as plt

    selected = list(examples)[: int(max_examples)]
    if not selected:
        raise ValueError("cannot plot zero qualitative examples")
    figure, axes = plt.subplots(len(selected), 1, figsize=(11, 2.6 * len(selected)), squeeze=False)
    for axis, example in zip(axes[:, 0], selected):
        starts = np.asarray(example["window_starts"], dtype=np.float64)
        ends = np.asarray(example["window_ends"], dtype=np.float64)
        for name, values in example["scores"].items():
            axis.plot((starts + ends) / 2.0, values, marker=".", linewidth=1.2, label=str(name))
        for event in example.get("events", []):
            interval = _event_interval(event)
            if interval is not None:
                axis.axvspan(interval[0], interval[1], color="tab:orange", alpha=0.18)
        axis.set_ylabel("score")
        axis.set_title(str(example.get("soundscape_id", "soundscape")))
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-windows", type=Path, required=True)
    parser.add_argument("--hop", type=float, choices=(0.5, 0.25), required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        required=True,
        help="NPZ with labels/scores or scores and class metadata",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = load_event_window_rows(args.event_windows, hop_seconds=args.hop)
    labels, class_names, soundscape_ids = event_active_window_labels(rows)
    with np.load(args.scores, allow_pickle=False) as archive:
        scores = archive["scores"]
        if "class_names" in archive:
            class_names = [str(value) for value in archive["class_names"].tolist()]
        if "soundscape_ids" in archive:
            soundscape_ids = [str(value) for value in archive["soundscape_ids"].tolist()]
    if scores.shape != labels.shape:
        raise ValueError(f"score shape {scores.shape} does not match prepared label shape {labels.shape}")
    result = compute_task6_metrics(
        labels,
        scores,
        class_names,
        soundscape_ids,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_windows": result["n_windows"], "n_soundscapes": result["n_soundscapes"]}, indent=2))


if __name__ == "__main__":
    main()
