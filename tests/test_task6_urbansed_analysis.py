import json

import numpy as np

from scripts.evaluation.task6_urbansed_analysis import (
    aggregate_attention_rollout_to_windows,
    ast_occlusion_class_logit_drop,
    ast_per_window_logit,
    binary_average_precision,
    binary_auroc,
    cbm_silence_baseline_contribution,
    cbm_summed_positive_contribution,
    compute_task6_metrics,
    event_active_window_labels,
    load_event_window_rows,
    plot_qualitative_examples,
    pointing_game,
    score_windows,
)


def _rows(hop=0.5):
    return [
        {
            "soundscape_id": "s0",
            "window_start": 0.0,
            "window_end": 1.0,
            "window_seconds": 1.0,
            "hop_seconds": hop,
            "events": [{"label": "dog_bark", "event_start": 0.75, "event_end": 1.5}],
        },
        {
            "soundscape_id": "s0",
            "window_start": hop,
            "window_end": hop + 1.0,
            "window_seconds": 1.0,
            "hop_seconds": hop,
            "events": [{"label": "dog_bark", "event_start": 0.75, "event_end": 1.5}],
        },
        {
            "soundscape_id": "s1",
            "window_start": 0.0,
            "window_end": 1.0,
            "window_seconds": 1.0,
            "hop_seconds": hop,
            "events": [],
        },
    ]


def test_event_labels_use_event_spans_and_validate_both_hops(tmp_path):
    rows = _rows()
    labels, classes, ids = event_active_window_labels(rows)
    assert classes == ["dog_bark"]
    assert labels[:, 0].tolist() == [1, 1, 0]
    assert ids == ["s0", "s0", "s1"]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert len(load_event_window_rows(path, hop_seconds=0.5)) == 3


def test_binary_metrics_and_pointing_game():
    y = [0, 1, 0, 1]
    s = [0.1, 0.9, 0.2, 0.8]
    assert binary_auroc(y, s) == 1.0
    assert binary_average_precision(y, s) == 1.0
    result = pointing_game(
        np.asarray([[0], [1], [0], [1]]),
        np.asarray([[0.1], [0.9], [0.2], [0.8]]),
        ["a", "a", "b", "b"],
    )
    assert result == {"hits": 2, "eligible_soundscapes": 2, "accuracy": 1.0}


def test_task6_metrics_include_per_class_bootstrap_intervals():
    labels = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int8)
    scores = np.asarray([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    result = compute_task6_metrics(labels, scores, ["a", "b"], ["s0", "s0", "s1", "s1"], bootstrap_resamples=50)
    assert result["by_class"]["a"]["auroc"] == 1.0
    assert len(result["by_class"]["a"]["auroc_ci"]) == 2
    assert len(result["by_class"]["b"]["average_precision_ci"]) == 2


def test_all_unified_scorers_support_target_class_windows():
    target = [1, 0]
    contributions = np.asarray(
        [
            [[[1, -2], [3, 4]], [[-1, 2], [5, -1]]],
            [[[2, 3], [-2, 4]], [[1, -1], [2, 3]]],
        ],
        dtype=float,
    )
    baseline = np.zeros_like(contributions)
    assert cbm_summed_positive_contribution(contributions, target).tolist() == [[7.0, 5.0], [5.0, 1.0]]
    assert cbm_silence_baseline_contribution(contributions, baseline, target).tolist() == [[7.0, 5.0], [5.0, 1.0]]
    logits = np.asarray([[[0, 2], [1, 3]], [[5, 1], [4, 2]]], dtype=float)
    np.testing.assert_allclose(ast_per_window_logit(logits, target), [[2, 3], [5, 4]])
    full = np.asarray([[0, 4], [3, 1]], dtype=float)
    occluded = np.asarray([[[0, 2], [0, 3]], [[2, 1], [1, 0]]], dtype=float)
    np.testing.assert_allclose(ast_occlusion_class_logit_drop(full, occluded, target), [[2, 1], [1, 2]])
    np.testing.assert_allclose(
        score_windows("ast_window_logit", window_logits=logits, target_classes=target),
        [[2, 3], [5, 4]],
    )


def test_attention_rollout_splits_token_mass_by_overlap():
    rollout = np.asarray([[0.2, 0.8]])
    tokens = np.asarray([[0.0, 1.0], [1.0, 2.0]])
    windows = np.asarray([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]])
    np.testing.assert_allclose(aggregate_attention_rollout_to_windows(rollout, tokens, windows), [[0.2, 0.5, 0.8]])
    np.testing.assert_allclose(
        score_windows(
            "ast_attention_rollout",
            target_classes=[0],
            rollout=rollout,
            token_intervals=tokens,
            window_intervals=windows,
        ),
        [[0.2, 0.5, 0.8]],
    )


def test_qualitative_plot_shades_event_spans(tmp_path):
    output = tmp_path / "qualitative.png"
    plot_qualitative_examples(
        [
            {
                "soundscape_id": "s0",
                "window_starts": [0.0, 0.5],
                "window_ends": [1.0, 1.5],
                "scores": {"AST": [0.1, 0.9]},
                "events": [{"label": "dog_bark", "event_start": 0.75, "event_end": 1.25}],
            }
        ],
        output,
    )
    assert output.is_file()
