import numpy as np

from scripts.evaluation.task5_descriptor_analysis import (
    CONTROLLED_EDIT_SPECS,
    aggregate_controlled_edits,
    aggregate_dataset_concept_correlations,
    average_ranks,
    bootstrap_mean_ci,
    bootstrap_spearman_ci,
    correlate_pair,
    expected_sign_matches,
    load_descriptor_map,
    plot_controlled_edit_figure,
    random_concept_descriptor_null,
    spearman_rho,
    validate_mapping_against_models,
)


def test_author_reviewed_map_is_small_and_validated_against_model_vocabularies():
    mapping = load_descriptor_map("results/revision/t5_descriptors/concept_descriptor_map.json")
    assert len(mapping["mappings"]) == 17
    model_concepts = {
        "cremad": {"lf": {"a voice with high pitch", "a voice with low pitch"}},
        "esc50": {"lf": {"high pitch", "broadband"}},
        "urbansound8k": {"full": {"broadband", "tonal", "harmonic", "narrow bandwidth", "low pitch", "sharp attacks"}},
    }
    # The compact fixture exercises the validator without loading model files.
    small = {"mappings": mapping["mappings"][:2]}
    checks = validate_mapping_against_models(small, model_concepts)
    assert len(checks) == 2


def test_rank_and_spearman_and_sign_check_are_tie_aware():
    np.testing.assert_allclose(average_ranks([2.0, 1.0, 1.0]), [3.0, 1.5, 1.5])
    assert np.isclose(spearman_rho([1, 2, 3], [10, 20, 30]), 1.0)
    assert np.isclose(spearman_rho([1, 2, 3], [30, 20, 10]), -1.0)
    assert spearman_rho([1, 1, 1], [1, 2, 3]) is None
    assert expected_sign_matches(0.2, 1) is True
    assert expected_sign_matches(-0.2, 1) is False
    assert expected_sign_matches(None, 1) is None


def test_pair_correlation_and_bootstrap_are_reproducible():
    result = correlate_pair([1, 2, 3, np.nan], [2, 4, 6, 9], expected_sign=1, seed=5, bootstrap_resamples=100)
    assert result["n"] == 3
    assert result["expected_sign_match"] is True
    np.testing.assert_allclose(bootstrap_spearman_ci([1, 2, 3], [2, 4, 6], seed=5, draws=100), (1.0, 1.0))
    mean_ci = bootstrap_mean_ci([1, 2, 3], seed=5, draws=100)
    assert mean_ci is not None and mean_ci[0] <= 2.0 <= mean_ci[1]


def test_dataset_concept_aggregation_and_null_pairings():
    mapping = {"mappings": [{"concept": "high", "datasets": ["d"], "descriptor": "pitch", "expected_sign": 1}]}
    rows = [
        {
            "dataset": "d",
            "model": "m",
            "concept": "high",
            "descriptor": "pitch",
            "concept_value": x,
            "descriptor_value": x * 2,
        }
        for x in (1, 2, 3, 4)
    ]
    aggregate = aggregate_dataset_concept_correlations(rows, mapping, bootstrap_resamples=50)
    assert np.isclose(aggregate[0]["spearman_rho"], 1.0)
    null = random_concept_descriptor_null(
        np.arange(12, dtype=float).reshape(4, 3),
        np.arange(8, dtype=float).reshape(4, 2),
        ["high", "other", "third"],
        ["pitch", "rms"],
        [("high", "pitch")],
        draws=1000,
        seed=3,
    )
    assert len(null) == 1000
    assert all((row["concept"], row["descriptor"]) != ("high", "pitch") for row in null)


def test_controlled_edit_aggregation_is_fixed_map_and_plottable(tmp_path):
    mapping = {"mappings": [{"concept": "high", "datasets": ["d"], "descriptor": "pitch", "expected_sign": 1}]}
    batch = {
        "dataset": "d",
        "edit": "pitch_shift_plus_4_semitones",
        "concept_names": ["high", "other"],
        "descriptor_names": ["pitch", "rms"],
        "concept_changes": np.asarray([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float),
        "descriptor_changes": np.asarray([[10, 1], [20, 2], [30, 3], [40, 4]], dtype=float),
    }
    summary = aggregate_controlled_edits([batch], mapping, unmatched_draws=20, seed=7)
    assert summary[0]["mapping_is_fixed"] is True
    assert np.isclose(summary[0]["matched_mean_rho"], 1.0)
    assert summary[0]["matched_ci"] is not None
    assert len(CONTROLLED_EDIT_SPECS) == 6
    output = tmp_path / "controlled.png"
    plot_controlled_edit_figure(summary, output)
    assert output.is_file()
