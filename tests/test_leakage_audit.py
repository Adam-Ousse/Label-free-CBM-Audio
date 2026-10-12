from __future__ import annotations

import numpy as np

from scripts.concepts.leakage_audit import (
    CLAP_THRESHOLD,
    MPNET_THRESHOLD,
    class_name_token_matches,
    content_lemmas,
    make_audit_rows,
    score_concepts,
)


def test_content_lemmas_ignore_stop_words_and_lemmatize_inflections():
    assert content_lemmas("The children are playing") == {"child", "play"}
    assert content_lemmas("a dog's barking") == {"dog", "bark"}


def test_class_name_overlap_is_lemma_based_and_ignores_stop_words():
    matches = class_name_token_matches(
        "children playing in the street", ["children_playing", "dog_bark", "the"]
    )
    assert matches == {"children_playing": ["child", "play"]}


def test_leakage_reasons_are_recorded_independently_and_strictly():
    concepts = ["dog barking", "soft texture", "exact boundary"]
    classes = ["dog_bark", "cat"]
    mpnet = np.asarray([[0.71, 0.1], [0.1, 0.2], [MPNET_THRESHOLD, 0.2]])
    clap = np.asarray([[0.1, 0.2], [0.86, 0.1], [0.2, CLAP_THRESHOLD]])
    scores = score_concepts(
        concepts,
        classes,
        mpnet_embeddings=mpnet,
        class_mpnet_embeddings=np.eye(2),
        clap_embeddings=clap,
        class_clap_embeddings=np.eye(2),
    )
    rows = make_audit_rows("toy", "lf", concepts, classes, scores)
    assert rows[0]["flagged"] is True
    assert rows[0]["reasons"] == "class_name_token_overlap;mpnet_similarity_gt_0.7"
    assert rows[1]["reasons"] == "clap_text_similarity_gt_0.85"
    assert rows[2]["flagged"] is False
    assert rows[2]["reasons"] == ""


def test_flagged_weight_mass_is_carried_per_concept():
    scores = score_concepts(
        ["dog bark"],
        ["dog"],
        mpnet_embeddings=np.asarray([[0.0]]),
        class_mpnet_embeddings=np.asarray([[1.0]]),
        clap_embeddings=np.asarray([[0.0]]),
        class_clap_embeddings=np.asarray([[1.0]]),
    )
    row = make_audit_rows("toy", "lf", ["dog bark"], ["dog"], scores, weights={"dog bark": 3.5})[0]
    assert row["flagged"] is True
    assert row["abs_W_F_mass"] == 3.5
