import json
from pathlib import Path

from data.prepare_urbansed import resolve_event_source


def test_resolve_event_source_cross_checks_manifest_fold():
    index = {"100032-3-0-0": [{"id": "100032-3-0-0", "fold": 5, "class": "dog_bark"}]}
    result = resolve_event_source(
        {"source_file": "/source/fold5/100032-3-0-0.wav", "label": "dog_bark"}, index
    )

    assert result["status"] == "resolved"
    assert result["source"]["fold"] == 5


def test_resolve_event_source_logs_conflicting_fold_as_ambiguous():
    index = {"100032-3-0-0": [{"id": "100032-3-0-0", "fold": 5, "class": "dog_bark"}]}
    result = resolve_event_source(
        {"source_file": "/source/fold9/100032-3-0-0.wav", "label": "dog_bark"}, index
    )

    assert result["status"] == "ambiguous"
    assert result["reason"] == "source_path_fold_conflicts_with_manifest"


def test_missing_source_is_ambiguous():
    result = resolve_event_source({"label": "dog_bark"}, {})

    assert result["status"] == "ambiguous"
    assert result["reason"] == "missing_or_nonunique_source_clip_id"


def test_background_source_is_not_treated_as_urbansound_provenance():
    result = resolve_event_source(
        {
            "role": "background",
            "source_file": "/source/background/noise/brownian.wav",
            "label": "noise",
        },
        {},
    )

    assert result["status"] == "background"
