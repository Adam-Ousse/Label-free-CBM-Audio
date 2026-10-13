import json

from scripts.concepts.generation_variance import _overlap, _parse_concepts, _prompt_record


def test_variance_overlap_reports_exact_and_normalized_metrics():
    result = _overlap(
        ["high-pitched", "rough", "rough"], ["high pitched", "metallic"]
    )
    assert result["vocabulary_size"] == 2
    assert result["exact_overlap_count"] == 0
    assert result["normalized_phrase_overlap_count"] == 1
    assert result["duplicate_count_exact"] == 1


def test_prompt_record_hashes_exact_prompt():
    record = _prompt_record("system", "user")
    assert record["prompt_hash"]
    assert record["system_prompt"] == "system"
    assert record["user_prompt"] == "user"


def test_parse_concepts_keeps_raw_values_and_applies_canonical_cleaning():
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "concepts": [
                                " rough. ",
                                "rough",
                                "too many words here now and again",
                            ]
                        }
                    )
                }
            }
        ]
    }
    raw, cleaned = _parse_concepts(response)
    assert raw == [" rough. ", "rough", "too many words here now and again"]
    assert cleaned == ["rough"]
