import json
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[1] / "docs"


def test_showcase_schema_and_audio_references_are_complete():
    payload = json.loads(
        (DOCS_ROOT / "assets/data/esc50_showcase.json").read_text(encoding="utf-8")
    )
    examples = [example for item in payload["classes"] for example in item["examples"]]

    assert payload["dataset"] == "esc50"
    assert payload["variant"] == "lf_broad"
    assert len(payload["classes"]) == 50
    assert len(examples) == 100
    assert any(example["explanation"]["correct"] for example in examples)
    assert any(not example["explanation"]["correct"] for example in examples)

    referenced_audio = {example["audio"] for example in examples}
    available_audio = {
        "assets/audio/" + path.name for path in (DOCS_ROOT / "assets/audio").glob("*.wav")
    }
    assert referenced_audio == available_audio

    for example in examples:
        assert 1 <= len(example["explanation"]["top_concepts"]) <= 5
        assert len(example["explanation"]["base_logits"]) == 50
        assert example["temporal"]["centers_sec"]
        assert example["temporal"]["series"]
