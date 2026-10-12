from pathlib import Path

from scripts.training.run_t2_clean_retraining import (
    checkpoint_input_equivalence,
    vocabulary_equivalence,
)


def test_vocabulary_equivalence_ignores_only_terminal_newline(tmp_path: Path):
    original = tmp_path / "original.txt"
    clean = tmp_path / "clean.txt"
    original.write_bytes(b"alpha\nbeta")
    clean.write_bytes(b"alpha\nbeta\n")

    result = vocabulary_equivalence(original, clean)

    assert result["content_equal"] is True
    assert result["byte_equal"] is False
    assert result["original_concepts"] == result["clean_concepts"] == 2


def test_checkpoint_input_equivalence_requires_existing_matching_file(tmp_path: Path):
    original = tmp_path / "original.txt"
    original.write_text("alpha\nbeta\n", encoding="utf-8")

    missing = checkpoint_input_equivalence(original, {"concept_set": "missing.txt"})
    assert missing["canonical_input_exists"] is False
    assert missing["canonical_input_content_equal"] is False

    canonical = tmp_path / "canonical.txt"
    canonical.write_text("alpha\nbeta\n", encoding="utf-8")
    matching = checkpoint_input_equivalence(original, {"concept_set": str(canonical)})
    assert matching["canonical_input_exists"] is True
    assert matching["canonical_input_content_equal"] is True
