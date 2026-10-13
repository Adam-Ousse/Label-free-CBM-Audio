from pathlib import Path

import numpy as np
import torch

from scripts.evaluation.task4_cv import (
    checkpoint_complete,
    fold_split_names,
    paired_mcnemar,
)


def test_fold_split_names_use_next_cyclic_fold():
    assert fold_split_names("esc50", 1) == ("fold1_train", "fold1_val", "fold1_test", 2)
    assert fold_split_names("esc50", 5)[3] == 1
    assert fold_split_names("urbansound8k", 10)[3] == 1


def test_paired_mcnemar_exact_table():
    result = paired_mcnemar(
        [0, 0, 1, 1, 2, 2],
        [0, 1, 1, 0, 2, 1],
        [0, 0, 0, 1, 1, 1],
    )
    assert result["n"] == 6
    assert result["cbm_correct_ast_wrong"] == 2
    assert result["ast_correct_cbm_wrong"] == 2
    assert 0.9 < result["exact_p_value"] <= 1.0


def test_checkpoint_complete_requires_model_config_processor_and_summary(tmp_path: Path):
    checkpoint = tmp_path / "best_model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"x")
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "preprocessor_config.json").write_text("{}")
    summary = tmp_path / "fold_summary.json"
    assert not checkpoint_complete(checkpoint, summary)
    summary.write_text("{}")
    assert checkpoint_complete(checkpoint, summary)
