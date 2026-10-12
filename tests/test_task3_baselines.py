import torch

from scripts.evaluation.task3_baselines import (
    class_prompts,
    generate_random_directions,
    generate_random_vocabulary,
    mean_nonzeros_per_class,
    merge_vocabularies,
    select_path_entry,
)


def test_nec_path_selection_uses_first_lambda_reaching_target():
    path = [
        {"lam": 1.0, "metrics": {"acc_val": 0.50}, "weight": torch.tensor([[1.0, 0.0, 0.0, 0.0]])},
        {"lam": 0.1, "metrics": {"acc_val": 0.90}, "weight": torch.tensor([[1.0, 1.0, 1.0, 0.0]])},
        {"lam": 0.01, "metrics": {"acc_val": 0.99}, "weight": torch.tensor([[1.0, 1.0, 1.0, 1.0]])},
    ]
    selected = select_path_entry(path, target_nec=2)
    assert selected["lam"] == 1.0
    assert selected["actual_mean_nnz_per_class"] == 1.0
    assert selected["nec_reached"] is True


def test_validation_selection_never_uses_test_metrics():
    path = [
        {
            "lam": 0.2,
            "metrics": {"acc_val": 0.8, "acc_test": 0.99},
            "weight": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        },
        {
            "lam": 0.1,
            "metrics": {"acc_val": 0.7, "acc_test": 0.01},
            "weight": torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        },
    ]
    selected = select_path_entry(path)
    assert selected["lam"] == 0.2
    assert "acc_test" in selected["metrics"]  # present only as an inert fixture field


def test_random_direction_generation_is_reproducible_and_shaped():
    first = generate_random_directions(7, 4, seed=123)
    second = generate_random_directions(7, 4, seed=123)
    different = generate_random_directions(7, 4, seed=124)
    assert first.shape == (4, 7)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_random_word_generation_is_reproducible_unique_and_width_exact():
    first = generate_random_vocabulary(20, seed=123)
    second = generate_random_vocabulary(20, seed=123)
    different = generate_random_vocabulary(20, seed=124)
    assert first == second
    assert first != different
    assert len(first) == len(set(first)) == 20


def test_prompt_protocols_are_dataset_specific():
    assert class_prompts("esc50", ["dog", "rain"]) == ["the sound of dog", "the sound of rain"]
    assert class_prompts("urbansound8k", ["dog bark"]) == ["the sound of dog bark"]
    assert class_prompts("cremad", ["happy"]) == ["a person speaking in a happy voice"]


def test_vocab_merge_is_stable_and_clean_filter_is_applied(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    clean = tmp_path / "clean.txt"
    first.write_text("alpha\nbeta\n", encoding="utf-8")
    second.write_text("beta\ngamma\n", encoding="utf-8")
    clean.write_text("gamma\nbeta\n", encoding="utf-8")
    assert merge_vocabularies([first, second]) == ["alpha", "beta", "gamma"]
    assert merge_vocabularies([first, second], clean_filter=clean) == ["beta", "gamma"]


def test_mean_nonzeros_per_class_uses_expected_threshold():
    weights = torch.tensor([[1.0, 1e-6, 0.0], [0.0, 2.0, 3.0]])
    assert mean_nonzeros_per_class(weights) == 1.5
