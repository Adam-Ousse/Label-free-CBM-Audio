from scripts.audit.revision_preparation import (
    actor_id_from_row,
    deterministic_actor_assignment,
    pairwise_intersections,
)


def test_actor_id_and_overlap_are_manifest_level():
    rows = {
        "train": [{"id": "1001_DFA_ANG_XX"}],
        "val": [{"id": "1001_IEO_ANG_HI"}, {"id": "1002_DFA_ANG_XX"}],
        "test": [{"id": "1003_DFA_ANG_XX"}],
    }
    assert actor_id_from_row(rows["train"][0]) == "1001"
    overlaps = pairwise_intersections(rows)
    assert overlaps["train_and_val"] == ["1001"]
    assert overlaps["train_and_test"] == []


def test_deterministic_actor_assignment_is_disjoint_and_repeatable():
    rows = [{"id": f"{actor}_DFA_ANG_XX"} for actor in ("1001", "1002", "1003", "1004", "1005")]
    first, quotas = deterministic_actor_assignment(rows, seed=17)
    second, second_quotas = deterministic_actor_assignment(rows, seed=17)
    assert first == second
    assert quotas == second_quotas
    assert set(first) == {"1001", "1002", "1003", "1004", "1005"}
    assert len(set(first.values())) == 3


def test_deterministic_actor_assignment_accepts_explicit_quotas():
    rows = [{"id": f"{actor}_DFA_ANG_XX"} for actor in ("1001", "1002", "1003", "1004", "1005")]
    assignment, quotas = deterministic_actor_assignment(
        rows,
        seed=42,
        actor_counts={"train": 3, "val": 1, "test": 1},
    )
    assert quotas == {"train": 3, "val": 1, "test": 1}
    assert {split: list(assignment.values()).count(split) for split in ("train", "val", "test")} == quotas
