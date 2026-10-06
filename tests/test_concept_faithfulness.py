import json

import torch

from scripts.evaluation import evaluate_concept_faithfulness as faithfulness
from scripts.visualization import plot_concept_faithfulness as plotting


def test_removing_standardized_activation_matches_logit_update():
    a_hat = torch.tensor([2.0, -1.0])
    W_g = torch.tensor([[3.0, 4.0], [1.0, -2.0]])
    b_g = torch.tensor([0.5, -0.25])

    original = a_hat @ W_g.T + b_g
    direct = faithfulness.intervene_logits(a_hat, W_g, b_g, [0])
    formula = original - a_hat[0] * W_g[:, 0]

    assert torch.allclose(direct, formula)


def test_positive_ranking_and_random_population(tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    torch.save(torch.eye(3), checkpoint_dir / "W_c.pt")
    torch.save(torch.tensor([[2.0, 1.0, 0.5], [0.0, 1.0, 1.0]]), checkpoint_dir / "W_g.pt")
    torch.save(torch.zeros(2), checkpoint_dir / "b_g.pt")
    torch.save(torch.zeros(3), checkpoint_dir / "proj_mean.pt")
    torch.save(torch.ones(3), checkpoint_dir / "proj_std.pt")
    (checkpoint_dir / "concepts.txt").write_text("c0\nc1\nc2\n", encoding="utf-8")
    (checkpoint_dir / "args.txt").write_text(json.dumps({"dataset": "esc50"}), encoding="utf-8")

    rows = [
        {"id": "sample", "audio_path": "sample.wav", "label_idx": 1, "label": "class1", "dataset": "esc50"}
    ]
    features = torch.tensor([[1.0, 0.0, 1.0]])
    checkpoint = faithfulness.load_checkpoint(checkpoint_dir)
    a_hat, logits, contributions = faithfulness.reconstruct(features, checkpoint)

    # Class 0 is wrong; its positive contributions rank c0 (2.0) above c2 (0.5).
    assert int(logits.argmax()) == 0
    assert faithfulness._rank_positive(contributions[0, 0], 0.0) == [0, 2]

    records, summary = faithfulness.run_interventions(
        "esc50", rows, checkpoint, a_hat, logits, contributions,
        ks=[1], random_draws=3, seed=42, positive_tol=0.0
    )
    top = [row for row in records if row["strategy"] == "top"]
    random_rows = [row for row in records if row["strategy"] == "random"]
    assert top[0]["removed_concept_indices"] == "0"
    assert all(row["eligible_positive_count"] == 2 for row in random_rows)
    assert summary["by_k"]["1"]["top_successes"] == 1


def test_plot_table_validator_rejects_inconsistent_delta():
    rows = []
    for dataset in plotting.DATASETS:
        for k in plotting.KS:
            rows.append(
                {
                    "dataset": dataset,
                    "variant": "lf_broad",
                    "k": str(k),
                    "n_errors": "2",
                    "n_eligible": "2",
                    "n_excluded": "0",
                    "topk_recovered": "1",
                    "topk_recovery_rate": "0.5",
                    "topk_ci_low": "0.0",
                    "topk_ci_high": "1.0",
                    "random_recovery_mean": "0.25",
                    "random_ci_low": "0.0",
                    "random_ci_high": "0.5",
                    "delta_recovery": "0.2",
                    "delta_ci_low": "0.0",
                    "delta_ci_high": "0.5",
                }
            )

    try:
        plotting.validate_table(rows)
    except ValueError as error:
        assert "Delta" in str(error)
    else:
        raise AssertionError("inconsistent delta should be rejected")
