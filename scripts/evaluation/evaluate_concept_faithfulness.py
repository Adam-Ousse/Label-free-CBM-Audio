#!/usr/bin/env python3
"""Run quantitative concept-faithfulness interventions from cached AST features.

The evaluator intentionally does not import the AST/transformers stack.  It consumes
the pooled AST features used by the CBM training code and reconstructs the model as

    c = x @ W_c.T
    a_hat = (c - proj_mean) / proj_std
    logits = a_hat @ W_g.T + b_g

For each originally misclassified sample, the predicted wrong class is used to rank
concepts by strictly positive contribution ``a_hat[j] * W_g[predicted, j]``.  The
random control samples the same number of concepts from that sample's positive
contribution population, so zero-weight and non-supporting concepts cannot weaken
the baseline artificially.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


CANONICAL = {
    "esc50": {
        "test_split": "fold1_test",
        "expected_samples": 400,
        "expected_classes": 50,
        "expected_concepts": 671,
        "expected_errors": 26,
        "expected_backbone": "ast_esc50",
    },
    "urbansound8k": {
        "test_split": "fold10_test",
        "expected_samples": 837,
        "expected_classes": 10,
        "expected_concepts": 212,
        "expected_errors": 85,
        "expected_backbone": "ast_urbansound8k",
    },
    "cremad": {
        "test_split": "test",
        "expected_samples": 1489,
        "expected_classes": 6,
        "expected_concepts": 548,
        "expected_errors": 443,
        "expected_backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
    },
}


def _torch_load(path: Path) -> Any:
    """Load tensor-only or metadata bundles across supported torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions before the weights_only keyword
        return torch.load(path, map_location="cpu")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError("Invalid JSON on {}:{}".format(path, line_number)) from exc
                if not isinstance(row, dict):
                    raise ValueError("Manifest row {}:{} is not an object".format(path, line_number))
                rows.append(row)
    return rows


def load_manifest(path: str | Path, dataset: str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Manifest not found: {}".format(path))
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("samples", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Manifest JSON must contain a list or a 'samples' list: {}".format(path))

    if not rows:
        raise ValueError("Manifest is empty: {}".format(path))
    required = ["id", "audio_path", "label_idx", "label"]
    if dataset in {"esc50", "urbansound8k"}:
        required.append("fold")
    else:
        required.append("split")
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError("Manifest is missing required fields {}: {}".format(missing, path))
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest sample IDs are not unique: {}".format(path))
    for row in rows:
        if row.get("dataset", dataset) != dataset:
            raise ValueError("Manifest row has a different dataset than --dataset: {}".format(row))
    return rows


def _load_features(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("AST feature cache not found: {}".format(path))
    if path.suffix.lower() == ".npy":
        import numpy as np

        features = torch.from_numpy(np.load(path))
        metadata: dict[str, Any] = {}
    elif path.suffix.lower() == ".npz":
        import numpy as np

        archive = np.load(path)
        key = "features" if "features" in archive else archive.files[0]
        features = torch.from_numpy(archive[key])
        metadata = {
            key: archive[key].tolist() if key in {"sample_ids", "labels"} else None
            for key in archive.files
            if key in {"sample_ids", "labels"}
        }
    else:
        payload = _torch_load(path)
        if isinstance(payload, torch.Tensor):
            features, metadata = payload, {}
        elif isinstance(payload, dict):
            key = next((name for name in ("features", "ast_features", "x") if name in payload), None)
            if key is None:
                raise ValueError("Feature bundle has no features/ast_features/x tensor: {}".format(path))
            features = payload[key]
            metadata = payload
        else:
            raise ValueError("Unsupported AST feature cache payload: {}".format(path))
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    features = features.detach().cpu().float()
    if features.ndim != 2:
        raise ValueError("Expected AST features with shape [N, D], got {}".format(tuple(features.shape)))
    return features, metadata


def _vector(payload: Any, name: str) -> torch.Tensor:
    tensor = _torch_load(Path(payload)) if isinstance(payload, (str, Path)) else payload
    tensor = torch.as_tensor(tensor).detach().cpu().float()
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError("{} must be a vector, got {}".format(name, tuple(tensor.shape)))
    return tensor


def load_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir)
    required = ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt", "args.txt")
    missing = [name for name in required if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Checkpoint is missing {}: {}".format(missing, checkpoint_dir))

    def tensor(name: str) -> torch.Tensor:
        value = _torch_load(checkpoint_dir / (name + ".pt"))
        return torch.as_tensor(value).detach().cpu().float()

    concepts = [line.strip() for line in (checkpoint_dir / "concepts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    args = json.loads((checkpoint_dir / "args.txt").read_text(encoding="utf-8"))
    return {
        "dir": str(checkpoint_dir),
        "W_c": tensor("W_c"),
        "W_g": tensor("W_g"),
        "b_g": tensor("b_g"),
        "proj_mean": _vector(checkpoint_dir / "proj_mean.pt", "proj_mean"),
        "proj_std": _vector(checkpoint_dir / "proj_std.pt", "proj_std"),
        "concepts": concepts,
        "args": args,
    }


def validate_inputs(
    dataset: str,
    manifest_rows: Sequence[dict[str, Any]],
    features: torch.Tensor,
    feature_metadata: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    spec = CANONICAL[dataset]
    if len(manifest_rows) != features.shape[0]:
        raise ValueError("Manifest/features sample mismatch: {} vs {}".format(len(manifest_rows), features.shape[0]))
    if features.shape[1] != checkpoint["W_c"].shape[1]:
        raise ValueError("AST feature width {} != W_c input width {}".format(features.shape[1], checkpoint["W_c"].shape[1]))
    W_c, W_g, b_g = checkpoint["W_c"], checkpoint["W_g"], checkpoint["b_g"]
    concepts = checkpoint["concepts"]
    if W_c.ndim != 2 or W_g.ndim != 2:
        raise ValueError("W_c and W_g must be matrices")
    if W_c.shape[0] != len(concepts) or W_g.shape[1] != len(concepts):
        raise ValueError("Checkpoint concept order/count mismatch")
    if W_g.shape[0] != b_g.numel():
        raise ValueError("W_g class dimension and b_g length differ")
    if checkpoint["proj_mean"].numel() != len(concepts) or checkpoint["proj_std"].numel() != len(concepts):
        raise ValueError("Projection normalization dimension mismatch")
    if not torch.isfinite(features).all() or not all(torch.isfinite(checkpoint[name]).all() for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")):
        raise ValueError("Non-finite value found in AST features or checkpoint")
    if torch.any(checkpoint["proj_std"] <= 0):
        raise ValueError("proj_std contains a non-positive value; refusing to change the checkpoint normalization")

    ids = feature_metadata.get("sample_ids") if isinstance(feature_metadata, dict) else None
    if ids is not None and [str(value) for value in ids] != [str(row["id"]) for row in manifest_rows]:
        raise ValueError("Feature-cache sample_ids do not match manifest ordering")
    cached_labels = feature_metadata.get("labels") if isinstance(feature_metadata, dict) else None
    if cached_labels is not None:
        cached = torch.as_tensor(cached_labels).view(-1).long()
        manifest_labels = torch.tensor([int(row["label_idx"]) for row in manifest_rows])
        if not torch.equal(cached, manifest_labels):
            raise ValueError("Feature-cache labels do not match manifest labels")

    args = checkpoint["args"]
    if args.get("dataset") not in (None, dataset):
        raise ValueError("Checkpoint args dataset is {}, expected {}".format(args.get("dataset"), dataset))
    if args.get("backbone") not in (None, spec["expected_backbone"]):
        raise ValueError("Checkpoint backbone is {}, expected {}".format(args.get("backbone"), spec["expected_backbone"]))
    if dataset == "esc50" and {int(row.get("fold")) for row in manifest_rows} != {1}:
        raise ValueError("ESC-50 manifest is not fold 1")
    if dataset == "urbansound8k" and {int(row.get("fold")) for row in manifest_rows} != {10}:
        raise ValueError("UrbanSound8K manifest is not fold 10")
    if dataset == "cremad" and {str(row.get("split")) for row in manifest_rows} != {"test"}:
        raise ValueError("CREMA-D manifest is not the canonical test split")
    if len(manifest_rows) != spec["expected_samples"]:
        raise ValueError("{} canonical test manifest must contain {}, found {}".format(dataset, spec["expected_samples"], len(manifest_rows)))
    if W_g.shape[0] != spec["expected_classes"]:
        raise ValueError("{} canonical checkpoint must have {} classes, found {}".format(dataset, spec["expected_classes"], W_g.shape[0]))
    if len(concepts) != spec["expected_concepts"]:
        raise ValueError("{} canonical checkpoint must retain {} concepts, found {}".format(dataset, spec["expected_concepts"], len(concepts)))


def reconstruct(features: torch.Tensor, checkpoint: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return standardized activations, logits, and per-class contributions."""
    W_c, W_g, b_g = checkpoint["W_c"], checkpoint["W_g"], checkpoint["b_g"]
    concepts = features @ W_c.T
    a_hat = (concepts - checkpoint["proj_mean"]) / checkpoint["proj_std"]
    logits = a_hat @ W_g.T + b_g
    contributions = a_hat.unsqueeze(1) * W_g.unsqueeze(0)
    return a_hat, logits, contributions


def intervene_logits(a_hat: torch.Tensor, W_g: torch.Tensor, b_g: torch.Tensor, indices: Iterable[int]) -> torch.Tensor:
    """Recompute logits after setting selected standardized concept activations to zero."""
    selected = list(indices)
    masked = a_hat.clone()
    if selected:
        masked[selected] = 0
    return masked @ W_g.T + b_g


def _rank_positive(contribution: torch.Tensor, positive_tol: float) -> list[int]:
    indices = [index for index, value in enumerate(contribution.tolist()) if value > positive_tol]
    return sorted(indices, key=lambda index: (-float(contribution[index]), index))


def _sample_random(indices: Sequence[int], k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(list(indices), k=k))


def _bootstrap_ci(values: Sequence[float], seed: int, draws: int = 10000) -> tuple[float, float] | None:
    if not values:
        return None
    tensor = torch.tensor(list(values), dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(0, tensor.numel(), (int(draws), tensor.numel()), generator=generator)
    means = tensor[indices].mean(dim=1)
    quantiles = torch.quantile(means, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return float(quantiles[0]), float(quantiles[1])


def _paired_permutation_p(values_a: Sequence[float], values_b: Sequence[float], seed: int, draws: int = 10000) -> float | None:
    if not values_a or len(values_a) != len(values_b):
        return None
    differences = torch.tensor(values_a, dtype=torch.float64) - torch.tensor(values_b, dtype=torch.float64)
    observed = float(abs(differences.mean()))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(0, 2, (int(draws), differences.numel()), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(torch.float64)
    permuted = (signs * differences.unsqueeze(0)).mean(dim=1).abs()
    return float((1 + int((permuted >= observed).sum())) / (int(draws) + 1))


def _class_name(rows: Sequence[dict[str, Any]], index: int) -> str:
    for row in rows:
        if int(row["label_idx"]) == int(index) and row.get("label") is not None:
            return str(row["label"])
    return str(index)


def run_interventions(
    dataset: str,
    rows: Sequence[dict[str, Any]],
    checkpoint: dict[str, Any],
    a_hat: torch.Tensor,
    logits: torch.Tensor,
    contributions: torch.Tensor,
    ks: Sequence[int],
    random_draws: int,
    seed: int,
    positive_tol: float,
    bootstrap_resamples: int = 10000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = torch.tensor([int(row["label_idx"]) for row in rows], dtype=torch.long, device=logits.device)
    predictions = logits.argmax(dim=1)
    wrong_indices = [int(index) for index in torch.nonzero(predictions != labels, as_tuple=False).view(-1).tolist()]
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"by_k": {}, "positive_tolerance": positive_tol}
    W_g, b_g = checkpoint["W_g"], checkpoint["b_g"]

    for k in ks:
        if k <= 0:
            raise ValueError("All k values must be positive")
        eligible: dict[int, list[int]] = {
            index: _rank_positive(contributions[index, int(predictions[index])], positive_tol)
            for index in wrong_indices
        }
        usable = [index for index in wrong_indices if len(eligible[index]) >= k]
        summary["by_k"][str(k)] = {
            "misclassified_samples": len(wrong_indices),
            "usable_samples": len(usable),
            "excluded_fewer_than_k_positive_contributors": len(wrong_indices) - len(usable),
            "random_draws": int(random_draws),
        }

        for sample_index in usable:
            sample_id = str(rows[sample_index]["id"])
            wrong_class = int(predictions[sample_index])
            truth = int(labels[sample_index])
            ranked = eligible[sample_index]
            selected_sets = [("top", 0, ranked[:k])]
            for draw in range(random_draws):
                draw_seed = int(seed) + sample_index * 1_000_003 + k * 9_176 + draw
                selected_sets.append(("random", draw, _sample_random(ranked, k, draw_seed)))

            for strategy, draw, selected in selected_sets:
                new_logits = intervene_logits(a_hat[sample_index], W_g, b_g, selected)
                new_prediction = int(new_logits.argmax())
                wrong_class_before = float(logits[sample_index, wrong_class])
                wrong_class_after = float(new_logits[wrong_class])
                truth_before = float(logits[sample_index, truth])
                truth_after = float(new_logits[truth])
                records.append(
                    {
                        "dataset": dataset,
                        "sample_index": sample_index,
                        "sample_id": sample_id,
                        "ground_truth": truth,
                        "ground_truth_name": _class_name(rows, truth),
                        "original_prediction": wrong_class,
                        "original_prediction_name": _class_name(rows, wrong_class),
                        "k": int(k),
                        "strategy": strategy,
                        "draw": int(draw),
                        "eligible_positive_count": len(ranked),
                        "removed_concept_indices": ";".join(str(index) for index in selected),
                        "removed_concepts": "||".join(checkpoint["concepts"][index] for index in selected),
                        "removed_wrong_class_contribution": float(sum(contributions[sample_index, wrong_class, index] for index in selected)),
                        "wrong_class_logit_before": wrong_class_before,
                        "wrong_class_logit_after": wrong_class_after,
                        "ground_truth_logit_before": truth_before,
                        "ground_truth_logit_after": truth_after,
                        "margin_before": wrong_class_before - truth_before,
                        "margin_after": float(new_logits[wrong_class] - new_logits[truth]),
                        "new_prediction": new_prediction,
                        "new_prediction_name": _class_name(rows, new_prediction),
                        "becomes_ground_truth": int(new_prediction == truth),
                    }
                )

        for strategy in ("top", "random"):
            subset = [row for row in records if row["k"] == k and row["strategy"] == strategy]
            successes = sum(row["becomes_ground_truth"] for row in subset)
            key = "{}_trials".format(strategy)
            summary["by_k"][str(k)][key] = len(subset)
            summary["by_k"][str(k)]["{}_successes".format(strategy)] = successes
            summary["by_k"][str(k)]["{}_success_rate".format(strategy)] = successes / len(subset) if subset else None
            summary["by_k"][str(k)]["{}_mean_wrong_class_logit_drop".format(strategy)] = (
                sum(row["wrong_class_logit_before"] - row["wrong_class_logit_after"] for row in subset) / len(subset)
                if subset else None
            )

        top_by_sample = {
            sample_index: next(row for row in records if row["k"] == k and row["strategy"] == "top" and row["sample_index"] == sample_index)
            for sample_index in usable
        }
        random_by_sample = {
            sample_index: [
                row for row in records
                if row["k"] == k and row["strategy"] == "random" and row["sample_index"] == sample_index
            ]
            for sample_index in usable
        }
        top_success = [float(top_by_sample[index]["becomes_ground_truth"]) for index in usable]
        random_success = [
            sum(row["becomes_ground_truth"] for row in random_by_sample[index]) / float(random_draws)
            for index in usable
        ]
        top_drop = [
            float(top_by_sample[index]["wrong_class_logit_before"] - top_by_sample[index]["wrong_class_logit_after"])
            for index in usable
        ]
        random_drop = [
            sum(row["wrong_class_logit_before"] - row["wrong_class_logit_after"] for row in random_by_sample[index]) / float(random_draws)
            for index in usable
        ]
        top_ci = _bootstrap_ci(top_success, seed + k * 101, draws=bootstrap_resamples)
        random_ci = _bootstrap_ci(random_success, seed + k * 103, draws=bootstrap_resamples)
        delta_values = [a - b for a, b in zip(top_success, random_success)]
        delta_ci = _bootstrap_ci(delta_values, seed + k * 107, draws=bootstrap_resamples)
        random_drop_ci = _bootstrap_ci(random_drop, seed + k * 109, draws=bootstrap_resamples)
        stats = summary["by_k"][str(k)]
        stats.update(
            {
                "n_original_errors": len(wrong_indices),
                "n_eligible": len(usable),
                "topk_recovered": int(sum(top_success)),
                "topk_recovery_rate": float(sum(top_success) / len(top_success)) if top_success else None,
                "topk_ci_low": top_ci[0] if top_ci else None,
                "topk_ci_high": top_ci[1] if top_ci else None,
                "random_recovery_mean": float(sum(random_success) / len(random_success)) if random_success else None,
                "random_ci_low": random_ci[0] if random_ci else None,
                "random_ci_high": random_ci[1] if random_ci else None,
                "delta_recovery": float(sum(delta_values) / len(delta_values)) if delta_values else None,
                "delta_ci_low": delta_ci[0] if delta_ci else None,
                "delta_ci_high": delta_ci[1] if delta_ci else None,
                "paired_permutation_p": _paired_permutation_p(top_success, random_success, seed + k * 113),
                "wrong_logit_drop_topk": float(sum(top_drop) / len(top_drop)) if top_drop else None,
                "wrong_logit_drop_random": float(sum(random_drop) / len(random_drop)) if random_drop else None,
                "wrong_logit_drop_random_ci_low": random_drop_ci[0] if random_drop_ci else None,
                "wrong_logit_drop_random_ci_high": random_drop_ci[1] if random_drop_ci else None,
            }
        )
    return records, summary


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CANONICAL), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ast-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--random-draws", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-tol", type=float, default=0.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_draws <= 0:
        raise ValueError("--random-draws must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    rows = load_manifest(args.manifest, args.dataset)
    features, feature_metadata = _load_features(args.ast_features)
    checkpoint = load_checkpoint(args.checkpoint_dir)
    validate_inputs(args.dataset, rows, features, feature_metadata, checkpoint)
    if args.device == "cuda":
        features = features.to("cuda")
        checkpoint = {
            key: (value.to("cuda") if isinstance(value, torch.Tensor) else value)
            for key, value in checkpoint.items()
        }
    a_hat, logits, contributions = reconstruct(features, checkpoint)
    labels = torch.tensor([int(row["label_idx"]) for row in rows], dtype=torch.long, device=logits.device)
    predictions = logits.argmax(dim=1)
    errors = int((predictions != labels).sum())
    print(json.dumps({
        "dataset": args.dataset,
        "samples": len(rows),
        "features": list(features.shape),
        "concepts": int(a_hat.shape[1]),
        "classes": int(logits.shape[1]),
        "accuracy": float((predictions == labels).float().mean()),
        "errors": errors,
        "expected_errors": CANONICAL[args.dataset]["expected_errors"],
        "reported_error_count_matches": errors == CANONICAL[args.dataset]["expected_errors"],
        "dry_run": bool(args.dry_run),
    }, indent=2))
    if args.dry_run:
        return
    if errors != CANONICAL[args.dataset]["expected_errors"]:
        raise RuntimeError(
            "Reconstructed error count {} does not match canonical reported count {}. "
            "Refusing to run interventions; check the exact AST cache, manifest, and checkpoint.".format(
                errors, CANONICAL[args.dataset]["expected_errors"]
            )
        )

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError("Refusing to overwrite non-empty output directory; use --overwrite: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    records, intervention_summary = run_interventions(
        args.dataset,
        rows,
        checkpoint,
        a_hat,
        logits,
        contributions,
        args.ks,
        args.random_draws,
        args.seed,
        args.positive_tol,
        args.bootstrap_resamples,
    )
    wrong_indices = torch.nonzero(predictions != labels, as_tuple=False).view(-1)
    torch.save(
        {
            "dataset": args.dataset,
            "sample_ids": [str(row["id"]) for row in rows],
            "labels": labels,
            "predictions": predictions,
            "logits": logits,
            "standardized_concept_activations": a_hat,
            "wrong_class_contributions": contributions[wrong_indices, predictions[wrong_indices]],
            "misclassified_indices": wrong_indices,
            "concepts": checkpoint["concepts"],
            "checkpoint_dir": str(args.checkpoint_dir),
            "formula": "logits = ((features @ W_c.T - proj_mean) / proj_std) @ W_g.T + b_g",
        },
        output_dir / "original_outputs.pt",
    )
    _write_csv(output_dir / "interventions.csv", records)
    payload = {
        "dataset": args.dataset,
        "variant": "full" if args.dataset == "cremad" else "lf_broad",
        "checkpoint_dir": str(args.checkpoint_dir),
        "manifest": str(args.manifest),
        "ast_features": str(args.ast_features),
        "ks": [int(k) for k in args.ks],
        "random_draws": int(args.random_draws),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
        "positive_contribution_definition": "a_hat[j] * W_g[predicted_wrong_class, j] > positive_tol",
        "removal_definition": "set standardized activation a_hat[j] to zero",
        "logit_update": "logits_prime = logits - sum_j(a_hat[j] * W_g[:, j])",
        "original": {
            "num_samples": len(rows),
            "num_misclassified": errors,
            "accuracy": float((predictions == labels).float().mean()),
        },
        "interventions": intervention_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Wrote {} intervention rows to {}".format(len(records), output_dir))


if __name__ == "__main__":
    main()
