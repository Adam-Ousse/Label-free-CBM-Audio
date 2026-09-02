#!/usr/bin/env python3
"""Train the fixed LF/Broad/Contrastive CBM ablation matrix."""

import argparse
import csv
import json
from pathlib import Path

import torch

import train_cbm


DATASET_CONFIG = {
    "esc50": {
        "backbone": "ast_esc50",
        "train_split": "fold1_train",
        "val_split": "fold1_val",
        "test_split": "fold1_test",
        "enforce_esc50_fold1_protocol": True,
    },
    "urbansound8k": {
        "backbone": "ast_urbansound8k",
        "train_split": "fold10_train",
        "val_split": "fold10_val",
        "test_split": "fold10_test",
        "enforce_esc50_fold1_protocol": False,
    },
    "cremad": {
        "backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "enforce_esc50_fold1_protocol": False,
    },
}

VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_CONFIG),
        default=list(DATASET_CONFIG),
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--saga-batch-size", type=int, default=256)
    parser.add_argument("--proj-batch-size", type=int, default=50000)
    parser.add_argument("--proj-steps", type=int, default=1000)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--concept-activation-cutoff", type=float, default=0.25)
    parser.add_argument("--interpretability-cutoff", type=float, default=0.45)
    parser.add_argument("--lam", type=float, default=0.0007)
    parser.add_argument("--elastic-alpha", type=float, default=0.99)
    parser.add_argument("--similarity-objective", choices=("cosine", "cosine_cubed"), default="cosine_cubed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation-root", default="saved_activations/audio_concept_ablation")
    parser.add_argument("--results-root", default="results/audio_concept_ablation/cbm")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def _concept_path(dataset, variant):
    return Path("data/concept_sets") / dataset / "concepts_ablation_{}.txt".format(variant)


def _line_count(path):
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _complete_model_dirs(models_dir):
    if not models_dir.exists():
        return []
    required = {"W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "metrics.txt", "concepts.txt", "args.txt"}
    return sorted(
        path
        for path in models_dir.iterdir()
        if path.is_dir() and required.issubset({child.name for child in path.iterdir()})
    )


def _summarize_model(dataset, variant, concept_path, model_dir):
    metrics = json.loads((model_dir / "metrics.txt").read_text(encoding="utf-8"))
    retained_concepts = _line_count(model_dir / "concepts.txt")
    input_concepts = _line_count(concept_path)
    W_g = torch.load(model_dir / "W_g.pt", map_location="cpu").float()
    nonzero_mask = W_g.abs() > 1e-5
    nonzero_per_class = nonzero_mask.sum(dim=1)
    test = metrics.get("test_metrics", {})
    sparsity = metrics.get("sparsity", {})
    return {
        "dataset": dataset,
        "variant": variant,
        "model_dir": str(model_dir),
        "concept_set": str(concept_path),
        "input_concepts": input_concepts,
        "retained_concepts": retained_concepts,
        "retained_fraction": retained_concepts / float(input_concepts),
        "test_loss": test.get("loss"),
        "test_accuracy": test.get("accuracy"),
        "test_f1_macro": test.get("f1_macro"),
        "test_f1_weighted": test.get("f1_weighted"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "nonzero_weights": int(sparsity.get("Non-zero weights", nonzero_mask.sum().item())),
        "total_weights": int(sparsity.get("Total weights", W_g.numel())),
        "fraction_nonzero": float(sparsity.get("Percentage non-zero", nonzero_mask.float().mean().item())),
        "sparsity_fraction": 1.0 - float(sparsity.get("Percentage non-zero", nonzero_mask.float().mean().item())),
        "avg_nonzero_per_class": float(nonzero_per_class.float().mean().item()),
        "min_nonzero_per_class": int(nonzero_per_class.min().item()),
        "max_nonzero_per_class": int(nonzero_per_class.max().item()),
    }


def _write_reports(results_root, rows, args):
    results_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "saga_batch_size": args.saga_batch_size,
            "proj_batch_size": args.proj_batch_size,
            "proj_steps": args.proj_steps,
            "n_iters": args.n_iters,
            "concept_activation_cutoff": args.concept_activation_cutoff,
            "interpretability_cutoff": args.interpretability_cutoff,
            "lam": args.lam,
            "elastic_alpha": args.elastic_alpha,
            "similarity_objective": args.similarity_objective,
            "nonzero_threshold": 1e-5,
        },
        "runs": rows,
    }
    (results_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        fields = list(rows[0])
        with (results_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _train_args(args, dataset, variant, concept_path, models_dir):
    config = DATASET_CONFIG[dataset]
    return argparse.Namespace(
        dataset=dataset,
        concept_set=str(concept_path),
        backbone=config["backbone"],
        clap_model="laion/clap-htsat-unfused",
        device=args.device,
        batch_size=args.batch_size,
        saga_batch_size=args.saga_batch_size,
        proj_batch_size=args.proj_batch_size,
        feature_layer="layer4",
        activation_dir=str(Path(args.activation_root) / dataset),
        save_dir=str(models_dir),
        clip_cutoff=args.concept_activation_cutoff,
        concept_activation_cutoff=args.concept_activation_cutoff,
        projection_threshold=None,
        proj_steps=args.proj_steps,
        interpretability_cutoff=args.interpretability_cutoff,
        lam=args.lam,
        elastic_alpha=args.elastic_alpha,
        n_iters=args.n_iters,
        print=False,
        train_split=config["train_split"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        enforce_esc50_fold1_protocol=config["enforce_esc50_fold1_protocol"],
        audioset_streaming=False,
        audioset_cache_dir=None,
        audioset_max_items=None,
        audioset_subset=None,
        audioset_train_subset=None,
        audioset_val_subset=None,
        audioset_test_subset=None,
        similarity_objective=args.similarity_objective,
        prompt_template=None,
        max_concepts=None,
        results_dir=str(Path(args.results_root) / dataset / variant),
        run_name=variant,
        seed=args.seed,
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    results_root = Path(args.results_root)
    rows = []

    for dataset in args.datasets:
        for variant in args.variants:
            concept_path = _concept_path(dataset, variant)
            if not concept_path.is_file():
                raise FileNotFoundError("Missing ablation concept set: {}".format(concept_path))
            run_dir = results_root / dataset / variant
            models_dir = run_dir / "models"
            run_dir.mkdir(parents=True, exist_ok=True)
            complete = _complete_model_dirs(models_dir)
            if complete and not args.restart:
                model_dir = complete[-1]
                print("\n[resume] {} {} -> {}".format(dataset, variant, model_dir))
            else:
                models_dir.mkdir(parents=True, exist_ok=True)
                print("\n=== train {} / {} ===".format(dataset, variant))
                print("concepts={} backbone={}".format(_line_count(concept_path), DATASET_CONFIG[dataset]["backbone"]))
                model_dir = Path(
                    train_cbm.train_cbm_and_save(
                        _train_args(args, dataset, variant, concept_path, models_dir)
                    )
                )
            row = _summarize_model(dataset, variant, concept_path, model_dir)
            rows.append(row)
            (run_dir / "run_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            _write_reports(results_root, rows, args)
            print(
                "result accuracy={:.4f} macro-F1={:.4f} retained={}/{} nonzero={:.2%}".format(
                    row["test_accuracy"],
                    row["test_f1_macro"],
                    row["retained_concepts"],
                    row["input_concepts"],
                    row["fraction_nonzero"],
                )
            )
            torch.cuda.empty_cache()

    print("\nCompleted {} runs. Reports: {}".format(len(rows), results_root))


if __name__ == "__main__":
    main()
