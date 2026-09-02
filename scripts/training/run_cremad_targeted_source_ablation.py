#!/usr/bin/env python3
"""Run a matched CREMA-D source ablation for the speech-targeted vocabulary.

The full-vocabulary validation sweep selected the shared hyperparameters. This
runner freezes them and varies only which provenance sources enter the CBM.
"""

import argparse
import csv
import json
from pathlib import Path

import torch

import train_cbm
from .run_cremad_targeted_rerun import completed_models, model_args, summarize


VARIANTS = (
    ("lf", "Targeted LF"),
    ("lf_broad", "Targeted LF + broad"),
    ("lf_contrastive", "Targeted LF + contrastive"),
    ("full", "Targeted full union"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proj-steps", type=int, default=1000)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interpretability-cutoff", type=float, default=0.40)
    parser.add_argument("--lam", type=float, default=0.0015)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.experiment_root
    generation_dir = root / "generation" / "source_ablations_canonical"
    output_dir = root / "source_ablation_canonical"
    summary_path = output_dir / "source_ablation_summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite completed ablation: {}".format(summary_path))

    manifest_path = generation_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Missing {}. Run generate_cremad_targeted_concepts.py --export-existing first.".format(manifest_path)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant, display_name in VARIANTS:
        concept_path = generation_dir / manifest["variants"][variant]["grounded_file"]
        if not concept_path.exists():
            raise FileNotFoundError(concept_path)
        print("\n=== {}: {} input concepts ===".format(display_name, manifest["variants"][variant]["input_concepts"]), flush=True)
        save_dir = output_dir / "models" / variant
        save_dir.mkdir(parents=True, exist_ok=True)
        complete = completed_models(save_dir)
        if complete:
            model_dir = complete[-1]
            print("resume", model_dir, flush=True)
        else:
            model_dir = Path(train_cbm.train_cbm_and_save(model_args(
                args,
                concept_path,
                save_dir,
                args.interpretability_cutoff,
                args.lam,
                test_split="test",
            )))
        row = {
            "variant": variant,
            "model": display_name,
            "sources": manifest["variants"][variant]["sources"],
            "input_concepts": manifest["variants"][variant]["input_concepts"],
            "model_dir": str(model_dir),
            "validation": summarize(model_dir, "val", root / "activations"),
            "test": summarize(model_dir, "test", root / "activations"),
        }
        rows.append(row)
        (output_dir / "progress.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(
            "test accuracy={accuracy:.4f} macro-F1={macro_f1:.4f} retained={retained_concepts} sparsity={sparsity:.2%}".format(
                **row["test"]
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "selection_rule": (
            "Projection cutoff and sparse-head lambda selected once by validation macro-F1 on the full targeted "
            "vocabulary, then frozen for every source variant; test is not used for hyperparameter selection."
        ),
        "protocol": {
            "dataset": "cremad",
            "seed": args.seed,
            "concept_grounding": "a voice with {concept}",
            "activation_cutoff": 0.25,
            "interpretability_cutoff": args.interpretability_cutoff,
            "projection_steps": args.proj_steps,
            "similarity_objective": "cosine_cubed",
            "saga_iterations": args.n_iters,
            "lambda": args.lam,
            "elastic_alpha": 0.99,
            "shared_hyperparameters_across_variants": True,
            "test_used_for_selection": False,
        },
        "generation_manifest": str(manifest_path),
        "results": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = output_dir / "source_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant", "model", "input_concepts", "retained_concepts", "accuracy", "macro_f1",
            "weighted_f1", "balanced_accuracy", "sparsity", "nonzero_weights", "avg_nonzero_per_class",
            "model_dir",
        ])
        for row in rows:
            metrics = row["test"]
            writer.writerow([
                row["variant"], row["model"], row["input_concepts"], metrics["retained_concepts"],
                metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], metrics["balanced_accuracy"],
                metrics["sparsity"], metrics["nonzero_weights"], metrics["avg_nonzero_per_class"], row["model_dir"],
            ])
    print("wrote", summary_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
