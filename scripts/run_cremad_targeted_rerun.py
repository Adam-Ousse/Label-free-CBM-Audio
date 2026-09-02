#!/usr/bin/env python3
"""Validation-select and evaluate isolated targeted CREMA-D CBMs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import train_cbm


CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
TRIALS = [
    {"name": "atomic_i40_l7", "concepts": "atomic", "interpretability": 0.40, "lam": 0.0007},
    {"name": "grounded_i35_l3", "concepts": "grounded", "interpretability": 0.35, "lam": 0.0003},
    {"name": "grounded_i35_l7", "concepts": "grounded", "interpretability": 0.35, "lam": 0.0007},
    {"name": "grounded_i40_l3", "concepts": "grounded", "interpretability": 0.40, "lam": 0.0003},
    {"name": "grounded_i40_l7", "concepts": "grounded", "interpretability": 0.40, "lam": 0.0007},
    {"name": "grounded_i45_l7", "concepts": "grounded", "interpretability": 0.45, "lam": 0.0007},
    {"name": "grounded_i40_l15", "concepts": "grounded", "interpretability": 0.40, "lam": 0.0015},
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proj-steps", type=int, default=1000)
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def labels(split):
    rows = [json.loads(x) for x in Path("data/cremad/manifests/{}.jsonl".format(split)).read_text().splitlines()]
    return np.asarray([row["label_idx"] for row in rows], dtype=np.int64)


def model_args(args, concept_path, save_dir, interpretability, lam, test_split):
    return argparse.Namespace(
        dataset="cremad",
        concept_set=str(concept_path),
        backbone="ast_hf__Adam-ousse__ast-cremad-finetuned",
        clap_model="laion/clap-htsat-unfused",
        device=args.device,
        batch_size=16,
        saga_batch_size=256,
        proj_batch_size=50000,
        feature_layer="layer4",
        activation_dir=str(args.experiment_root / "activations"),
        save_dir=str(save_dir),
        clip_cutoff=0.25,
        concept_activation_cutoff=0.25,
        projection_threshold=None,
        proj_steps=args.proj_steps,
        interpretability_cutoff=interpretability,
        lam=lam,
        elastic_alpha=0.99,
        n_iters=args.n_iters,
        print=False,
        train_split="train",
        val_split="val",
        test_split=test_split,
        disable_test_eval=test_split is None,
        enforce_esc50_fold1_protocol=False,
        audioset_streaming=False,
        audioset_cache_dir=None,
        audioset_max_items=None,
        audioset_subset=None,
        audioset_train_subset=None,
        audioset_val_subset=None,
        audioset_test_subset=None,
        similarity_objective="cosine_cubed",
        seed=args.seed,
    )


def predict(model_dir, split, activation_dir):
    model_dir = Path(model_dir)
    feature_path = activation_dir / "{}_backbone_ast_hf__Adam-ousse__ast-cremad-finetuned_layer4.pt".format(split)
    x = torch.load(feature_path, map_location="cpu").float()
    wc = torch.load(model_dir / "W_c.pt", map_location="cpu").float()
    wg = torch.load(model_dir / "W_g.pt", map_location="cpu").float()
    bg = torch.load(model_dir / "b_g.pt", map_location="cpu").float()
    mean = torch.load(model_dir / "proj_mean.pt", map_location="cpu").float()
    std = torch.load(model_dir / "proj_std.pt", map_location="cpu").float()
    logits = ((x @ wc.T - mean) / std) @ wg.T + bg
    logits = logits.detach()
    return logits.argmax(dim=1).numpy(), logits.numpy()


def completed_models(save_dir):
    required = {"W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "metrics.txt", "concepts.txt"}
    if not save_dir.exists():
        return []
    return sorted(path for path in save_dir.iterdir() if path.is_dir() and required.issubset({child.name for child in path.iterdir()}))


def summarize(model_dir, split, activation_dir):
    y = labels(split)
    pred, _ = predict(model_dir, split, activation_dir)
    wg = torch.load(Path(model_dir) / "W_g.pt", map_location="cpu").float()
    nnz = int((wg.abs() > 1e-5).sum())
    concepts = [x for x in (Path(model_dir) / "concepts.txt").read_text().splitlines() if x]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(np.mean([np.mean(pred[y == k] == k) for k in range(len(CLASSES))])),
        "retained_concepts": len(concepts),
        "nonzero_weights": nnz,
        "total_weights": int(wg.numel()),
        "sparsity": 1.0 - nnz / float(wg.numel()),
        "avg_nonzero_per_class": nnz / float(len(CLASSES)),
    }


def final_details(model_dir, activation_dir):
    y = labels("test")
    pred, _ = predict(model_dir, "test", activation_dir)
    concepts = [x for x in (Path(model_dir) / "concepts.txt").read_text().splitlines() if x]
    wg = torch.load(Path(model_dir) / "W_g.pt", map_location="cpu").float()
    top = {}
    for class_index, name in enumerate(CLASSES):
        indices = torch.argsort(wg[class_index].abs(), descending=True)[:15].tolist()
        top[name] = [
            {
                "concept": concepts[i].removeprefix("a voice with "),
                "grounding_text": concepts[i],
                "weight": float(wg[class_index, i]),
            }
            for i in indices
        ]
    return {
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0),
        "top_absolute_weights": top,
    }


def main():
    args = parse_args()
    root = args.experiment_root
    summary_path = root / "training_summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite completed rerun: {}".format(summary_path))
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    concept_paths = {
        "atomic": root / "generation/concepts_atomic_filtered.txt",
        "grounded": root / "generation/concepts_grounded_filtered.txt",
    }
    for path in concept_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    tuning = []
    for trial in TRIALS:
        print("\n=== tuning {} ===".format(trial["name"]), flush=True)
        save_dir = models_root / "tuning" / trial["name"]
        save_dir.mkdir(parents=True, exist_ok=True)
        complete = completed_models(save_dir)
        if complete:
            model_dir = complete[-1]
            print("resume", model_dir, flush=True)
        else:
            model_dir = Path(train_cbm.train_cbm_and_save(model_args(
                args,
                concept_paths[trial["concepts"]],
                save_dir,
                trial["interpretability"],
                trial["lam"],
                test_split=None,
            )))
        row = dict(trial)
        row.update({"model_dir": str(model_dir), "validation": summarize(model_dir, "val", root / "activations")})
        tuning.append(row)
        (root / "tuning_progress.json").write_text(json.dumps(tuning, indent=2), encoding="utf-8")
        print("val accuracy={accuracy:.4f} macro-F1={macro_f1:.4f} retained={retained_concepts} sparsity={sparsity:.2%}".format(**row["validation"]))
        torch.cuda.empty_cache()

    selected = max(tuning, key=lambda row: (row["validation"]["macro_f1"], row["validation"]["accuracy"], row["validation"]["sparsity"]))
    print("\n=== selected {} by validation macro-F1 ===".format(selected["name"]), flush=True)
    final_save = models_root / "final" / selected["name"]
    final_save.mkdir(parents=True, exist_ok=True)
    final_model = Path(train_cbm.train_cbm_and_save(model_args(
        args,
        concept_paths[selected["concepts"]],
        final_save,
        selected["interpretability"],
        selected["lam"],
        test_split="test",
    )))
    test = summarize(final_model, "test", root / "activations")
    details = final_details(final_model, root / "activations")

    prior = {}
    comparison_path = Path("results/audio_concept_ablation/comparison.csv")
    if comparison_path.exists():
        with comparison_path.open() as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] == "cremad":
                    prior[row["model"]] = {"accuracy": float(row["accuracy"]), "macro_f1": float(row["macro_f1"])}
    payload = {
        "selection_rule": "maximum validation macro-F1; accuracy and sparsity break ties",
        "protocol": {
            "seed": args.seed,
            "activation_cutoff": 0.25,
            "projection_steps": args.proj_steps,
            "saga_iterations": args.n_iters,
            "elastic_alpha": 0.99,
            "similarity_objective": "cosine_cubed",
            "test_split_hidden_during_tuning": True,
        },
        "tuning": tuning,
        "selected": selected,
        "final_model_dir": str(final_model),
        "test": test,
        "test_details": details,
        "prior_results": prior,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("final test accuracy={accuracy:.4f} macro-F1={macro_f1:.4f} retained={retained_concepts} sparsity={sparsity:.2%}".format(**test))
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
