#!/usr/bin/env python3
"""Evaluate the three fine-tuned AST baselines on their held-out splits."""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

import data_utils
from models.ast_classifier import build_ast_classifier


EXPERIMENTS = {
    "esc50": {
        "model": "Adam-ousse/ast-esc50-finetuned-fold1",
        "split": "fold1_test",
    },
    "urbansound8k": {
        "model": "Adam-ousse/ast-urbansound8k-finetuned-fold10",
        "split": "fold10_test",
    },
    "cremad": {
        "model": "Adam-ousse/ast-cremad-finetuned",
        "split": "test",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/audio_concept_ablation/baselines")
    return parser.parse_args()


def _config_labels(classifier):
    labels = []
    for index in range(classifier.num_labels):
        label = classifier.id2label.get(index, classifier.id2label.get(str(index)))
        labels.append(str(label) if label is not None else None)
    return labels


def evaluate_dataset(dataset_name, model_id, split, args):
    classes = data_utils.get_dataset_classes(dataset_name)
    dataset = data_utils.get_audio_dataset(dataset_name, split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )

    classifier = build_ast_classifier(model_id, args.device)
    classifier.model.eval()
    if classifier.num_labels != len(classes):
        raise ValueError(
            "{} checkpoint has {} labels, expected {}".format(
                dataset_name, classifier.num_labels, len(classes)
            )
        )
    config_labels = _config_labels(classifier)
    if config_labels != classes:
        raise ValueError(
            "{} checkpoint label order differs from dataset classes: {} vs {}".format(
                dataset_name, config_labels, classes
            )
        )

    logits_all = []
    targets_all = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            logits = classifier.predict_logits(
                batch["audio"], sample_rates=batch["sr"]
            ).detach().cpu()
            targets = batch["target"].long().cpu()
            loss_sum += float(F.cross_entropy(logits, targets).item()) * len(targets)
            count += len(targets)
            logits_all.append(logits)
            targets_all.append(targets)

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all).numpy()
    predictions = torch.argmax(logits, dim=1).numpy()
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=list(range(len(classes))),
        zero_division=0,
    )

    metrics = {
        "dataset": dataset_name,
        "model_type": "fine_tuned_ast",
        "model": model_id,
        "split": split,
        "num_samples": int(count),
        "loss": float(loss_sum / max(count, 1)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class": [
            {
                "class_idx": index,
                "class_name": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_class_f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(classes)
        ],
    }
    return metrics, classifier


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for dataset_name in args.datasets:
        spec = EXPERIMENTS[dataset_name]
        print("\n=== {} baseline ===".format(dataset_name))
        metrics, classifier = evaluate_dataset(
            dataset_name, spec["model"], spec["split"], args
        )
        results.append(metrics)
        with (output_dir / "{}.json".format(dataset_name)).open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metrics, handle, indent=2)
        print(
            "samples={} loss={:.4f} accuracy={:.4f} macro-F1={:.4f} balanced-acc={:.4f}".format(
                metrics["num_samples"],
                metrics["loss"],
                metrics["accuracy"],
                metrics["f1_macro"],
                metrics["balanced_accuracy"],
            )
        )
        del classifier
        torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "dataset",
            "model",
            "split",
            "num_samples",
            "loss",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "balanced_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})
    print("\nWrote baseline results to {}".format(output_dir))


if __name__ == "__main__":
    main()
