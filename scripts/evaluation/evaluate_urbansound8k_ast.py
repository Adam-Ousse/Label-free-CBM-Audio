import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_utils
from models.ast_classifier import build_ast_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UrbanSound8K AST checkpoint")
    parser.add_argument(
        "--model_id",
        type=str,
        default="saved_models/ast_urbansound8k/fold10/best_model",
        help="AST classifier name, hf id alias, or local checkpoint directory",
    )
    parser.add_argument("--manifest_root", type=str, default="data/urbansound8k")
    parser.add_argument("--split", type=str, default="fold10_test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--expected_num_labels", type=int, default=10)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _manifest_path(manifest_root, split):
    return os.path.join(manifest_root, "manifests", "{}.jsonl".format(split))


def main():
    args = parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = _manifest_path(args.manifest_root, args.split)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Missing UrbanSound8K manifest for split '{}': {}".format(args.split, manifest_path))

    dataset = data_utils.get_audio_dataset("urbansound8k", split=args.split, manifest_path=manifest_path)
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: int(args.max_samples)]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )

    classifier = build_ast_classifier(args.model_id, device)
    if int(classifier.num_labels) != int(args.expected_num_labels):
        raise ValueError(
            "Model outputs {} classes, expected {} for UrbanSound8K".format(
                classifier.num_labels,
                args.expected_num_labels,
            )
        )

    classes = data_utils.get_dataset_classes("urbansound8k")
    class_correct = [0 for _ in range(len(classes))]
    class_total = [0 for _ in range(len(classes))]

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            logits = classifier.predict_logits(batch["audio"], sample_rates=batch["sr"]).detach().cpu()
            targets = batch["target"].long().cpu()
            preds = torch.argmax(logits, dim=1)

            loss = F.cross_entropy(logits, targets)
            bs = targets.shape[0]

            total_loss += float(loss.item()) * bs
            total_correct += int(torch.sum(preds == targets).item())
            total_count += bs

            for pred, target in zip(preds.tolist(), targets.tolist()):
                class_total[int(target)] += 1
                if int(pred) == int(target):
                    class_correct[int(target)] += 1

    avg_loss = total_loss / max(total_count, 1)
    accuracy = float(total_correct) / float(max(total_count, 1))

    print("UrbanSound8K evaluation")
    print("  model:", args.model_id)
    print("  split:", args.split)
    print("  samples:", total_count)
    print("  loss:", round(avg_loss, 6))
    print("  acc:", round(accuracy, 6))

    print("\nPer-class accuracy")
    per_class = []
    for idx, name in enumerate(classes):
        if class_total[idx] == 0:
            class_acc = None
            display = "n/a"
        else:
            class_acc = float(class_correct[idx]) / float(class_total[idx])
            display = "{:.4f}".format(class_acc)
        per_class.append(
            {
                "class_idx": idx,
                "class_name": name,
                "correct": class_correct[idx],
                "total": class_total[idx],
                "accuracy": class_acc,
            }
        )
        print("  {}: {} ({}/{})".format(name, display, class_correct[idx], class_total[idx]))

    metrics = {
        "model": args.model_id,
        "split": args.split,
        "num_samples": total_count,
        "loss": avg_loss,
        "accuracy": accuracy,
        "per_class": per_class,
    }

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("\nWrote metrics:", args.output_json)


if __name__ == "__main__":
    main()
