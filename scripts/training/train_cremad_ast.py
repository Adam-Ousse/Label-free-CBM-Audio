import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification, AutoFeatureExtractor, get_linear_schedule_with_warmup

import data_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train AST on CREMA-D")
    parser.add_argument("--base_model_id", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--cremad_root", type=str, default="data/cremad/raw")
    parser.add_argument("--manifest_root", type=str, default="data/cremad")
    parser.add_argument("--output_dir", type=str, default="saved_models/ast_cremad")

    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lf-cbm-audio-cremad")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="ast-cremad")
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--no_prepare_manifests", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_path(manifest_root, split):
    return os.path.join(manifest_root, "manifests", "{}.jsonl".format(split))


def ensure_cremad_manifests(args):
    required = [
        _manifest_path(args.manifest_root, args.train_split),
        _manifest_path(args.manifest_root, args.val_split),
        _manifest_path(args.manifest_root, args.test_split),
        os.path.join(args.manifest_root, "label_to_idx.json"),
        os.path.join(args.manifest_root, "idx_to_label.json"),
    ]

    missing = [path for path in required if not os.path.exists(path)]
    if len(missing) == 0:
        return

    if args.no_prepare_manifests:
        raise FileNotFoundError(
            "Missing CREMA-D metadata artifacts. Run data/prepare_cremad.py or remove --no_prepare_manifests. First missing: {}".format(
                missing[0]
            )
        )

    cmd = [
        sys.executable,
        "data/prepare_cremad.py",
        "--cremad_root",
        args.cremad_root,
        "--out_root",
        args.manifest_root,
        "--repo_root",
        ".",
        "--val_fraction",
        str(args.val_fraction),
        "--split_seed",
        str(args.split_seed),
    ]
    print("Preparing CREMA-D manifests via:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_label_maps_from_metadata():
    mappings = data_utils.get_audio_label_mappings("cremad")
    idx_to_label = mappings.get("idx_to_label", {})
    if len(idx_to_label) != 6:
        raise ValueError("Expected 6 labels in data/cremad/idx_to_label.json")
    id2label = {int(k): v for k, v in idx_to_label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def create_loaders(args, device):
    train_manifest = _manifest_path(args.manifest_root, args.train_split)
    val_manifest = _manifest_path(args.manifest_root, args.val_split)
    test_manifest = _manifest_path(args.manifest_root, args.test_split)

    train_ds = data_utils.get_audio_dataset("cremad", split="train", manifest_path=train_manifest)
    val_ds = data_utils.get_audio_dataset("cremad", split="val", manifest_path=val_manifest)
    test_ds = data_utils.get_audio_dataset("cremad", split="test", manifest_path=test_manifest)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=data_utils.collate_audio_batch,
    )
    return train_loader, val_loader, test_loader


def _compute_class_weights(train_manifest_path, num_labels, device):
    counts = torch.zeros(num_labels, dtype=torch.float32)

    with open(train_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = int(row["label_idx"])
            if not (0 <= idx < num_labels):
                raise ValueError("label_idx out of range in {}: {}".format(train_manifest_path, idx))
            counts[idx] += 1.0

    if torch.any(counts <= 0):
        raise ValueError("Cannot compute class weights with missing classes in train manifest: {}".format(counts.tolist()))

    weights = counts.sum() / (counts * float(num_labels))
    weights = weights / weights.mean()
    return weights.to(device), counts


def batch_to_model_inputs(batch, feature_extractor, device):
    audio = batch["audio"]
    if audio.dim() == 3 and audio.shape[1] == 1:
        audio = audio.squeeze(1)
    elif audio.dim() != 2:
        raise ValueError("Unexpected audio batch shape: {}".format(tuple(audio.shape)))

    waveforms = [audio[i].cpu().numpy() for i in range(audio.shape[0])]
    sample_rates = batch["sr"]
    if isinstance(sample_rates, torch.Tensor):
        sample_rates = sample_rates.cpu().tolist()

    unique_sr = sorted(set(int(sr) for sr in sample_rates))
    if len(unique_sr) != 1:
        raise ValueError("Mixed sample rates in batch: {}".format(unique_sr))

    inputs = feature_extractor(
        waveforms,
        sampling_rate=int(unique_sr[0]),
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = batch["target"].to(device)
    return inputs, labels


def evaluate_epoch(model, loader, feature_extractor, device, loss_weights=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch_to_model_inputs(batch, feature_extractor, device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=loss_weights)
            preds = torch.argmax(logits, dim=1)

            bs = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_correct += int(torch.sum(preds == labels).detach().cpu())
            total_count += bs

    return total_loss / total_count, float(total_correct) / float(total_count)


def build_model(base_model_id, device, id2label, label2id):
    model = ASTForAudioClassification.from_pretrained(base_model_id).to(device)

    if not hasattr(model, "classifier") or not hasattr(model.classifier, "dense"):
        raise ValueError("Unexpected AST classifier structure; expected model.classifier.dense")

    num_labels = len(id2label)
    old_out = int(model.classifier.dense.out_features)
    in_features = int(model.classifier.dense.in_features)

    if old_out != num_labels:
        model.classifier.dense = torch.nn.Linear(in_features, num_labels).to(device)
        print("reinitialized classifier head: {} -> {} classes".format(old_out, num_labels))
    else:
        print("classifier head already has {} classes".format(num_labels))

    model.config.num_labels = num_labels
    model.config.id2label = id2label
    model.config.label2id = label2id
    return model


def _split_tag(test_split):
    if "_" in test_split:
        return test_split.split("_")[0]
    return test_split


def main():
    args = parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    ensure_cremad_manifests(args)
    id2label, label2id = load_label_maps_from_metadata()
    train_loader, val_loader, test_loader = create_loaders(args, device)

    print("split setup")
    print("train split:", args.train_split, "samples:", len(train_loader.dataset))
    print("val split:", args.val_split, "samples:", len(val_loader.dataset))
    print("test split:", args.test_split, "samples:", len(test_loader.dataset))

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model_id)
    model = build_model(args.base_model_id, device, id2label, label2id)

    train_loss_weights = None
    if not args.disable_class_weights:
        train_manifest = _manifest_path(args.manifest_root, args.train_split)
        train_loss_weights, class_counts = _compute_class_weights(
            train_manifest_path=train_manifest,
            num_labels=len(id2label),
            device=device,
        )
        print("using class-weighted training loss")
        for idx in sorted(id2label.keys()):
            print(
                "  class {} ({}) count={} weight={:.4f}".format(
                    idx,
                    id2label[idx],
                    int(class_counts[idx].item()),
                    float(train_loss_weights[idx].item()),
                )
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(round(float(total_steps) * float(args.warmup_ratio)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_wandb = bool(args.use_wandb)
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config={
                "base_model_id": args.base_model_id,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "test_split": args.test_split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "class_weighted_loss": (not args.disable_class_weights),
            },
        )

    split_dir = os.path.join(args.output_dir, _split_tag(args.test_split))
    os.makedirs(split_dir, exist_ok=True)

    with open(os.path.join(split_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch in train_loader:
            inputs, labels = batch_to_model_inputs(batch, feature_extractor, device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=train_loss_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)
            bs = labels.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_correct += int(torch.sum(preds == labels).detach().cpu())
            train_count += bs

        train_loss = train_loss_sum / train_count
        train_acc = float(train_correct) / float(train_count)

        val_loss, val_acc = evaluate_epoch(model, val_loader, feature_extractor, device, loss_weights=None)

        print(
            "epoch {}/{} - train_loss {:.4f} train_acc {:.4f} val_loss {:.4f} val_acc {:.4f}".format(
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )
        )

        if use_wandb:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_dir = os.path.join(split_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            feature_extractor.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best_epoch": best_epoch, "best_val_acc": best_val_acc}, f, indent=2)

    best_model = ASTForAudioClassification.from_pretrained(os.path.join(split_dir, "best_model")).to(device)
    test_loss, test_acc = evaluate_epoch(best_model, test_loader, feature_extractor, device, loss_weights=None)

    print("best epoch:", best_epoch)
    print("best val acc:", round(best_val_acc, 4))
    print("test loss:", round(test_loss, 4))
    print("test acc:", round(test_acc, 4))

    summary = {
        "train_split": args.train_split,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "class_weighted_loss": (not args.disable_class_weights),
    }
    with open(os.path.join(split_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb_run.finish()


if __name__ == "__main__":
    main()
