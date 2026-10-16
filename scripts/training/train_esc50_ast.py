import argparse
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification, AutoFeatureExtractor, get_linear_schedule_with_warmup

import data_utils
from scripts.training.ast_input_cache import CachedASTInputDataset, collate_cached_ast_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Train AST on ESC-50")
    parser.add_argument("--base_model_id", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--esc50_root", type=str, default="data/esc50/raw/ESC-50-master")
    parser.add_argument("--output_dir", type=str, default="saved_models/ast_esc50")

    parser.add_argument("--test_fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--val_fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all_folds", action="store_true")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lf-cbm-audio-esc50")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="ast-esc50")
    parser.add_argument("--manifest_root", type=str, default="data/esc50")
    parser.add_argument("--no_prepare_manifests", action="store_true")
    parser.add_argument("--ast-input-cache", type=str, default=None)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Warm-start from a saved AST model checkpoint; optimizer state is not restored.",
    )
    parser.add_argument(
        "--resume_epoch",
        type=int,
        default=0,
        help="Last completed absolute epoch in --resume_from (training resumes at the next epoch).",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_esc50_manifests(args):
    manifest_dir = os.path.join(args.manifest_root, "manifests")
    required = [
        os.path.join(manifest_dir, "fold{}_train.jsonl".format(fold))
        for fold in [1, 2, 3, 4, 5]
    ]
    required += [
        os.path.join(manifest_dir, "fold{}_val.jsonl".format(fold))
        for fold in [1, 2, 3, 4, 5]
    ]
    required += [
        os.path.join(manifest_dir, "fold{}_test.jsonl".format(fold))
        for fold in [1, 2, 3, 4, 5]
    ]

    missing = [p for p in required if not os.path.exists(p)]
    if len(missing) == 0:
        return

    if args.no_prepare_manifests:
        raise FileNotFoundError(
            "Missing ESC-50 manifests. Run data/prepare_esc50.py or remove --no_prepare_manifests. First missing: {}".format(
                missing[0]
            )
        )

    cmd = [
        sys.executable,
        "data/prepare_esc50.py",
        "--esc50_root",
        args.esc50_root,
        "--out_root",
        args.manifest_root,
        "--repo_root",
        ".",
    ]
    print("Preparing ESC-50 manifests via:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_label_maps_from_metadata():
    mappings = data_utils.get_audio_label_mappings("esc50")
    idx_to_label = mappings.get("idx_to_label", {})
    if len(idx_to_label) != 50:
        raise ValueError("Expected 50 labels in data/esc50/idx_to_label.json")
    id2label = {int(k): v for k, v in idx_to_label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def create_fold_loaders(args, test_fold, val_fold, device):
    manifest_dir = os.path.join(args.manifest_root, "manifests")
    train_manifest = os.path.join(manifest_dir, "fold{}_train.jsonl".format(test_fold))
    val_manifest = os.path.join(manifest_dir, "fold{}_val.jsonl".format(test_fold))
    test_manifest = os.path.join(manifest_dir, "fold{}_test.jsonl".format(test_fold))

    # override val/test split if user sets a custom val fold
    if val_fold != ((test_fold % 5) + 1):
        all_rows = []
        all_manifest = os.path.join(manifest_dir, "all.jsonl")
        with open(all_manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_rows.append(json.loads(line))
        train_rows = [r for r in all_rows if int(r["fold"]) not in {test_fold, val_fold}]
        val_rows = [r for r in all_rows if int(r["fold"]) == val_fold]
        test_rows = [r for r in all_rows if int(r["fold"]) == test_fold]

        tmp_dir = os.path.join(args.output_dir, "tmp_manifests_fold{}".format(test_fold))
        os.makedirs(tmp_dir, exist_ok=True)
        train_manifest = os.path.join(tmp_dir, "train.jsonl")
        val_manifest = os.path.join(tmp_dir, "val.jsonl")
        test_manifest = os.path.join(tmp_dir, "test.jsonl")
        for path, rows in [(train_manifest, train_rows), (val_manifest, val_rows), (test_manifest, test_rows)]:
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    if args.ast_input_cache:
        dataset_factory = lambda manifest: CachedASTInputDataset(args.ast_input_cache, manifest)
        collate_fn = collate_cached_ast_batch
    else:
        dataset_factory = lambda manifest: data_utils.get_audio_dataset("esc50", split="train", manifest_path=manifest)
        collate_fn = data_utils.collate_audio_batch

    train_ds = dataset_factory(train_manifest)
    val_ds = dataset_factory(val_manifest)
    test_ds = dataset_factory(test_manifest)

    common = {
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
        "collate_fn": collate_fn,
    }
    if args.ast_input_cache and args.num_workers > 0:
        common.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def batch_to_model_inputs(batch, feature_extractor, device):
    if "input_values" in batch:
        inputs = {"input_values": batch["input_values"].to(device, non_blocking=(device == "cuda"))}
        labels = batch["target"].to(device, non_blocking=(device == "cuda"))
        return inputs, labels

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
    inputs = {k: v.to(device, non_blocking=(device == "cuda")) for k, v in inputs.items()}
    labels = batch["target"].to(device, non_blocking=(device == "cuda"))
    return inputs, labels


def evaluate_epoch(model, loader, feature_extractor, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch_to_model_inputs(batch, feature_extractor, device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=1)

            bs = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * bs
            total_correct += int(torch.sum(preds == labels).detach().cpu())
            total_count += bs

    return total_loss / total_count, float(total_correct) / float(total_count)


def build_esc50_model(base_model_id, device, id2label, label2id):
    model = ASTForAudioClassification.from_pretrained(base_model_id).to(device)

    if not hasattr(model, "classifier") or not hasattr(model.classifier, "dense"):
        raise ValueError("Unexpected AST classifier structure; expected model.classifier.dense")

    old_out = int(model.classifier.dense.out_features)
    in_features = int(model.classifier.dense.in_features)

    if old_out != 50:
        # replace only the task head (audioset 527 -> esc50 50)
        model.classifier.dense = torch.nn.Linear(in_features, 50).to(device)
        print("reinitialized classifier head: {} -> 50 classes".format(old_out))
    else:
        print("classifier head already has 50 classes")

    model.config.num_labels = 50
    model.config.id2label = id2label
    model.config.label2id = label2id
    return model


def train_single_fold(args, id2label, label2id, test_fold, val_fold, device):
    train_loader, val_loader, test_loader = create_fold_loaders(args, test_fold, val_fold, device)

    print("\nfold setup")
    print("test fold:", test_fold)
    print("val fold:", val_fold)
    print("train samples:", len(train_loader.dataset))
    print("val samples:", len(val_loader.dataset))
    print("test samples:", len(test_loader.dataset))

    model_source = args.resume_from if args.resume_from is not None else args.base_model_id
    if args.resume_from is not None:
        if args.resume_epoch < 1 or args.resume_epoch >= args.epochs:
            raise ValueError(
                "--resume_epoch must be in [1, epochs-1], got {} for {} epochs".format(
                    args.resume_epoch, args.epochs
                )
            )
        if not os.path.isdir(args.resume_from):
            raise FileNotFoundError("Resume checkpoint directory not found: {}".format(args.resume_from))

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
    model = build_esc50_model(model_source, device, id2label, label2id)

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

        run_name = "{}-fold{}".format(args.run_name, test_fold)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "base_model_id": args.base_model_id,
                "test_fold": test_fold,
                "val_fold": val_fold,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "num_workers": args.num_workers,
                "seed": args.seed,
            },
        )

    fold_dir = os.path.join(args.output_dir, "fold{}".format(test_fold))
    os.makedirs(fold_dir, exist_ok=True)

    with open(os.path.join(fold_dir, "train_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    best_val_acc = -1.0
    best_epoch = -1
    start_epoch = int(args.resume_epoch)
    if args.resume_from is not None:
        resume_metrics_path = os.path.join(args.resume_from, "best_metrics.json")
        if not os.path.isfile(resume_metrics_path):
            raise FileNotFoundError(
                "Resume checkpoint is missing best_metrics.json: {}".format(resume_metrics_path)
            )
        with open(resume_metrics_path, "r", encoding="utf-8") as f:
            resume_metrics = json.load(f)
        best_epoch = int(resume_metrics["best_epoch"])
        best_val_acc = float(resume_metrics["best_val_acc"])
        if best_epoch != start_epoch:
            raise ValueError(
                "Resume epoch {} does not match checkpoint best_epoch {}".format(
                    start_epoch, best_epoch
                )
            )
        # The frozen checkpoint contains model weights but no optimizer or scheduler
        # state. Advance the fresh scheduler to the absolute epoch boundary so the
        # LR schedule remains aligned with the original 30-epoch protocol.
        for _ in range(start_epoch * max(1, len(train_loader))):
            scheduler.step()
        print(
            "warm-starting from {} at completed epoch {} (optimizer/scheduler state not restored)".format(
                args.resume_from, start_epoch
            )
        )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0

        for batch in train_loader:
            inputs, labels = batch_to_model_inputs(batch, feature_extractor, device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(outputs.logits, dim=1)
            bs = labels.shape[0]
            train_loss_sum += float(loss.detach().cpu()) * bs
            train_correct += int(torch.sum(preds == labels).detach().cpu())
            train_count += bs

        train_loss = train_loss_sum / train_count
        train_acc = float(train_correct) / float(train_count)

        val_loss, val_acc = evaluate_epoch(model, val_loader, feature_extractor, device)

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
            best_dir = os.path.join(fold_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            feature_extractor.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"best_epoch": best_epoch, "best_val_acc": best_val_acc}, f, indent=2)

    best_model = ASTForAudioClassification.from_pretrained(os.path.join(fold_dir, "best_model")).to(device)
    test_loss, test_acc = evaluate_epoch(best_model, test_loader, feature_extractor, device)

    print("best epoch:", best_epoch)
    print("best val acc:", round(best_val_acc, 4))
    print("test loss:", round(test_loss, 4))
    print("test acc:", round(test_acc, 4))

    summary = {
        "test_fold": test_fold,
        "val_fold": val_fold,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "resume_from": args.resume_from,
        "resume_epoch": start_epoch,
        "optimizer_state_resumed": False if args.resume_from is not None else None,
    }
    with open(os.path.join(fold_dir, "fold_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb_run.log({"test/loss": test_loss, "test/acc": test_acc})
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    args = parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    ensure_esc50_manifests(args)
    id2label, label2id = load_label_maps_from_metadata()

    if args.all_folds:
        summaries = []
        for test_fold in [1, 2, 3, 4, 5]:
            val_fold = ((test_fold) % 5) + 1
            summaries.append(
                train_single_fold(
                    args,
                    id2label,
                    label2id,
                    test_fold=test_fold,
                    val_fold=val_fold,
                    device=device,
                )
            )

        mean_test_acc = float(np.mean([s["test_acc"] for s in summaries]))
        with open(os.path.join(args.output_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"folds": summaries, "mean_test_acc": mean_test_acc}, f, indent=2)
        print("\ncv mean test acc:", round(mean_test_acc, 4))
    else:
        test_fold = args.test_fold
        val_fold = args.val_fold if args.val_fold is not None else (((test_fold) % 5) + 1)
        if val_fold == test_fold:
            raise ValueError("val_fold must be different from test_fold")

        train_single_fold(
            args,
            id2label,
            label2id,
            test_fold=test_fold,
            val_fold=val_fold,
            device=device,
        )
