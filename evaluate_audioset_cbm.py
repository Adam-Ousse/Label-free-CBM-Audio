import argparse
import json

import torch

import cbm
import data_utils
from models.ast_backbone import build_ast_backbone
from models.ast_classifier import build_ast_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pretrained AST or CBM on AudioSet")
    parser.add_argument("--model_type", type=str, default="ast", choices=["ast", "cbm"])
    parser.add_argument("--model_name", type=str, default="ast_audioset", help="AST alias or HF model id")
    parser.add_argument("--cbm_dir", type=str, default=None, help="Path to trained CBM directory when model_type=cbm")

    parser.add_argument("--split", type=str, default="eval")
    parser.add_argument("--subset", type=str, default=None, help="AudioSet subset: balanced, unbalanced, or full")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audioset_streaming", action="store_true")
    parser.add_argument("--audioset_cache_dir", type=str, default=None)
    parser.add_argument("--audioset_max_items", type=int, default=None)

    parser.add_argument("--inspect_backbone", action="store_true", help="Run one batch through AST backbone and print feature shape")
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def _compute_multilabel_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = (probs >= float(threshold)).astype("int32")

    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    metrics = {}
    try:
        metrics["mAP_macro"] = float(average_precision_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["mAP_macro"] = None

    try:
        metrics["auc_roc_macro"] = float(roc_auc_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["auc_roc_macro"] = None

    try:
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ValueError:
        metrics["f1_micro"] = None
        metrics["f1_macro"] = None

    metrics["threshold"] = float(threshold)
    return metrics


def main():
    args = parse_args()
    device = args.device
    resolved_subset, resolved_split = data_utils.resolve_hf_audioset_subset_split(args.split, subset=args.subset)

    classes = data_utils.get_dataset_classes("audioset")
    loader = data_utils.get_audio_dataloader(
        dataset_name="audioset",
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        hf_streaming=args.audioset_streaming,
        hf_cache_dir=args.audioset_cache_dir,
        max_items=args.audioset_max_items,
        hf_subset=args.subset,
    )

    if args.model_type == "ast":
        model = build_ast_classifier(args.model_name, device)
        model.model.eval()
    else:
        if args.cbm_dir is None:
            raise ValueError("--cbm_dir is required when --model_type cbm")
        model = cbm.load_cbm(args.cbm_dir, device)
        model.eval()

    if args.inspect_backbone:
        backbone = build_ast_backbone(args.model_name, device)
        backbone.eval()
        first_batch = next(iter(loader))
        with torch.no_grad():
            feats = backbone(first_batch["audio"], sample_rates=first_batch["sr"])
        print("Backbone feature shape:", tuple(feats.shape))

    criterion = torch.nn.BCEWithLogitsLoss()
    logits_all = []
    targets_all = []

    loss_sum = 0.0
    n_items = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"]
            sr = batch["sr"]
            targets = batch["target"].float()

            if args.model_type == "ast":
                logits = model.predict_logits(audio, sample_rates=sr).detach().cpu()
            else:
                logits, _ = model(audio.to(device))
                logits = logits.detach().cpu()

            if logits.shape[1] != len(classes):
                raise ValueError(
                    "Class dimension mismatch: logits={} classes={}".format(
                        logits.shape[1], len(classes)
                    )
                )

            batch_loss = criterion(logits, targets).item()
            loss_sum += batch_loss * targets.shape[0]
            n_items += targets.shape[0]

            logits_all.append(logits)
            targets_all.append(targets)

    logits_all = torch.cat(logits_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    avg_loss = float(loss_sum / max(n_items, 1))
    metrics = {
        "model_type": args.model_type,
        "model_name": args.model_name if args.model_type == "ast" else args.cbm_dir,
        "subset": resolved_subset,
        "split": resolved_split,
        "num_samples": int(n_items),
        "bce_loss": avg_loss,
    }

    try:
        metrics.update(_compute_multilabel_metrics(logits_all, targets_all, threshold=args.threshold))
    except ImportError as exc:
        metrics["warning"] = "scikit-learn is required for mAP/AUC/F1 metrics: {}".format(exc)

    print("AudioSet evaluation summary")
    print("  subset: {}".format(metrics["subset"]))
    print("  split: {}".format(metrics["split"]))
    for key in ["num_samples", "bce_loss", "mAP_macro", "auc_roc_macro", "f1_micro", "f1_macro"]:
        if key in metrics:
            print("  {}: {}".format(key, metrics[key]))

    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("  wrote_metrics:", args.output_json)


if __name__ == "__main__":
    main()
