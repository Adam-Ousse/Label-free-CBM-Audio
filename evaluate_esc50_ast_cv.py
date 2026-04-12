import argparse
import csv
import os
import wave

import numpy as np
import torch
import torch.nn.functional as F

from models.ast_classifier import build_ast_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ESC-50 AST model per fold")
    parser.add_argument(
        "--model_id",
        type=str,
        default="ast_esc50",
        help="AST classifier name or hf id alias",
    )
    parser.add_argument(
        "--esc50_root",
        type=str,
        default="data/esc50/raw/ESC-50-master",
        help="ESC-50 root with audio/ and meta/esc50.csv",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional debug cap")
    parser.add_argument("--esc10_only", action="store_true", help="Evaluate only ESC-10 subset rows")
    parser.add_argument("--expected_num_labels", type=int, default=50, help="Expected classifier output classes for ESC-50")
    return parser.parse_args()


def load_wav(audio_path):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV sample width: {} for {}".format(sample_width, audio_path))

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return torch.from_numpy(np.ascontiguousarray(audio)).float(), int(sample_rate)


def resample_if_needed(audio, source_sr, target_sr):
    if source_sr == target_sr:
        return audio

    new_len = int(round(audio.shape[-1] * float(target_sr) / float(source_sr)))
    if new_len <= 0:
        raise ValueError("Invalid resampled length")

    audio = F.interpolate(audio.unsqueeze(0).unsqueeze(0), size=new_len, mode="linear", align_corners=False)
    return audio.squeeze(0).squeeze(0)


def load_esc50_rows(esc50_root, esc10_only=False, max_samples=None):
    csv_path = os.path.join(esc50_root, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_root, "audio")

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Missing metadata: {}".format(csv_path))
    if not os.path.exists(audio_dir):
        raise FileNotFoundError("Missing audio directory: {}".format(audio_dir))

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if esc10_only and str(row.get("esc10", "")).lower() not in {"true", "1"}:
                continue

            audio_path = os.path.join(audio_dir, row["filename"])
            rows.append(
                {
                    "audio_path": audio_path,
                    "target": int(row["target"]),
                    "category": row["category"],
                    "fold": int(row["fold"]),
                    "filename": row["filename"],
                }
            )

    if max_samples is not None:
        rows = rows[:max_samples]

    if len(rows) == 0:
        raise ValueError("No rows found for evaluation")

    return rows


def iter_batches(rows, batch_size):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def _normalize_label(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def build_esc50_label_map(label2id):
    norm_map = {}
    for k, v in label2id.items():
        norm_map[_normalize_label(k)] = int(v)
    return norm_map


def _get_target_id(row, esc50_label_map, num_labels):
    category_key = _normalize_label(row["category"])
    if category_key in esc50_label_map:
        return int(esc50_label_map[category_key])

    raw_target = int(row["target"])
    if 0 <= raw_target < int(num_labels):
        return raw_target

    raise ValueError("Could not map ESC-50 row target for category '{}'".format(row["category"]))


def evaluate_per_fold(classifier, rows, batch_size):
    target_sr = int(classifier.default_sample_rate)

    fold_correct = {}
    fold_total = {}
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for batch_rows in iter_batches(rows, batch_size):
            waveforms = []
            target_ids = []
            folds = []

            for row in batch_rows:
                wav, sr = load_wav(row["audio_path"])
                wav = resample_if_needed(wav, sr, target_sr)
                waveforms.append(wav.numpy())
                target_ids.append(_get_target_id(row, classifier.esc50_label_map, classifier.num_labels))
                folds.append(int(row["fold"]))

            logits = classifier.predict_logits(
                waveforms,
                sample_rates=[target_sr for _ in range(len(waveforms))],
            )
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            for pred, target_id, fold in zip(preds, target_ids, folds):
                if fold not in fold_total:
                    fold_total[fold] = 0
                    fold_correct[fold] = 0
                fold_total[fold] += 1
                total_count += 1
                if int(pred) == int(target_id):
                    fold_correct[fold] += 1
                    total_correct += 1

    return fold_correct, fold_total, total_correct, total_count


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_esc50_rows(args.esc50_root, esc10_only=args.esc10_only, max_samples=args.max_samples)

    print("Model:", args.model_id)
    print("Device:", device)
    print("Samples:", len(rows))
    print("ESC-10 only:", args.esc10_only)

    classifier = build_ast_classifier(args.model_id, device)
    classifier.esc50_label_map = build_esc50_label_map(classifier.label2id)

    print("Resolved model id:", classifier.model_id)
    print("Model num_labels:", classifier.num_labels)

    if int(classifier.num_labels) != int(args.expected_num_labels):
        raise ValueError(
            "Model outputs {} classes, expected {} for ESC-50. "
            "This checkpoint is not an ESC-50 classifier head.".format(classifier.num_labels, args.expected_num_labels)
        )

    fold_correct, fold_total, total_correct, total_count = evaluate_per_fold(classifier, rows, args.batch_size)

    print("\nFold accuracies")
    for fold in sorted(fold_total.keys()):
        acc = float(fold_correct[fold]) / float(fold_total[fold])
        print("fold {}: {:.4f} ({}/{})".format(fold, acc, fold_correct[fold], fold_total[fold]))

    overall = float(total_correct) / float(total_count)
    print("\nOverall accuracy: {:.4f} ({}/{})".format(overall, total_correct, total_count))