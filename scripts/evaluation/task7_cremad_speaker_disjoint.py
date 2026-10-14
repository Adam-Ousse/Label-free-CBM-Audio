#!/usr/bin/env python3
"""Run the CREMA-D speaker-disjoint feature, CBM, and reporting stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import data_utils
import utils
from clap import core as clap_core
from scripts.evaluation import task3_baselines as t3


SPLITS = ("train", "val", "test")
CLASSES = ["anger", "disgust", "fear", "happy", "neutral", "sad"]
VOCAB_SOURCE = Path("release/google_drive_bundle/concept_sets/cremad/full.txt")
AST_BACKBONE_TAG = "ast_task7_speaker_disjoint"


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _manifest_paths(root: Path) -> dict[str, Path]:
    return {split: root / "manifests" / (split + ".jsonl") for split in SPLITS}


def _labels_and_speakers(manifest: Path) -> tuple[torch.Tensor, set[str]]:
    rows = _rows(manifest)
    labels = torch.tensor([int(row["label_idx"]) for row in rows], dtype=torch.long)
    actors = {str(row.get("actor_id", str(row["id"]).split("_", 1)[0])) for row in rows}
    return labels, actors


def _verify_manifests(root: Path) -> dict[str, Any]:
    paths = _manifest_paths(root)
    rows = {split: _rows(path) for split, path in paths.items()}
    ids = {split: [str(row["id"]) for row in values] for split, values in rows.items()}
    all_ids = [item for split in SPLITS for item in ids[split]]
    actors = {split: {str(row["id"]).split("_", 1)[0] for row in rows[split]} for split in SPLITS}
    if any(len(ids[split]) != len(set(ids[split])) for split in SPLITS):
        raise ValueError("Speaker-disjoint manifests contain duplicate IDs")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Speaker-disjoint manifests assign an ID more than once")
    if actors["train"] & actors["val"] or actors["train"] & actors["test"] or actors["val"] & actors["test"]:
        raise ValueError("Speaker-disjoint manifests have actor overlap")
    if set().union(*actors.values()) != {str(i) for i in range(1001, 1092)}:
        raise ValueError("Speaker-disjoint manifests do not cover the expected 91 actors")
    if set(all_ids) != {str(row["id"]) for row in _rows(root / "manifests" / "all.jsonl")}:
        raise ValueError("Speaker-disjoint all.jsonl disagrees with split manifests")
    result = {
        "manifest_root": str(root / "manifests"),
        "actor_counts": {split: len(actors[split]) for split in SPLITS},
        "clip_counts": {split: len(rows[split]) for split in SPLITS},
        "actor_intersections": {
            "train_val": sorted(actors["train"] & actors["val"]),
            "train_test": sorted(actors["train"] & actors["test"]),
            "val_test": sorted(actors["val"] & actors["test"]),
        },
        "class_counts": {
            split: dict(sorted({name: sum(str(row["label"]) == name for row in rows[split]) for name in CLASSES}.items()))
            for split in SPLITS
        },
        "unique_actors": len(set().union(*actors.values())),
        "all_ids_unique": len(all_ids) == len(set(all_ids)),
    }
    if any(result["actor_intersections"].values()):
        raise ValueError("Actor overlap verification failed")
    return result


def _environment(repo_root: Path) -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "numpy", "sklearn"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[name] = str(exc)
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    except Exception:
        gpu = []
    return {
        "python": platform.python_version(),
        "packages": versions,
        "gpu": gpu,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
    }


def prepare(repo_root: Path, root: Path) -> None:
    source = repo_root / "results/revision/t7_cremad/split_audit"
    manifest_root = root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    for split in (*SPLITS, "all"):
        shutil.copy2(source / "manifests" / (split + ".jsonl"), manifest_root / (split + ".jsonl"))
    for name in ("label_to_idx.json", "idx_to_label.json"):
        shutil.copy2(repo_root / "data/cremad" / name, manifest_root / name)
    verification = _verify_manifests(root)
    vocab = repo_root / VOCAB_SOURCE
    if not vocab.is_file():
        raise FileNotFoundError(vocab)
    vocab_path = root / "cbm" / "vocabulary" / "cremad_full.txt"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocab, vocab_path)
    _json_dump(root / "manifest_verification.json", verification)
    _json_dump(root / "cbm" / "vocabulary" / "provenance.json", {
        "source": str(VOCAB_SOURCE),
        "copied_path": str(vocab_path.relative_to(repo_root)),
        "nonempty_concepts": sum(bool(line.strip()) for line in vocab.read_text(encoding="utf-8").splitlines()),
        "sha256": _sha256(vocab),
        "provenance": "canonical targeted CREMA-D full vocabulary from the release bundle",
    })
    _json_dump(root / "run_config.json", {
        "protocol": "speaker-disjoint CREMA-D",
        "seed": 42,
        "actor_quota": {"train": 64, "val": 9, "test": 18},
        "validation_only_selection": True,
        "gender_balancing": "UNKNOWN; no demographic metadata was available",
        "environment": _environment(repo_root),
        "command": " ".join(sys.argv),
    })


def _feature_path(root: Path, split: str) -> Path:
    return root / "features" / (split + "_backbone_" + AST_BACKBONE_TAG + "_layer4.pt")


def build_features(repo_root: Path, root: Path, checkpoint: Path, device: str, batch_size: int) -> None:
    verification = _verify_manifests(root)
    paths = _manifest_paths(root)
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError("Missing selected AST checkpoint: {}".format(checkpoint))
    ast_model, _ = data_utils.get_target_model("ast_local__" + str(checkpoint.relative_to(repo_root)).replace("/", "__"), device)
    root.joinpath("features").mkdir(parents=True, exist_ok=True)
    ast_shapes = {}
    for split in SPLITS:
        target = _feature_path(root, split)
        dataset = data_utils.get_audio_dataset("cremad", split=split, manifest_path=str(paths[split]))
        utils.save_backbone_audio_features(ast_model, dataset, str(target), batch_size=batch_size, device=device)
        tensor = torch.load(target, map_location="cpu", weights_only=True).float()
        if tensor.shape[0] != len(dataset):
            raise ValueError("AST feature row mismatch for {}".format(split))
        ast_shapes[split] = list(tensor.shape)
    del ast_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vocab = root / "cbm/vocabulary/cremad_full.txt"
    clap_bundle = clap_core.load_clap_model(model_name="laion/clap-htsat-unfused", device=device)
    text_path = root / "features/full_clap_text.pt"
    concepts = [line.strip() for line in vocab.read_text(encoding="utf-8").splitlines() if line.strip()]
    utils.save_clap_text_features(clap_bundle, concepts, str(text_path), batch_size=64)
    clap_shapes = {"text": list(torch.load(text_path, map_location="cpu", weights_only=True).shape)}
    for split in SPLITS:
        audio_path = root / "features" / (split + "_clap_audio.pt")
        dataset = data_utils.get_audio_dataset("cremad", split=split, manifest_path=str(paths[split]))
        utils.save_clap_audio_features(clap_bundle, dataset, str(audio_path), batch_size=batch_size)
        audio = torch.load(audio_path, map_location="cpu", weights_only=True)
        if audio.shape[0] != len(dataset):
            raise ValueError("CLAP feature row mismatch for {}".format(split))
        clap_shapes[split] = list(audio.shape)
    del clap_bundle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_file = None
    if checkpoint.is_dir():
        for candidate in (checkpoint / "model.safetensors", checkpoint / "pytorch_model.bin"):
            if candidate.is_file():
                checkpoint_file = candidate
                break
    _json_dump(root / "features/metadata.json", {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint_file) if checkpoint_file is not None else None,
        "ast_shapes": ast_shapes,
        "clap_shapes": clap_shapes,
        "manifest_verification": verification,
        "device": device,
        "batch_size": batch_size,
        "canonical_clap_model": "laion/clap-htsat-unfused",
    })


def fit_cbm(repo_root: Path, root: Path, device: str) -> None:
    verification = _verify_manifests(root)
    manifest_paths = _manifest_paths(root)
    labels = tuple(_labels_and_speakers(manifest_paths[split])[0] for split in SPLITS)
    features = t3.SplitFeatures(*(torch.load(_feature_path(root, split), map_location="cpu", weights_only=True).float() for split in SPLITS))
    vocab = root / "cbm/vocabulary/cremad_full.txt"
    paths = {
        split: (_feature_path(root, split), root / "features" / (split + "_clap_audio.pt"), root / "features/full_clap_text.pt")
        for split in SPLITS
    }
    reference = t3._reference_stats(repo_root / "release/google_drive_bundle/checkpoints/cremad/full")
    args = {
        "device": device, "seed": 42, "concept_activation_cutoff": 0.25,
        "interpretability_cutoff": 0.45, "projection_steps": 1000,
        "projection_batch_size": 50000, "saga_batch_size": 256,
        "saga_step_size": 0.1, "saga_iters": 1000, "saga_epsilon": 1e-4,
        "saga_path_points": 25, "saga_group": True, "saga_include_zero": False,
    }
    out = root / "cbm"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    results = t3._run_concept_bottleneck(
        args, "cremad", {"name": "full"}, vocab, features, paths, out,
        reference["mean_nnz_per_class"], labels=labels,
    )
    selected = next(item for item in results if item["selection"] == "full")
    full_path = out / "full.pt"
    t3._save_selected_predictions(full_path, features.test, labels[2], out)
    payload = json.loads((out / "full.json").read_text(encoding="utf-8"))
    payload.update({"manifest_verification": verification, "runtime_seconds": time.time() - start})
    _json_dump(out / "full.json", payload)
    selected_payload = torch.load(full_path, map_location="cpu", weights_only=True)
    def logits(x: torch.Tensor) -> torch.Tensor:
        projected = x.float()
        if "W_c" in selected_payload:
            projected = torch.nn.functional.linear(projected, selected_payload["W_c"].float())
        if "proj_mean" in selected_payload:
            projected = (projected - selected_payload["proj_mean"].float()) / selected_payload["proj_std"].float().clamp_min(1e-6)
        return torch.nn.functional.linear(projected, selected_payload["W_g"].float(), selected_payload["b_g"].float())
    val_pred = logits(features.val).argmax(dim=1).numpy()
    val_true = labels[1].numpy()
    test_pred = logits(features.test).argmax(dim=1).numpy()
    test_true = labels[2].numpy()
    _json_dump(out / "selection_validation.json", {
        "selection_split": "val",
        "accuracy": float(np.mean(val_pred == val_true)),
        "macro_f1": float(f1_score(val_true, val_pred, average="macro", zero_division=0)),
        "n": int(len(val_true)),
    })
    details = {
        "n_test": int(len(test_true)),
        "unique_test_speakers": verification["actor_counts"]["test"],
        "classification_report": classification_report(test_true, test_pred, target_names=CLASSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(test_true, test_pred).tolist(),
        "accuracy": float(np.mean(test_pred == test_true)),
        "macro_f1": float(f1_score(test_true, test_pred, average="macro", zero_division=0)),
        "selection": selected,
    }
    _json_dump(out / "test_details.json", details)


def write_summary(repo_root: Path, root: Path) -> None:
    verification = _verify_manifests(root)
    ast_path = root / "ast/test_metrics.json"
    cbm_path = root / "cbm/full.json"
    cbm_details_path = root / "cbm/test_details.json"
    if not ast_path.exists() or not cbm_path.exists() or not cbm_details_path.exists():
        raise FileNotFoundError("Run AST evaluation and CBM fitting before summary")
    ast = json.loads(ast_path.read_text(encoding="utf-8"))
    cbm = json.loads(cbm_path.read_text(encoding="utf-8"))
    details = json.loads(cbm_details_path.read_text(encoding="utf-8"))
    summary = {
        "protocol": {
            "seed": 42, "actor_counts": verification["actor_counts"], "clip_counts": verification["clip_counts"],
            "actor_intersections": verification["actor_intersections"], "validation_only_selection": True,
        },
        "ast": ast,
        "cbm_targeted_full": {**cbm, "test_details": details},
        "historical_context": {
            "original_utterance_split_ast_accuracy": 0.7005,
            "original_utterance_split_targeted_full_cbm_accuracy": 0.7025,
            "original_utterance_split_targeted_full_cbm_macro_f1": 0.5158,
            "task4_five_seed_targeted_full_accuracy_mean": 0.6877,
            "task4_five_seed_targeted_full_accuracy_sd": 0.0061,
            "note": "These protocols are not directly identical; speaker-disjoint evaluation is stricter and the difference is not attributed solely to leakage.",
        },
        "environment": _environment(repo_root),
    }
    _json_dump(root / "summary.json", summary)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "accuracy", "macro_f1", "n_test", "unique_test_speakers"])
        writer.writeheader()
        writer.writerow({"model": "AST", "accuracy": ast["accuracy"], "macro_f1": ast["macro_f1"], "n_test": ast["num_samples"], "unique_test_speakers": verification["actor_counts"]["test"]})
        writer.writerow({"model": "alpha-CBM targeted full", "accuracy": details["accuracy"], "macro_f1": details["macro_f1"], "n_test": details["n_test"], "unique_test_speakers": details["unique_test_speakers"]})
    (root / "table.tex").write_text(
        "\\begin{tabular}{lrrr}\\toprule\nModel & Accuracy & Macro-F1 & Test clips \\\\\n"
        "\\midrule\nAST & {:.2f}\\% & {:.2f}\\% & {} \\\\\n"
        "\\alpha-CBM targeted full & {:.2f}\\% & {:.2f}\\% & {} \\\\\n"
        "\\bottomrule\\end{tabular}\n".format(
            ast["accuracy"] * 100, ast["macro_f1"] * 100, ast["num_samples"],
            details["accuracy"] * 100, details["macro_f1"] * 100, details["n_test"],
        ), encoding="utf-8",
    )
    lines = [
        "# Task 7 CREMA-D speaker-disjoint evaluation", "",
        "Protocol: deterministic actor-disjoint split, seed 42, with 64 train, 9 validation, and 18 test actors.",
        "No demographic metadata was available; gender balancing was not claimed or applied.", "",
        "| Split | Actors | Clips | Class counts |", "|---|---:|---:|---|",
    ]
    for split in SPLITS:
        counts = ", ".join("{}={}".format(k, v) for k, v in verification["class_counts"][split].items())
        lines.append("| {} | {} | {} | {} |".format(split, verification["actor_counts"][split], verification["clip_counts"][split], counts))
    lines += ["", "Actor intersections are all empty; all 91 actors and all 7,442 clips are assigned exactly once.", "",
              "## Test results", "", "| Model | Accuracy | Macro-F1 | Test clips | Test speakers |", "|---|---:|---:|---:|---:|",
              "| AST | {:.2f}% | {:.2f}% | {} | {} |".format(ast["accuracy"]*100, ast["macro_f1"]*100, ast["num_samples"], verification["actor_counts"]["test"]),
              "| α-CBM targeted full | {:.2f}% | {:.2f}% | {} | {} |".format(details["accuracy"]*100, details["macro_f1"]*100, details["n_test"], details["unique_test_speakers"]),
              "", "Checkpoint selection and CBM projection/head selection used train/validation only; the speaker-disjoint test split was evaluated after selection.",
              "", "The exact commands, checkpoint/vocabulary provenance, environment, and caveats are recorded in `run_config.json`, `ast/`, `features/metadata.json`, and `cbm/`.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--stage", choices=("prepare", "features", "cbm", "summary"), required=True)
    parser.add_argument("--root", type=Path, default=Path("results/revision/t7_cremad/speaker_disjoint"))
    parser.add_argument("--ast-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    root = (repo_root / args.root).resolve() if not args.root.is_absolute() else args.root.resolve()
    if args.stage == "prepare":
        prepare(repo_root, root)
    elif args.stage == "features":
        checkpoint = args.ast_checkpoint or (root / "ast/test/best_model")
        build_features(repo_root, root, checkpoint, args.device, args.batch_size)
    elif args.stage == "cbm":
        fit_cbm(repo_root, root, args.device)
    else:
        write_summary(repo_root, root)


if __name__ == "__main__":
    main()
