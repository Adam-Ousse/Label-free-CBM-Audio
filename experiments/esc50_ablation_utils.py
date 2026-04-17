from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "esc50_ablation_runs"


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(experiment_dir: Path, experiment_name: str) -> logging.Logger:
    ensure_dir(experiment_dir)
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        file_handler = logging.FileHandler(experiment_dir / "run.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    return logger


def log_run_start(logger: logging.Logger, index: int, total: int, run_name: str, details: str) -> None:
    logger.info("Run %d/%d | %s | %s", index, total, run_name, details)


def make_run_dir(results_root: Path, experiment_name: str, run_name: str) -> Path:
    experiment_root = ensure_dir(Path(results_root) / experiment_name)
    run_dir = experiment_root / f"{run_name}_{timestamp_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_train_namespace(
    base_args,
    run_dir: Path,
    *,
    concept_set: str,
    train_split: str,
    val_split: str,
    test_split: str | None,
    similarity_objective: str | None = None,
    prompt_template: str | None = None,
    max_concepts: int | None = None,
    projection_threshold: float | None = None,
    lam: float | None = None,
    elastic_alpha: float | None = None,
    run_name: str | None = None,
):
    return argparse.Namespace(
        dataset="esc50",
        concept_set=str(concept_set),
        backbone=base_args.backbone,
        clap_model=getattr(base_args, "clap_model", "laion/clap-htsat-unfused"),
        device=base_args.device,
        batch_size=getattr(base_args, "batch_size", 512),
        saga_batch_size=getattr(base_args, "saga_batch_size", 256),
        proj_batch_size=getattr(base_args, "proj_batch_size", 50000),
        feature_layer=getattr(base_args, "feature_layer", "layer4"),
        activation_dir=str(run_dir / "activations"),
        save_dir=str(run_dir / "models"),
        clip_cutoff=getattr(base_args, "clip_cutoff", 0.25),
        concept_activation_cutoff=getattr(base_args, "concept_activation_cutoff", None),
        proj_steps=getattr(base_args, "proj_steps", 1000),
        interpretability_cutoff=getattr(base_args, "interpretability_cutoff", 0.45),
        projection_threshold=projection_threshold,
        lam=getattr(base_args, "lam", 0.0007) if lam is None else lam,
        elastic_alpha=getattr(base_args, "elastic_alpha", 0.99) if elastic_alpha is None else elastic_alpha,
        n_iters=getattr(base_args, "n_iters", 1000),
        print=False,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        enforce_esc50_fold1_protocol=not bool(getattr(base_args, "allow_custom_splits", False)),
        audioset_streaming=False,
        audioset_cache_dir=None,
        audioset_max_items=None,
        similarity_objective=similarity_objective or getattr(base_args, "similarity_objective", "cosine_cubed"),
        prompt_template=prompt_template if prompt_template is not None else getattr(base_args, "prompt_template", None),
        max_concepts=max_concepts if max_concepts is not None else getattr(base_args, "max_concepts", None),
        results_dir=str(run_dir),
        run_name=run_name,
    )


def write_results_bundle(
    experiment_dir: Path,
    experiment_name: str,
    rows: list[dict],
    *,
    note: str | None = None,
    commands: list[str] | None = None,
):
    ensure_dir(experiment_dir)

    results_json = {
        "experiment": experiment_name,
        "note": note,
        "runs": rows,
    }
    with (experiment_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=True, default=str)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (experiment_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    best_row = None
    if rows:
        def _score(row):
            value = row.get("best_val_accuracy", row.get("val_accuracy", float("-inf")))
            try:
                return float(value)
            except Exception:
                return float("-inf")

        best_row = max(rows, key=_score)

    summary_lines = [
        f"Experiment: {experiment_name}",
        f"Runs: {len(rows)}",
    ]
    if note:
        summary_lines.append(f"Note: {note}")
    if best_row is not None:
        summary_lines.extend([
            f"Best run: {best_row.get('run_name', 'unknown')}",
            f"Best val accuracy: {best_row.get('best_val_accuracy', best_row.get('val_accuracy', 'n/a'))}",
            f"Test accuracy: {best_row.get('test_accuracy', 'n/a')}",
            f"Retained concepts: {best_row.get('retained_concepts', 'n/a')}",
            f"Avg nonzeros per class: {best_row.get('avg_nnz_per_class', 'n/a')}",
            f"Model dir: {best_row.get('model_dir', 'n/a')}",
        ])

    with (experiment_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    if commands is not None:
        with (experiment_dir / "commands.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(commands) + "\n")

    return results_json
