#!/usr/bin/env python3
"""Compare dense and sparse ESC-50 final layers with the fixed LF-CBM pipeline."""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_cbm

from esc50_ablation_utils import (
    DEFAULT_RESULTS_ROOT,
    build_train_namespace,
    log_run_start,
    make_run_dir,
    setup_logger,
    write_results_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESC-50 sparse vs dense final-layer ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="ast_esc50")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--concept_set", type=str, default="data/concept_sets/esc50_filtered_qwen.txt")
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--test_fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--val_fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--allow_custom_splits", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_splits(args: argparse.Namespace):
    if args.allow_custom_splits:
        test_fold = args.test_fold
        val_fold = args.val_fold if args.val_fold is not None else ((test_fold % 5) + 1)
    else:
        if args.test_fold != 1:
            raise ValueError("Default sparse_vs_dense_final_layer protocol uses test_fold=1 unless --allow_custom_splits is set")
        if args.val_fold is not None and args.val_fold != 2:
            raise ValueError("Default sparse_vs_dense_final_layer protocol uses val_fold=2 unless --allow_custom_splits is set")
        test_fold = 1
        val_fold = 2
    train_split = f"fold{test_fold}_train"
    val_split = f"fold{val_fold}_val"
    test_split = f"fold{test_fold}_test"
    return train_split, val_split, test_split


def build_settings():
    return [
        {"run_name": "dense", "lam": 1e-6, "elastic_alpha": 0.0, "label": "dense"},
        {"run_name": "sparse_moderate", "lam": 7e-4, "elastic_alpha": 0.99, "label": "sparse_moderate"},
        {"run_name": "sparse_strong", "lam": 2e-3, "elastic_alpha": 0.99, "label": "sparse_strong"},
    ]


def main() -> None:
    args = parse_args()
    train_split, val_split, test_split = resolve_splits(args)
    settings = build_settings()
    if args.max_runs is not None:
        settings = settings[: max(0, args.max_runs)]

    experiment_name = "sparse_vs_dense"
    experiment_root = Path(args.results_root) / experiment_name
    logger = setup_logger(experiment_root, experiment_name)
    rows = []
    commands = []

    for index, setting in enumerate(tqdm(settings, desc=experiment_name, unit="run"), start=1):
        run_dir = make_run_dir(Path(args.results_root), experiment_name, setting["run_name"])
        train_args = build_train_namespace(
            args,
            run_dir,
            concept_set=args.concept_set,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            lam=setting["lam"],
            elastic_alpha=setting["elastic_alpha"],
            run_name=setting["run_name"],
        )
        log_run_start(logger, index, len(settings), setting["run_name"], f"lam={setting['lam']} alpha={setting['elastic_alpha']}")
        commands.append(f"{setting['run_name']}: lam={setting['lam']} alpha={setting['elastic_alpha']}")

        if args.dry_run:
            logger.info("[dry-run] %s -> %s", setting["run_name"], run_dir)
            continue

        logger.info("Starting %s in %s", setting["run_name"], run_dir)
        result = train_cbm.train_cbm_and_save(train_args)
        logger.info(
            "Finished %s | val=%.4f | test=%.4f | retained=%s | nnz/class=%.4f",
            setting["run_name"],
            float(result.get("best_val_accuracy", float("nan"))),
            float(result.get("test_accuracy", float("nan"))),
            result.get("retained_concepts"),
            float(result.get("avg_nnz_per_class", float("nan"))),
        )
        row = dict(result)
        row.update(
            {
                "experiment": experiment_name,
                "setting": setting["label"],
                "run_name": setting["run_name"],
                "lam": setting["lam"],
                "elastic_alpha": setting["elastic_alpha"],
                "command": f"train_cbm.train_cbm_and_save(...) -> {run_dir}",
                "output_dir": str(run_dir),
            }
        )
        rows.append(row)

    note = "Dense uses alpha=0.0 and a near-zero lambda; sparse settings vary lambda with alpha fixed at 0.99."
    write_results_bundle(experiment_root, experiment_name, rows, note=note, commands=commands)

    if rows:
        best = max(rows, key=lambda row: float(row.get("best_val_accuracy", float("-inf"))))
        logger.info("Best setting: %s (val=%s, test=%s)", best["run_name"], best.get("best_val_accuracy"), best.get("test_accuracy"))
    else:
        logger.info("Dry run completed; no training was executed.")


if __name__ == "__main__":
    main()
