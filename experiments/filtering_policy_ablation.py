#!/usr/bin/env python3
"""Compare concept filtering policies by regenerating ESC-50 concept lists and training on each one."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import concept_pipeline
import conceptset_utils
import data_utils
import train_cbm

from esc50_ablation_utils import (
    DEFAULT_RESULTS_ROOT,
    build_train_namespace,
    ensure_dir,
    log_run_start,
    make_run_dir,
    setup_logger,
    write_results_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONCEPT_JSON_DIR = REPO_ROOT / "data" / "concept_sets" / "qwen_init"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESC-50 filtering-policy ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="ast_esc50")
    parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--results_root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--test_fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--val_fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--allow_custom_splits", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def resolve_splits(args: argparse.Namespace):
    if args.allow_custom_splits:
        test_fold = args.test_fold
        val_fold = args.val_fold if args.val_fold is not None else ((test_fold % 5) + 1)
    else:
        if args.test_fold != 1:
            raise ValueError("Default filtering-policy sweep uses test_fold=1 unless --allow_custom_splits is set")
        if args.val_fold is not None and args.val_fold != 2:
            raise ValueError("Default filtering-policy sweep uses val_fold=2 unless --allow_custom_splits is set")
        test_fold = 1
        val_fold = 2
    return f"fold{test_fold}_train", f"fold{val_fold}_val", f"fold{test_fold}_test"


def require_inputs() -> None:
    required = [
        CONCEPT_JSON_DIR / "qwen_esc50_important.json",
        CONCEPT_JSON_DIR / "qwen_esc50_superclass.json",
        CONCEPT_JSON_DIR / "qwen_esc50_around.json",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing concept source file: {path}")


def build_settings():
    return [
        {"run_name": "full", "policy": "full", "class_filter": True, "redundancy_filter": True},
        {"run_name": "no_class_similarity", "policy": "no_class_similarity", "class_filter": False, "redundancy_filter": True},
        {"run_name": "no_redundancy", "policy": "no_redundancy", "class_filter": True, "redundancy_filter": False},
    ]


def generate_concepts(policy: dict, run_dir: Path, device: str) -> tuple[Path, int]:
    ensure_dir(run_dir)
    important = concept_pipeline.load_json(CONCEPT_JSON_DIR / "qwen_esc50_important.json")
    superclass = concept_pipeline.load_json(CONCEPT_JSON_DIR / "qwen_esc50_superclass.json")
    around = concept_pipeline.load_json(CONCEPT_JSON_DIR / "qwen_esc50_around.json")

    concepts = concept_pipeline.merge_prompt_dicts([important, superclass, around])
    concepts = conceptset_utils.remove_too_long(concepts, max_len=30, print_prob=0)
    classes = data_utils.get_dataset_classes("esc50")

    if policy["class_filter"]:
        concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, 0.85, device=device, print_prob=0)
    if policy["redundancy_filter"]:
        concepts = conceptset_utils.filter_too_similar(concepts, 0.9, device=device, print_prob=0)

    concepts = concept_pipeline.dedupe_case_insensitive(concepts)
    if not concepts:
        raise ValueError(f"Policy {policy['policy']} produced an empty concept list")

    out_path = run_dir / "generated_concepts.txt"
    concept_pipeline.save_concept_text(out_path, concepts)
    return out_path, len(concepts)


def main() -> None:
    args = parse_args()
    require_inputs()
    train_split, val_split, test_split = resolve_splits(args)
    settings = build_settings()
    if args.max_runs is not None:
        settings = settings[: max(0, args.max_runs)]

    experiment_name = "filtering_policy"
    experiment_root = Path(args.results_root) / experiment_name
    logger = setup_logger(experiment_root, experiment_name)
    rows = []
    commands = []

    for index, setting in enumerate(tqdm(settings, desc=experiment_name, unit="run"), start=1):
        run_dir = make_run_dir(Path(args.results_root), experiment_name, setting["run_name"])
        concept_file, generated_count = generate_concepts(setting, run_dir, args.device)
        train_args = build_train_namespace(
            args,
            run_dir,
            concept_set=str(concept_file),
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            run_name=setting["run_name"],
        )
        log_run_start(
            logger,
            index,
            len(settings),
            setting["run_name"],
            f"class_filter={setting['class_filter']} redundancy_filter={setting['redundancy_filter']}",
        )
        commands.append(
            f"{setting['run_name']}: class_filter={setting['class_filter']} redundancy_filter={setting['redundancy_filter']}"
        )

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
                "setting": setting["run_name"],
                "run_name": setting["run_name"],
                "policy": setting["policy"],
                "class_filter": setting["class_filter"],
                "redundancy_filter": setting["redundancy_filter"],
                "generated_concepts": generated_count,
                "generated_concept_file": str(concept_file),
                "command": f"train_cbm.train_cbm_and_save(...) -> {run_dir}",
                "output_dir": str(run_dir),
            }
        )
        rows.append(row)

    note = (
        "Full uses the current prompt-json pipeline; the other policies drop either the class-similarity filter "
        "or the redundancy filter before training."
    )
    write_results_bundle(experiment_root, experiment_name, rows, note=note, commands=commands)

    if rows:
        best = max(rows, key=lambda row: float(row.get("best_val_accuracy", float("-inf"))))
        logger.info("Best setting: %s (val=%s, test=%s)", best["run_name"], best.get("best_val_accuracy"), best.get("test_accuracy"))
    else:
        logger.info("Dry run completed; no training was executed.")


if __name__ == "__main__":
    main()
