#!/usr/bin/env python3
"""Generate DeepSeek candidate concept sets for audio LF-CBMs.

With no ``--mode`` this retains the original DeepSeek LF-CBM generation and
filtering workflow. The explicit source modes add broad and group-wise
contrastive candidates while continuing to use the project's existing CLAP
text-filtering functions.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

import concept_pipeline
import data_utils
from concept_generation_deepseek import (
    DeepSeekGenerator,
    discover_confusion_groups,
    generate_broad_concepts,
    generate_contrastive_concepts,
    generate_dataset_concepts,
)


DEFAULT_DATASETS = ("esc50", "urbansound8k", "cremad")
PROMPT_TYPES = {
    "esc50": ("important", "superclass", "around"),
    "urbansound8k": ("important", "superclass", "around"),
    "cremad": ("important", "superclass"),
}
SOURCE_ORDER = ("lf", "broad", "contrastive")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and process DeepSeek concept sets for audio CBMs."
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        choices=DEFAULT_DATASETS,
        help="Run one dataset (convenient singular alias for --datasets).",
    )
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        help="Datasets to run (default: all three).",
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "lf", "broad", "contrastive", "all"),
        default="legacy",
        help=(
            "Candidate source to generate. 'legacy' preserves the original script "
            "behavior; 'all' unions LF, broad, and contrastive candidates."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "generate", "process"),
        default="all",
        help="Run generation, existing CLAP text filtering, or both.",
    )
    parser.add_argument("--model", default=None, help="DeepSeek model; defaults to DEEPSEEK_MODEL.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="DeepSeek OpenAI-compatible API base URL (default: https://api.deepseek.com).",
    )
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--group-trials", type=int, default=1)
    parser.add_argument("--broad-trials", type=int, default=1)
    parser.add_argument("--broad-size", type=int, default=80)
    parser.add_argument("--concepts-per-group", type=int, default=8)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--device", default="auto", help="CLAP filtering device: auto, cuda, or cpu.")
    parser.add_argument("--max-len", type=int, default=30, help="Maximum concept length in characters.")
    parser.add_argument("--class-sim-cutoff", type=float, default=0.85)
    parser.add_argument("--other-sim-cutoff", type=float, default=0.90)
    parser.add_argument("--print-prob", type=float, default=0.0)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Regenerate requested sources instead of resuming checkpoints.",
    )
    parser.add_argument(
        "--init-dir",
        default="data/concept_sets/deepseek_init",
        help="Directory for resumable DeepSeek JSON checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/concept_sets",
        help="Root output directory; source files are saved in <output-dir>/<dataset>.",
    )
    args = parser.parse_args(argv)
    if args.dataset:
        args.datasets = [args.dataset]
    elif not args.datasets:
        args.datasets = list(DEFAULT_DATASETS)
    return args


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def checkpoint_path(init_dir, dataset, prompt_type):
    """Keep the original LF checkpoint naming convention."""
    return Path(init_dir) / "deepseek_{}_{}.json".format(dataset, prompt_type)


def broad_checkpoint_path(init_dir):
    return Path(init_dir) / "deepseek_broad.json"


def group_checkpoint_path(init_dir, dataset):
    return Path(init_dir) / "deepseek_{}_contrastive_groups.json".format(dataset)


def contrastive_checkpoint_path(init_dir, dataset):
    return Path(init_dir) / "deepseek_{}_contrastive.json".format(dataset)


def mode_sources(mode):
    if mode == "all":
        return SOURCE_ORDER
    if mode in SOURCE_ORDER:
        return (mode,)
    return ("lf",)


def _upsert_record(records, concept, source, classes, **extra):
    concept = str(concept).strip()
    if not concept:
        return
    key = (source, concept.casefold())
    if key not in records:
        records[key] = {
            "concept": concept,
            "source": source,
            "classes": list(classes),
        }
    else:
        for class_name in classes:
            if class_name not in records[key]["classes"]:
                records[key]["classes"].append(class_name)

    for field, values in extra.items():
        if values is None:
            continue
        values = values if isinstance(values, list) else [values]
        records[key].setdefault(field, [])
        for value in values:
            if value not in records[key][field]:
                records[key][field].append(value)


def load_lf_records(dataset, init_dir, required=False):
    paths = [checkpoint_path(init_dir, dataset, kind) for kind in PROMPT_TYPES[dataset]]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        if required:
            raise FileNotFoundError("Missing LF checkpoints: " + ", ".join(missing))
        return []

    records = {}
    for prompt_type, path in zip(PROMPT_TYPES[dataset], paths):
        prompt_dict = concept_pipeline.load_json(str(path))
        for class_name, concepts in prompt_dict.items():
            for concept in concepts:
                _upsert_record(
                    records,
                    concept,
                    "lf",
                    [class_name],
                    prompt_types=prompt_type,
                )
    return list(records.values())


def load_broad_records(init_dir, required=False):
    path = broad_checkpoint_path(init_dir)
    if not path.exists():
        if required:
            raise FileNotFoundError("Missing broad checkpoint: {}".format(path))
        return []
    payload = concept_pipeline.load_json(str(path))
    concepts = payload.get("concepts", []) if isinstance(payload, dict) else payload
    return [
        {"concept": concept, "source": "broad", "classes": []}
        for concept in concept_pipeline.dedupe_case_insensitive(concepts)
    ]


def load_contrastive_records(dataset, init_dir, required=False):
    path = contrastive_checkpoint_path(init_dir, dataset)
    if not path.exists():
        if required:
            raise FileNotFoundError("Missing contrastive checkpoint: {}".format(path))
        return []
    payload = concept_pipeline.load_json(str(path))
    group_records = payload.get("groups", {}) if isinstance(payload, dict) else {}
    if not isinstance(group_records, dict):
        raise ValueError("Invalid contrastive checkpoint: {}".format(path))

    records = {}
    for group_record in group_records.values():
        classes = group_record.get("classes", [])
        for concept in group_record.get("concepts", []):
            _upsert_record(
                records,
                concept,
                "contrastive",
                classes,
                groups=[list(classes)],
            )
    return list(records.values())


def load_source_records(dataset, args, required_sources=()):
    required_sources = set(required_sources)
    return {
        "lf": load_lf_records(dataset, args.init_dir, "lf" in required_sources),
        "broad": load_broad_records(args.init_dir, "broad" in required_sources),
        "contrastive": load_contrastive_records(
            dataset, args.init_dir, "contrastive" in required_sources
        ),
    }


def flatten_records(records_by_source, sources=SOURCE_ORDER):
    concepts = []
    for source in sources:
        concepts.extend(record["concept"] for record in records_by_source[source])
    return concept_pipeline.dedupe_case_insensitive(concepts)


def save_candidate_outputs(dataset, args, required_sources=()):
    """Export every available source plus a flat union and provenance metadata."""
    records_by_source = load_source_records(dataset, args, required_sources)
    output_dir = Path(args.output_dir) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    available_sources = []
    for source in SOURCE_ORDER:
        concepts = flatten_records(records_by_source, (source,))
        counts[source] = len(concepts)
        if concepts:
            available_sources.append(source)
            concept_pipeline.save_concept_text(
                str(output_dir / "concepts_{}.txt".format(source)), concepts
            )

    all_concepts = flatten_records(records_by_source, available_sources)
    if not all_concepts:
        raise ValueError("No candidate concepts are available for {}".format(dataset))
    concept_pipeline.save_concept_text(str(output_dir / "concepts_all.txt"), all_concepts)

    metadata = []
    for source in SOURCE_ORDER:
        metadata.extend(records_by_source[source])
    with (output_dir / "concepts_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    groups_path = group_checkpoint_path(args.init_dir, dataset)
    if groups_path.exists():
        groups_payload = concept_pipeline.load_json(str(groups_path))
        with (output_dir / "contrastive_groups.json").open("w", encoding="utf-8") as handle:
            json.dump(groups_payload, handle, indent=2, ensure_ascii=False)

    summary = {
        "dataset": dataset,
        "mode": args.mode,
        "model": args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "temperature": args.temperature,
        "num_trials": args.num_trials,
        "group_trials": args.group_trials,
        "broad_trials": args.broad_trials,
        "broad_size": args.broad_size,
        "concepts_per_group": args.concepts_per_group,
        "max_group_size": args.max_group_size,
        "source_counts": counts,
        "all_unique_concepts": len(all_concepts),
        "available_sources": available_sources,
    }
    with (output_dir / "generation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("[{}] candidates {} -> {}".format(dataset, counts, output_dir))
    return records_by_source, summary


def process_dataset(dataset, args, device):
    """Original LF-only merge/filter path retained for legacy invocations."""
    paths = [
        checkpoint_path(args.init_dir, dataset, prompt_type)
        for prompt_type in PROMPT_TYPES[dataset]
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot process dataset because checkpoints are missing: " + ", ".join(missing)
        )

    prompt_dicts = [concept_pipeline.load_json(str(path)) for path in paths]
    classes = data_utils.get_dataset_classes(dataset)
    candidates = concept_pipeline.merge_prompt_dicts(prompt_dicts)
    print("[{}] raw unique candidates: {}".format(dataset, len(candidates)))

    concepts = concept_pipeline.filter_concepts(
        candidates,
        classes,
        max_len=args.max_len,
        class_sim_cutoff=args.class_sim_cutoff,
        other_sim_cutoff=args.other_sim_cutoff,
        device=device,
        print_prob=args.print_prob,
    )
    concepts = concept_pipeline.dedupe_case_insensitive(concepts)
    output_path = Path(args.output_dir) / "{}_filtered_deepseek.txt".format(dataset)
    concept_pipeline.save_concept_text(str(output_path), concepts)

    summary = {
        "dataset": dataset,
        "prompt_types": list(PROMPT_TYPES[dataset]),
        "raw_unique_concepts": len(candidates),
        "filtered_concepts": len(concepts),
        "model": args.model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "num_trials": args.num_trials,
        "max_len": args.max_len,
        "class_sim_cutoff": args.class_sim_cutoff,
        "other_sim_cutoff": args.other_sim_cutoff,
        "output_path": str(output_path),
    }
    summary_path = Path(args.output_dir) / "{}_deepseek_summary.json".format(dataset)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print("[{}] retained {} concepts -> {}".format(dataset, len(concepts), output_path))
    return summary


def process_candidate_sources(dataset, mode, args, device):
    sources = mode_sources(mode)
    records_by_source = load_source_records(dataset, args, required_sources=sources)
    candidates = flatten_records(records_by_source, sources)
    classes = data_utils.get_dataset_classes(dataset)
    print("[{}:{}] raw unique candidates: {}".format(dataset, mode, len(candidates)))
    concepts = concept_pipeline.filter_concepts(
        candidates,
        classes,
        max_len=args.max_len,
        class_sim_cutoff=args.class_sim_cutoff,
        other_sim_cutoff=args.other_sim_cutoff,
        device=device,
        print_prob=args.print_prob,
    )
    concepts = concept_pipeline.dedupe_case_insensitive(concepts)
    output_path = Path(args.output_dir) / dataset / "concepts_{}_filtered.txt".format(mode)
    concept_pipeline.save_concept_text(str(output_path), concepts)
    print("[{}:{}] retained {} concepts -> {}".format(dataset, mode, len(concepts), output_path))
    return {
        "dataset": dataset,
        "mode": mode,
        "raw_unique_concepts": len(candidates),
        "filtered_concepts": len(concepts),
        "output_path": str(output_path),
    }


def make_generator(args):
    kwargs = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    return DeepSeekGenerator(**kwargs)


def run_legacy(args):
    """Run the baseline script behavior unchanged when --mode is omitted."""
    device = resolve_device(args.device)
    generator = make_generator(args) if args.stage in ("all", "generate") else None
    summaries = []

    for dataset in args.datasets:
        print("\n=== {} ===".format(dataset))
        if generator is not None:
            classes = data_utils.get_dataset_classes(dataset)
            for prompt_type in PROMPT_TYPES[dataset]:
                path = checkpoint_path(args.init_dir, dataset, prompt_type)
                print("[{}] prompt family: {}".format(dataset, prompt_type))
                generate_dataset_concepts(
                    dataset=dataset,
                    classes=classes,
                    prompt_type=prompt_type,
                    generator=generator,
                    save_path=path,
                    num_trials=args.num_trials,
                    resume=not args.restart,
                    temperature=args.temperature,
                )

        if args.stage in ("all", "process"):
            summaries.append(process_dataset(dataset, args, device))
    return summaries


def run_extended(args):
    sources = mode_sources(args.mode)
    generator = make_generator(args) if args.stage in ("all", "generate") else None
    broad_concepts = None
    summaries = []

    for dataset in args.datasets:
        print("\n=== {}: {} ===".format(dataset, args.mode))
        classes = data_utils.get_dataset_classes(dataset)
        if generator is not None and "lf" in sources:
            for prompt_type in PROMPT_TYPES[dataset]:
                print("[{}] LF prompt family: {}".format(dataset, prompt_type))
                generate_dataset_concepts(
                    dataset=dataset,
                    classes=classes,
                    prompt_type=prompt_type,
                    generator=generator,
                    save_path=checkpoint_path(args.init_dir, dataset, prompt_type),
                    num_trials=args.num_trials,
                    resume=not args.restart,
                    temperature=args.temperature,
                )

        if generator is not None and "broad" in sources and broad_concepts is None:
            broad_concepts = generate_broad_concepts(
                generator=generator,
                save_path=broad_checkpoint_path(args.init_dir),
                num_trials=args.broad_trials,
                num_concepts=args.broad_size,
                resume=not args.restart,
                temperature=args.temperature,
            )

        if generator is not None and "contrastive" in sources:
            groups = discover_confusion_groups(
                dataset=dataset,
                classes=classes,
                generator=generator,
                save_path=group_checkpoint_path(args.init_dir, dataset),
                num_trials=args.group_trials,
                max_group_size=args.max_group_size,
                resume=not args.restart,
                temperature=min(args.temperature, 0.3),
            )
            generate_contrastive_concepts(
                dataset=dataset,
                groups=groups,
                generator=generator,
                save_path=contrastive_checkpoint_path(args.init_dir, dataset),
                num_trials=args.num_trials,
                concepts_per_group=args.concepts_per_group,
                resume=not args.restart,
                temperature=args.temperature,
                forbidden_classes=classes,
            )

        save_candidate_outputs(dataset, args, required_sources=sources)
        if args.stage in ("all", "process"):
            summaries.append(
                process_candidate_sources(
                    dataset, args.mode, args, resolve_device(args.device)
                )
            )
    return summaries


def main(argv=None):
    args = parse_args(argv)
    load_dotenv()
    summaries = run_legacy(args) if args.mode == "legacy" else run_extended(args)

    if summaries:
        print("\n=== completed ===")
        for summary in summaries:
            print(
                "{}: {} raw -> {} filtered".format(
                    summary["dataset"],
                    summary["raw_unique_concepts"],
                    summary["filtered_concepts"],
                )
            )


if __name__ == "__main__":
    main()
