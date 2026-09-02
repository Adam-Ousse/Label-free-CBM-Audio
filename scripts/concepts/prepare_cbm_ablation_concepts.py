#!/usr/bin/env python3
"""Build consistently filtered LF/Broad/Contrastive CBM ablation sets."""

import argparse
import json
from pathlib import Path

import concept_pipeline
import data_utils


DATASETS = ("esc50", "urbansound8k", "cremad")
VARIANTS = {
    "lf": ("lf",),
    "lf_broad": ("lf", "broad"),
    "lf_contrastive": ("lf", "contrastive"),
    "full": ("lf", "broad", "contrastive"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--concept-root", default="data/concept_sets")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--class-sim-cutoff", type=float, default=0.85)
    parser.add_argument("--other-sim-cutoff", type=float, default=0.90)
    return parser.parse_args()


def load_source(directory, source):
    path = directory / "concepts_{}.txt".format(source)
    if not path.is_file():
        raise FileNotFoundError("Missing source concept set: {}".format(path))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    args = parse_args()
    root = Path(args.concept_root)
    report = {
        "filtering": {
            "max_len": args.max_len,
            "class_sim_cutoff": args.class_sim_cutoff,
            "other_sim_cutoff": args.other_sim_cutoff,
        },
        "datasets": {},
    }

    for dataset in args.datasets:
        print("\n=== {} ===".format(dataset))
        directory = root / dataset
        classes = data_utils.get_dataset_classes(dataset)
        sources = {
            source: load_source(directory, source)
            for source in ("lf", "broad", "contrastive")
        }
        report["datasets"][dataset] = {}
        for variant, source_names in VARIANTS.items():
            candidates = []
            for source in source_names:
                candidates.extend(sources[source])
            candidates = concept_pipeline.dedupe_case_insensitive(candidates)
            print("[{}] {} raw unique candidates".format(variant, len(candidates)))
            concepts = concept_pipeline.filter_concepts(
                candidates,
                classes,
                max_len=args.max_len,
                class_sim_cutoff=args.class_sim_cutoff,
                other_sim_cutoff=args.other_sim_cutoff,
                device=args.device,
                print_prob=0.0,
            )
            concepts = concept_pipeline.dedupe_case_insensitive(concepts)
            output_path = directory / "concepts_ablation_{}.txt".format(variant)
            concept_pipeline.save_concept_text(str(output_path), concepts)
            report["datasets"][dataset][variant] = {
                "sources": list(source_names),
                "raw_unique": len(candidates),
                "text_filtered": len(concepts),
                "path": str(output_path),
            }
            print("[{}] retained {} -> {}".format(variant, len(concepts), output_path))

    report_path = root / "ablation_concept_summary.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print("\nWrote {}".format(report_path))


if __name__ == "__main__":
    main()
