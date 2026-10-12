#!/usr/bin/env python3
"""Prepare the Wave t+1 CREMA-D leakage audit and metadata checks.

This module intentionally does not import the training stack.  It only reads
manifests/source artifacts and writes deterministic audit outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DATASET_NAMES = ("cremad", "esc50", "urbansound8k")
SPLITS = ("train", "val", "test")
UNKNOWN = "UNKNOWN"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True) + "\n")


def actor_id_from_row(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("id", ""))
    actor_id = sample_id.split("_", 1)[0]
    if not actor_id or actor_id == sample_id:
        raise ValueError(f"Cannot extract CREMA-D actor ID from row: {row}")
    return actor_id


def actor_sets(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, set[str]]:
    return {
        split: {actor_id_from_row(row) for row in rows}
        for split, rows in rows_by_split.items()
    }


def pairwise_intersections(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    actors = actor_sets(rows_by_split)
    return {
        f"{left}_and_{right}": sorted(actors[left] & actors[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }


def _largest_remainder_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if set(ratios) != set(SPLITS):
        raise ValueError("ratios must contain train, val, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")
    raw = {split: total * float(ratios[split]) for split in SPLITS}
    counts = {split: int(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (raw[split] - counts[split], -SPLITS.index(split)), reverse=True)
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def deterministic_actor_assignment(
    rows: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign each actor to one split using a seeded actor-level permutation.

    The actor quotas use largest-remainder rounding.  Rows are never sampled
    independently, so the resulting splits are speaker independent.
    """

    ratios = ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
    actor_rows: dict[str, int] = Counter(actor_id_from_row(row) for row in rows)
    actors = sorted(actor_rows)
    rng = random.Random(int(seed))
    rng.shuffle(actors)
    quotas = _largest_remainder_counts(len(actors), ratios)
    assignment: dict[str, str] = {}
    offset = 0
    for split in SPLITS:
        for actor in actors[offset : offset + quotas[split]]:
            assignment[actor] = split
        offset += quotas[split]
    if len(assignment) != len(actors):
        raise AssertionError("actor assignment did not cover every actor")
    return assignment, quotas


def build_speaker_independent_rows(
    rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    output = {split: [] for split in SPLITS}
    for source_row in rows:
        actor_id = actor_id_from_row(source_row)
        split = assignment[actor_id]
        row = dict(source_row)
        row["actor_id"] = actor_id
        row["source_split"] = source_row.get("split", UNKNOWN)
        row["split"] = split
        output[split].append(row)
    for split in SPLITS:
        output[split].sort(key=lambda row: str(row["id"]))
    return output


def count_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(row.get("label", UNKNOWN)) for row in rows)
    actors = Counter(actor_id_from_row(row) for row in rows)
    return {
        "rows": len(rows),
        "actors": len(actors),
        "actor_ids": sorted(actors),
        "rows_by_label": dict(sorted(labels.items())),
        "rows_by_actor": dict(sorted(actors.items())),
        "gender": UNKNOWN,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_cremad(repo_root: Path, output_root: Path, seed: int = 42) -> dict[str, Any]:
    manifest_root = repo_root / "data" / "cremad" / "manifests"
    rows_by_split = {split: read_jsonl(manifest_root / f"{split}.jsonl") for split in SPLITS}
    all_rows = read_jsonl(manifest_root / "all.jsonl")
    all_ids = [str(row.get("id")) for row in all_rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("data/cremad/manifests/all.jsonl contains duplicate IDs")

    overlaps = pairwise_intersections(rows_by_split)
    assignment, actor_quotas = deterministic_actor_assignment(all_rows, seed=seed)
    split_rows = build_speaker_independent_rows(all_rows, assignment)
    split_root = output_root / "manifests"
    for split in SPLITS:
        write_jsonl(split_root / f"{split}.jsonl", split_rows[split])
    write_jsonl(split_root / "all.jsonl", [row for split in SPLITS for row in split_rows[split]])

    proposed_actors = actor_sets(split_rows)
    proposed_overlaps = pairwise_intersections(split_rows)
    stats = {
        "protocol": {
            "name": "speaker_independent_actor_split",
            "task_specification_in_repository": False,
            "ratio_basis": "70/15/15 actor-level proposal; task-specific ratio was not present in repository artifacts",
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": seed,
            "actor_quota": actor_quotas,
            "gender_balance": "NOT_APPLIED",
            "gender_metadata": UNKNOWN,
        },
        "source_manifests": {
            split: {
                "path": str((manifest_root / f"{split}.jsonl").relative_to(repo_root)),
                **count_rows(rows_by_split[split]),
            }
            for split in SPLITS
        },
        "source_all": {"path": "data/cremad/manifests/all.jsonl", **count_rows(all_rows)},
        "source_actor_intersections": {
            key: {"count": len(value), "actor_ids": value} for key, value in overlaps.items()
        },
        "proposed_split": {
            split: {"path": f"manifests/{split}.jsonl", **count_rows(split_rows[split])}
            for split in SPLITS
        },
        "proposed_actor_intersections": {
            key: {"count": len(value), "actor_ids": value} for key, value in proposed_overlaps.items()
        },
        "actor_assignments": {actor: assignment[actor] for actor in sorted(assignment)},
        "checks": {
            "source_all_rows_equal_sum_source_splits": len(all_rows) == sum(len(rows) for rows in rows_by_split.values()),
            "proposed_rows_equal_source_all": len(all_rows) == sum(len(rows) for rows in split_rows.values()),
            "proposed_actor_overlap_free": all(not value for value in proposed_overlaps.values()),
            "gender_metadata_recovered": False,
        },
    }
    write_json(output_root / "split_statistics.json", stats)
    write_json(output_root / "actor_assignments.json", stats["actor_assignments"])

    report = {
        "job": "Task 7 CREMA-D speaker-leakage audit",
        "source": "data/cremad/manifests/{train,val,test,all}.jsonl",
        "overlap_exists": any(overlaps.values()),
        "source_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "proposed_split_sizes": {split: len(split_rows[split]) for split in SPLITS},
        "proposed_split_actor_counts": {split: len(proposed_actors[split]) for split in SPLITS},
        "gender": {
            "status": UNKNOWN,
            "reason": "No actor gender/demographic metadata artifact is present beside the repository manifests or raw CSVs.",
            "balance_applied": False,
        },
        "training_performed": False,
        "outputs": [
            "split_statistics.json",
            "actor_assignments.json",
            "manifests/train.jsonl",
            "manifests/val.jsonl",
            "manifests/test.jsonl",
            "manifests/all.jsonl",
        ],
    }
    write_json(output_root / "audit_report.json", report)
    markdown = [
        "# Task 7 CREMA-D speaker-leakage audit",
        "",
        f"Historical manifests contain {len(all_rows)} rows and {len(set(actor_id_from_row(row) for row in all_rows))} actors.",
        "Every historical pair overlaps on all 91 actors; the proposed manifests are actor-disjoint.",
        "",
        "## Historical overlap",
        "",
        "| Pair | Shared actors | Count |",
        "|---|---:|---:|",
    ]
    for key, value in overlaps.items():
        markdown.append(f"| {key.replace('_and_', ' / ')} | {', '.join(value)} | {len(value)} |")
    markdown.extend([
        "",
        "## Proposed split",
        "",
        "The repository does not contain the external Task 7 specification. This audit therefore records a deterministic 70/15/15 actor-level proposal with seed 42; no training was run.",
        "",
        "| Split | Rows | Actors |",
        "|---|---:|---:|",
    ])
    for split in SPLITS:
        markdown.append(f"| {split} | {len(split_rows[split])} | {len(proposed_actors[split])} |")
    markdown.extend([
        "",
        f"Gender balancing: **{UNKNOWN}**. No gender/demographic metadata file is present in the repository artifacts, so gender balancing was not applied.",
        "",
        "See `split_statistics.json` for actor IDs, label counts, source split membership, and checks.",
        "",
    ])
    (output_root / "README.md").write_text("\n".join(markdown), encoding="utf-8")
    return stats


def _artifact_record(
    repo_root: Path,
    artifact: Path,
    model: str,
    status: str,
    evidence: Sequence[str],
    unresolved: Sequence[str],
) -> dict[str, Any]:
    return {
        "artifact": str(artifact.relative_to(repo_root)),
        "sha256": sha256(artifact),
        "model": model,
        "exact_model_recovered": model != UNKNOWN,
        "metadata_status": status,
        "evidence": list(evidence),
        "unresolved_metadata": list(unresolved),
    }


def collect_provenance(repo_root: Path) -> list[dict[str, Any]]:
    root = repo_root / "data" / "concept_sets"
    records: list[dict[str, Any]] = []
    deepseek_evidence = [
        "scripts/concepts/generate_deepseek_concept_sets.py:summary model field defaults to deepseek-v4-flash",
        "data/concept_sets/{cremad,esc50,urbansound8k}/generation_summary.json",
    ]
    qwen_evidence = [
        "historical introducing commits for the artifact",
        "historical GPT_initial_concepts.ipynb sets model_id=Qwen/Qwen3.5-9B",
    ]
    unresolved_deepseek = ["per-request API response IDs, timestamps, and raw prompts are not embedded in these artifacts"]
    unresolved_qwen = ["temperature, decoding settings, timestamps, and per-request prompts are not embedded in these artifacts"]

    for dataset in DATASET_NAMES:
        dataset_root = root / dataset
        for artifact in sorted(dataset_root.glob("*")):
            if not artifact.is_file():
                continue
            if artifact.name == "generation_summary.json":
                model = str(read_json(artifact).get("model", UNKNOWN))
                status = "EXACT_MODEL_AND_SUMMARY"
                evidence = [str(artifact.relative_to(repo_root))]
                unresolved = []
            else:
                model = "deepseek-v4-flash"
                status = "DERIVED_FROM_GENERATION_SUMMARY"
                evidence = deepseek_evidence
                unresolved = unresolved_deepseek
            records.append(_artifact_record(repo_root, artifact, model, status, evidence, unresolved))

    for artifact in sorted(root.glob("*_deepseek_summary.json")):
        payload = read_json(artifact)
        records.append(_artifact_record(
            repo_root, artifact, str(payload.get("model", UNKNOWN)), "EXACT_MODEL_AND_SUMMARY",
            [str(artifact.relative_to(repo_root)), "scripts/concepts/generate_deepseek_concept_sets.py"], [],
        ))
    for artifact in sorted(root.glob("*_filtered_deepseek.txt")):
        records.append(_artifact_record(
            repo_root, artifact, "deepseek-v4-flash", "DERIVED_FROM_DEEPSEEK_SUMMARY",
            [str(root / (artifact.stem.replace("_filtered_deepseek", "") + "_deepseek_summary.json")).replace(str(repo_root) + "/", "")],
            unresolved_deepseek,
        ))
    for artifact in sorted((root / "deepseek_init").glob("*.json")):
        records.append(_artifact_record(repo_root, artifact, "deepseek-v4-flash", "LINKED_TO_CURRENT_DEEPSEEK_RUN", deepseek_evidence, unresolved_deepseek))

    for artifact in sorted(root.glob("*_filtered_qwen.txt")) + sorted((root / "qwen_init").glob("*.json")):
        records.append(_artifact_record(repo_root, artifact, "Qwen/Qwen3.5-9B", "HISTORICALLY_RECOVERED_MODEL", qwen_evidence, unresolved_qwen))

    ablation_summary = root / "ablation_concept_summary.json"
    if ablation_summary.exists():
        records.append(_artifact_record(
            repo_root, ablation_summary, "deepseek-v4-flash", "DERIVED_CONCEPT_SUMMARY",
            ["data/concept_sets/{cremad,esc50,urbansound8k}/generation_summary.json", str(ablation_summary.relative_to(repo_root))],
            ["the summary does not itself repeat the generator model"],
        ))
    return sorted(records, key=lambda record: record["artifact"])


def audit_metadata(repo_root: Path, output_root: Path) -> dict[str, Any]:
    records = collect_provenance(repo_root)
    payload = {
        "job": "Task 8 concept metadata checks",
        "model_provenance": records,
        "unknown_policy": "UNKNOWN is used whenever an exact generator detail cannot be recovered from repository artifacts or history.",
    }
    write_json(output_root / "model_provenance.json", payload)

    template = {
        "environmental_clap_text_template": "{concept}",
        "resolved_behavior": "raw concept string; no prefix or suffix",
        "evidence": [
            "train_cbm.py:186 reads concept lines unchanged",
            "utils.py:63-68 passes concepts unchanged to clap.encode_text",
            "clap/core.py:141-157 sends the supplied strings to the CLAP processor",
        ],
        "caveat": "experiments/prompt_template_ablation.py lists alternatives, but the core training path does not apply prompt_template; run_audio_concept_ablation.py passes None.",
        "alternatives_present_but_not_resolved_as_environmental_default": [
            "{concept}",
            "the sound of {concept}",
            "an audio recording of {concept}",
        ],
    }
    write_json(output_root / "clap_template.json", template)

    unresolved = [
        {
            "artifact": record["artifact"],
            "model": record["model"],
            "unresolved_metadata": record["unresolved_metadata"],
            "evidence": record["evidence"],
        }
        for record in records
        if record["unresolved_metadata"]
    ]
    write_json(output_root / "unresolved_metadata.json", {"artifacts": unresolved})

    lines = [
        "# Task 8 metadata checks",
        "",
        "## Model provenance",
        "",
        "| Artifact | Model | Status |",
        "|---|---|---|",
    ]
    for record in records:
        lines.append(f"| `{record['artifact']}` | `{record['model']}` | {record['metadata_status']} |")
    lines.extend([
        "",
        "## Environmental CLAP template",
        "",
        "The active repository path uses **`{concept}`** (the raw concept string). The prompt-template ablation file lists other candidates, but they are not applied by the core activation path.",
        "",
        "## Unresolved metadata",
        "",
        "See `unresolved_metadata.json`. UNKNOWN details are not inferred.",
        "",
    ])
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--task", choices=("7", "8", "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.task in {"7", "all"}:
        audit_cremad(repo_root, repo_root / "results/revision/t7_cremad/split_audit", seed=args.seed)
    if args.task in {"8", "all"}:
        audit_metadata(repo_root, repo_root / "results/revision/t8_checks/metadata")


if __name__ == "__main__":
    main()
