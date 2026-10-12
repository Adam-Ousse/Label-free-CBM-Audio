#!/usr/bin/env python3
"""Audit concept vocabularies for class-name leakage and write clean copies.

The audit is deliberately independent of model training.  It checks each
concept against the dataset class names using three fixed rules:

* lemmatized token overlap after English stop-word removal;
* maximum cosine similarity from ``all-mpnet-base-v2`` greater than 0.7;
* maximum normalized CLAP text similarity greater than 0.85.

The four inputs are the matched source-ablation vocabularies used by this
repository's sparse CBM checkpoints.  The final classifier is stored as
``W_g.pt`` in the release bundle; this script calls its absolute-weight mass
``|W_F|`` in the report to match the revision brief.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


DATASETS = ("esc50", "urbansound8k", "cremad")
VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")
VARIANT_FILES = {
    "lf": "concepts_ablation_lf.txt",
    "lf_broad": "concepts_ablation_lf_broad.txt",
    "lf_contrastive": "concepts_ablation_lf_contrastive.txt",
    "full": "concepts_ablation_full.txt",
}
CLASS_FILES = {
    "esc50": "data/esc50_classes.txt",
    "urbansound8k": "data/urbansound8k_classes.txt",
    "cremad": "data/cremad_classes.txt",
}
MPNET_THRESHOLD = 0.70
CLAP_THRESHOLD = 0.85
MPNET_MODEL = "all-mpnet-base-v2"
CLAP_MODEL = "laion/clap-htsat-unfused"
_MPNET_MODELS = {}
_CLAP_BUNDLES = {}

# This is intentionally local and versioned rather than relying on an
# unpinned NLP corpus download.  It covers the function words present in the
# dataset labels and generated concepts, plus the standard English stop-word
# list used for this kind of lexical filter.
STOP_WORDS = frozenset(
    (
        "a an and are as at be been being but by for from had has have he her here hers "
        "him his how i if in into is it its itself me more most my myself no nor not of "
        "on once only or our ours ourselves out over own same she so some such than that "
        "the their theirs them themselves then there these they this those through to too "
        "under until up very was we were what when where which while who whom why will "
        "with you your yours yourself yourselves s t d ll re ve"
    ).split()
)

IRREGULAR_LEMMAS = {
    "children": "child",
    "men": "man",
    "women": "woman",
    "people": "person",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "wolves": "wolf",
    "knives": "knife",
    "leaves": "leaf",
    "lives": "life",
    "sounds": "sound",
}


@dataclass(frozen=True)
class SimilarityScores:
    """Maximum concept-to-class scores and their corresponding class names."""

    mpnet_max: float | None
    mpnet_class: str
    clap_max: float | None
    clap_class: str


def tokenize(text: str) -> list[str]:
    """Tokenize labels/concepts, treating underscores and hyphens as spaces."""

    return re.findall(r"[a-z0-9]+", str(text).casefold())


def lemmatize_token(token: str) -> str:
    """Apply a small deterministic English lemmatizer for lexical filtering."""

    if token in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes")) and len(token) > 4:
        return token[:-2]
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem if stem.endswith("e") else stem
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def content_lemmas(text: str) -> set[str]:
    """Return lower-case, lemmatized, non-stop-word tokens."""

    return {
        lemma
        for token in tokenize(text)
        if token not in STOP_WORDS
        for lemma in (lemmatize_token(token),)
        if lemma not in STOP_WORDS
    }


def class_name_token_matches(concept: str, class_names: Sequence[str]) -> dict[str, list[str]]:
    """Return class names sharing one or more content lemmas with a concept."""

    concept_tokens = content_lemmas(concept)
    return {
        class_name: sorted(concept_tokens & content_lemmas(class_name))
        for class_name in class_names
        if concept_tokens & content_lemmas(class_name)
    }


def _max_score_row(
    scores: np.ndarray | None,
    row: int,
    class_names: Sequence[str],
) -> tuple[float | None, str]:
    if scores is None:
        return None, ""
    values = np.asarray(scores[row], dtype=float)
    index = int(np.argmax(values))
    return float(values[index]), str(class_names[index])


def score_concepts(
    concepts: Sequence[str],
    class_names: Sequence[str],
    mpnet_embeddings: np.ndarray | None = None,
    clap_embeddings: np.ndarray | None = None,
    class_mpnet_embeddings: np.ndarray | None = None,
    class_clap_embeddings: np.ndarray | None = None,
) -> list[SimilarityScores]:
    """Compute max cosine scores against classes from precomputed embeddings.

    Embeddings must be row-normalized.  Keeping this function independent of
    model loading makes the filtering policy unit-testable without network or
    GPU access.
    """

    def pairwise(concept_embeddings: np.ndarray | None, class_embeddings: np.ndarray | None):
        if concept_embeddings is None or class_embeddings is None:
            return None
        return np.asarray(concept_embeddings) @ np.asarray(class_embeddings).T

    mpnet_scores = pairwise(mpnet_embeddings, class_mpnet_embeddings)
    clap_scores = pairwise(clap_embeddings, class_clap_embeddings)
    return [
        SimilarityScores(
            *_max_score_row(mpnet_scores, index, class_names),
            *_max_score_row(clap_scores, index, class_names),
        )
        for index in range(len(concepts))
    ]


def _encode_mpnet(texts: Sequence[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    key = (model_name, device)
    if key not in _MPNET_MODELS:
        _MPNET_MODELS[key] = SentenceTransformer(model_name, device=device)
    model = _MPNET_MODELS[key]
    return np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def _encode_clap(
    texts: Sequence[str], model_name: str, device: str, batch_size: int
) -> np.ndarray:
    from clap.core import encode_text, load_clap_model

    key = (model_name, device)
    if key not in _CLAP_BUNDLES:
        _CLAP_BUNDLES[key] = load_clap_model(model_name=model_name, device=device)
    bundle = _CLAP_BUNDLES[key]
    return encode_text(list(texts), clap_bundle=bundle, batch_size=batch_size, normalize=True).numpy()


def compute_similarity_scores(
    concepts: Sequence[str],
    class_names: Sequence[str],
    *,
    mpnet_model: str = MPNET_MODEL,
    clap_model: str = CLAP_MODEL,
    device: str = "cpu",
    batch_size: int = 64,
) -> list[SimilarityScores]:
    """Load the two text encoders once and return per-concept max scores."""

    all_text = list(concepts) + list(class_names)
    mpnet = _encode_mpnet(all_text, mpnet_model, device, batch_size)
    clap = _encode_clap(all_text, clap_model, device, batch_size)
    concept_count = len(concepts)
    return score_concepts(
        concepts,
        class_names,
        mpnet[:concept_count],
        clap[:concept_count],
        mpnet[concept_count:],
        clap[concept_count:],
    )


def load_concepts(path: Path) -> list[str]:
    """Read non-empty concept lines without changing their original text."""

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _weight_mass(
    checkpoint_dir: Path | None,
    concepts: Sequence[str],
) -> tuple[dict[str, float], float, str]:
    """Return per-concept absolute W_g mass and coverage status."""

    if checkpoint_dir is None:
        return {}, 0.0, "not_requested"
    concepts_path = checkpoint_dir / "concepts.txt"
    weights_path = checkpoint_dir / "W_g.pt"
    if not concepts_path.is_file() or not weights_path.is_file():
        return {}, 0.0, "missing_checkpoint_files"

    import torch

    head_concepts = load_concepts(concepts_path)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    values = weights.detach().float().abs().sum(dim=0).numpy()
    if len(head_concepts) != len(values):
        raise ValueError(
            "Checkpoint concept/weight mismatch: {} has {}, W_g has {}".format(
                concepts_path, len(head_concepts), len(values)
            )
        )
    by_key: dict[str, float] = {}
    for concept, mass in zip(head_concepts, values):
        key = concept.casefold()
        by_key[key] = by_key.get(key, 0.0) + float(mass)
    input_keys = {concept.casefold() for concept in concepts}
    head_keys = set(by_key)
    matched = {concept: by_key[concept.casefold()] for concept in concepts if concept.casefold() in by_key}
    if not matched:
        status = "no_concept_overlap"
    elif head_keys.issubset(input_keys):
        status = "checkpoint_subset_of_input"
    elif input_keys.issubset(head_keys):
        status = "input_subset_of_checkpoint"
    else:
        status = "partial_concept_overlap"
    return matched, float(values.sum()), status


def make_audit_rows(
    dataset: str,
    variant: str,
    concepts: Sequence[str],
    class_names: Sequence[str],
    scores: Sequence[SimilarityScores],
    *,
    weights: Mapping[str, float] | None = None,
    mpnet_threshold: float = MPNET_THRESHOLD,
    clap_threshold: float = CLAP_THRESHOLD,
) -> list[dict[str, object]]:
    """Build complete, one-row-per-input-concept audit records."""

    weights = weights or {}
    rows = []
    for index, (concept, score) in enumerate(zip(concepts, scores)):
        overlaps = class_name_token_matches(concept, class_names)
        reasons = []
        if overlaps:
            reasons.append("class_name_token_overlap")
        if score.mpnet_max is not None and score.mpnet_max > mpnet_threshold:
            reasons.append("mpnet_similarity_gt_0.7")
        if score.clap_max is not None and score.clap_max > clap_threshold:
            reasons.append("clap_text_similarity_gt_0.85")
        rows.append(
            {
                "dataset": dataset,
                "vocabulary": variant,
                "concept_index": index,
                "concept": concept,
                "flagged": bool(reasons),
                "reasons": ";".join(reasons),
                "class_name_overlap_classes": ";".join(overlaps),
                "class_name_overlap_tokens": ";".join(
                    sorted({token for tokens in overlaps.values() for token in tokens})
                ),
                "mpnet_max_similarity": score.mpnet_max,
                "mpnet_match_class": score.mpnet_class,
                "clap_max_similarity": score.clap_max,
                "clap_match_class": score.clap_class,
                "abs_W_F_mass": float(weights.get(concept, 0.0)),
            }
        )
    return rows


def _dataset_classes(dataset: str, repo_root: Path) -> list[str]:
    path = repo_root / CLASS_FILES[dataset]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "dataset",
        "vocabulary",
        "concept_index",
        "concept",
        "flagged",
        "reasons",
        "class_name_overlap_classes",
        "class_name_overlap_tokens",
        "mpnet_max_similarity",
        "mpnet_match_class",
        "clap_max_similarity",
        "clap_match_class",
        "abs_W_F_mass",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float | None) -> str:
    return "NA" if value is None else "{:.6f}".format(value)


def _write_report(
    path: Path,
    *,
    command: str,
    repo_root: Path,
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
    model_config: Mapping[str, str],
) -> None:
    lines = [
        "# Task 2 leakage audit",
        "",
        "This is the preparation-only audit. No model retraining was launched.",
        "",
        "## Fixed protocol",
        "",
        "- Lexical rule: lemmatized content-token overlap between a concept and any dataset class name.",
        "- MPNet rule: maximum cosine similarity against class names **> 0.70**.",
        "- CLAP rule: maximum normalized text similarity against class names **> 0.85**.",
        "- Stop words are removed before lexical comparison; threshold equality is not flagged.",
        "- The sparse-head mass uses `sum_classes(abs(W_g[:, concept]))`, reported as `|W_F|`.",
        "- Similarity thresholds were fixed before inspecting test results; test data are not loaded.",
        "",
        "## Reproduction",
        "",
        "```bash",
        command,
        "```",
        "",
        "- Repository commit observed when the audit ran: `{}`".format(_git_commit(repo_root)),
        "- MPNet model: `{}`".format(model_config["mpnet_model"]),
        "- CLAP model: `{}`".format(model_config["clap_model"]),
        "- Device: `{}`".format(model_config["device"]),
        "",
        "## Counts and sparse-head mass",
        "",
        "| Dataset | Vocabulary | Concepts | Flagged | Clean | Flagged / total | Flagged `|W_F|` mass | Total `|W_F|` mass | Mass fraction | Weight coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {dataset} | {vocabulary} | {total} | {flagged} | {clean} | {flagged_fraction:.4f} | {flagged_mass:.6f} | {total_mass:.6f} | {mass_fraction:.4f} | {weight_status} |".format(
                **summary
            )
        )
    lines += [
        "",
        "## Flagging reasons",
        "",
        "`audit.csv` is complete: it contains one row for every non-empty line in every input vocabulary. The `reasons` field records every rule that fired, and the adjacent match/value columns record the exact class and score used for the decision.",
        "",
        "Flagged concept lists are under `flagged_concepts/`; clean copies are under `data/concept_sets/<dataset>/clean/`.",
        "",
    ]
    unexpected = [
        s
        for s in summaries
        if s["weight_status"] in {"no_concept_overlap", "partial_concept_overlap"}
    ]
    if unexpected:
        lines += [
            "## Human review required",
            "",
            "The following checkpoint/vocabulary pairings do not have safe concept coverage: {}. Their mass fractions use only checkpoint concepts that matched by case-insensitive text, so review the provenance before using these fractions for paper claims.".format(
                ", ".join("{}/{}".format(s["dataset"], s["vocabulary"]) for s in unexpected)
            ),
            "",
        ]
    else:
        lines += ["## Human review", "", "No checkpoint/vocabulary coverage anomalies were detected.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output-dir", type=Path, default=Path("results/revision/t2_leakage/audit"))
    parser.add_argument("--concept-root", type=Path, default=Path("data/concept_sets"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("release/google_drive_bundle/checkpoints"))
    parser.add_argument("--device", default="cpu", help="MPNet/CLAP device; CPU is the default for this audit.")
    parser.add_argument("--mpnet-model", default=MPNET_MODEL)
    parser.add_argument("--clap-model", default=CLAP_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mpnet-threshold", type=float, default=MPNET_THRESHOLD)
    parser.add_argument("--clap-threshold", type=float, default=CLAP_THRESHOLD)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the audit and write all requested artifacts."""

    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    concept_root = (repo_root / args.concept_root).resolve()
    checkpoint_root = (repo_root / args.checkpoint_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    flagged_dir = output_dir / "flagged_concepts"
    flagged_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries = []
    for dataset in args.datasets:
        classes = _dataset_classes(dataset, repo_root)
        vocabulary_concepts = {}
        unique_concepts = []
        seen_concepts = set()
        for variant in args.variants:
            input_path = concept_root / dataset / VARIANT_FILES[variant]
            if not input_path.is_file():
                raise FileNotFoundError("Missing vocabulary: {}".format(input_path))
            concepts = load_concepts(input_path)
            vocabulary_concepts[variant] = concepts
            for concept in concepts:
                key = concept.casefold()
                if key not in seen_concepts:
                    seen_concepts.add(key)
                    unique_concepts.append(concept)
        unique_scores = compute_similarity_scores(
            unique_concepts,
            classes,
            mpnet_model=args.mpnet_model,
            clap_model=args.clap_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        score_by_key = {
            concept.casefold(): score for concept, score in zip(unique_concepts, unique_scores)
        }
        for variant in args.variants:
            concepts = vocabulary_concepts[variant]
            scores = [score_by_key[concept.casefold()] for concept in concepts]
            weights, total_mass, weight_status = _weight_mass(
                checkpoint_root / dataset / variant,
                concepts,
            )
            rows = make_audit_rows(
                dataset,
                variant,
                concepts,
                classes,
                scores,
                weights=weights,
                mpnet_threshold=args.mpnet_threshold,
                clap_threshold=args.clap_threshold,
            )
            all_rows.extend(rows)
            flagged = [row["concept"] for row in rows if row["flagged"]]
            (flagged_dir / dataset).mkdir(parents=True, exist_ok=True)
            (flagged_dir / dataset / (VARIANT_FILES[variant])).write_text(
                "\n".join(flagged) + ("\n" if flagged else ""), encoding="utf-8"
            )
            clean_dir = concept_root / dataset / "clean"
            clean_dir.mkdir(parents=True, exist_ok=True)
            (clean_dir / VARIANT_FILES[variant]).write_text(
                "\n".join(row["concept"] for row in rows if not row["flagged"]) + "\n",
                encoding="utf-8",
            )
            flagged_mass = sum(row["abs_W_F_mass"] for row in rows if row["flagged"])
            summaries.append(
                {
                    "dataset": dataset,
                    "vocabulary": variant,
                    "total": len(rows),
                    "flagged": len(flagged),
                    "clean": len(rows) - len(flagged),
                    "flagged_fraction": len(flagged) / len(rows) if rows else 0.0,
                    "flagged_mass": flagged_mass,
                    "total_mass": total_mass,
                    "mass_fraction": flagged_mass / total_mass if total_mass else 0.0,
                    "weight_status": weight_status,
                }
            )

    _write_csv(output_dir / "audit.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "mpnet_threshold": args.mpnet_threshold,
                    "clap_threshold": args.clap_threshold,
                    "mpnet_rule": "strictly greater than",
                    "clap_rule": "strictly greater than",
                    "stop_words_removed": True,
                    "lemmatized_token_overlap": True,
                    "test_data_loaded": False,
                },
                "summaries": summaries,
                "git_commit": _git_commit(repo_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = " ".join(["/venv/main/bin/python", "scripts/concepts/leakage_audit.py"] + list(getattr(args, "_argv", [])))
    _write_report(
        output_dir / "README.md",
        command=command,
        repo_root=repo_root,
        rows=all_rows,
        summaries=summaries,
        model_config={
            "mpnet_model": args.mpnet_model,
            "clap_model": args.clap_model,
            "device": args.device,
        },
    )
    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args._argv = list(sys.argv[1:]) if argv is None else list(argv)
    summaries = run(args)
    for summary in summaries:
        print(
            "{dataset}/{vocabulary}: flagged={flagged}/{total} clean={clean} "
            "mass_fraction={mass_fraction:.6f}".format(**summary)
        )


if __name__ == "__main__":
    main()
