"""CPU analysis utilities for Task 5 concept--descriptor validation.

The module deliberately separates the author-reviewed map from all numerical
analysis.  A mapping is never selected from the observed correlations.  The
same fixed mapping is used for the matched analysis, sign checks, null pairs,
and controlled-edit summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from features.audio_descriptors import DESCRIPTOR_NAMES


CONTROLLED_EDIT_SPECS = (
    {"name": "pitch_shift_minus_4_semitones", "family": "pitch_shift", "semitones": -4.0},
    {"name": "pitch_shift_plus_4_semitones", "family": "pitch_shift", "semitones": 4.0},
    {"name": "low_pass_2khz", "family": "low_pass", "cutoff_hz": 2000.0},
    {"name": "gain_minus_20db", "family": "gain", "gain_db": -20.0},
    {"name": "synthetic_reverb_rt60_1s", "family": "reverb", "rt60_seconds": 1.0},
    {"name": "time_stretch_0p7x", "family": "time_stretch", "rate": 0.7},
)


def load_descriptor_map(path: str | Path) -> dict[str, Any]:
    """Load and validate the small, explicit author-reviewed map."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    mappings = payload.get("mappings") if isinstance(payload, dict) else None
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("descriptor map must contain a non-empty 'mappings' list")
    seen: set[tuple[str, str, str]] = set()
    for item in mappings:
        if not isinstance(item, dict):
            raise ValueError("each descriptor mapping must be an object")
        concept = str(item.get("concept", "")).strip()
        descriptor = str(item.get("descriptor", "")).strip()
        datasets = item.get("datasets")
        sign = item.get("expected_sign")
        if not concept or not descriptor or not isinstance(datasets, list) or not datasets:
            raise ValueError(f"invalid descriptor mapping: {item!r}")
        if sign not in (-1, 1):
            raise ValueError(f"expected_sign must be -1 or 1: {item!r}")
        for dataset in datasets:
            key = (str(dataset), concept, descriptor)
            if key in seen:
                raise ValueError(f"duplicate descriptor mapping: {key}")
            seen.add(key)
    return payload


def validate_mapping_against_models(
    descriptor_map: Mapping[str, Any],
    model_concepts: Mapping[str, Mapping[str, Iterable[str]] | Iterable[str]],
) -> list[dict[str, Any]]:
    """Confirm every mapped concept occurs in at least one trained model.

    ``model_concepts`` is keyed by dataset and may contain either one iterable
    of concepts or a mapping of model name to concept iterable.
    """

    checks: list[dict[str, Any]] = []
    for item in descriptor_map["mappings"]:
        dataset_hits: dict[str, list[str]] = {}
        for dataset in item["datasets"]:
            candidates = model_concepts.get(str(dataset), {})
            if isinstance(candidates, Mapping):
                hits = [str(model) for model, concepts in candidates.items() if item["concept"] in set(concepts)]
            else:
                hits = ["provided"] if item["concept"] in set(candidates) else []
            dataset_hits[str(dataset)] = sorted(hits)
        if not any(dataset_hits.values()):
            raise ValueError(f"mapped concept is absent from all trained models: {item['concept']!r}")
        checks.append({"concept": item["concept"], "datasets": dataset_hits})
    return checks


def _finite_pairs(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"paired inputs must have equal shape, got {left.shape} and {right.shape}")
    keep = np.isfinite(left) & np.isfinite(right)
    return left[keep], right[keep]


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Compute tie-aware Spearman correlation, returning ``None`` if undefined."""

    left, right = _finite_pairs(x, y)
    if left.size < 2:
        return None
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def bootstrap_spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int = 42,
    draws: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Bootstrap a paired percentile CI for Spearman rho."""

    left, right = _finite_pairs(x, y)
    if left.size < 2 or draws <= 0 or not 0.0 < confidence < 1.0:
        return None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, left.size, size=(int(draws), left.size))
    values = [spearman_rho(left[index], right[index]) for index in indices]
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if values.size == 0:
        return None
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def expected_sign_matches(rho: float | None, expected_sign: int) -> bool | None:
    """Check a pre-registered sign; zero/undefined correlations are failures."""

    if rho is None:
        return None
    if expected_sign not in (-1, 1):
        raise ValueError("expected_sign must be -1 or 1")
    return bool(float(rho) * int(expected_sign) > 0.0)


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int = 42,
    draws: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Bootstrap a percentile CI for the mean of finite values."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0 or draws <= 0 or not 0.0 < confidence < 1.0:
        return None
    rng = np.random.default_rng(int(seed))
    samples = array[rng.integers(0, array.size, size=(int(draws), array.size))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def correlate_pair(
    concept_values: Sequence[float],
    descriptor_values: Sequence[float],
    *,
    expected_sign: int | None = None,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Return correlation, finite sample count, sign check, and bootstrap CI."""

    left, right = _finite_pairs(concept_values, descriptor_values)
    rho = spearman_rho(left, right)
    ci = bootstrap_spearman_ci(left, right, seed=seed, draws=bootstrap_resamples)
    result: dict[str, Any] = {
        "n": int(left.size),
        "spearman_rho": rho,
        "ci_low": ci[0] if ci else None,
        "ci_high": ci[1] if ci else None,
    }
    if expected_sign is not None:
        result["expected_sign"] = int(expected_sign)
        result["expected_sign_match"] = expected_sign_matches(rho, int(expected_sign))
    return result


def aggregate_dataset_concept_correlations(
    observations: Iterable[Mapping[str, Any]],
    descriptor_map: Mapping[str, Any],
    *,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
) -> list[dict[str, Any]]:
    """Aggregate observations by dataset, mapped concept, and descriptor."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    mapping_by_key = {
        (str(dataset), str(item["concept"]), str(item["descriptor"])): item
        for item in descriptor_map["mappings"]
        for dataset in item["datasets"]
    }
    for row in observations:
        key = (str(row["dataset"]), str(row["concept"]), str(row["descriptor"]))
        if key in mapping_by_key:
            grouped[key].append(row)
    results: list[dict[str, Any]] = []
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        mapping = mapping_by_key[key]
        stats = correlate_pair(
            [row["concept_value"] for row in rows],
            [row["descriptor_value"] for row in rows],
            expected_sign=int(mapping["expected_sign"]),
            seed=int(seed) + index * 1009,
            bootstrap_resamples=bootstrap_resamples,
        )
        results.append(
            {
                "dataset": key[0],
                "concept": key[1],
                "descriptor": key[2],
                "models": sorted({str(row.get("model", "unknown")) for row in rows}),
                **stats,
            }
        )
    return results


def random_concept_descriptor_null(
    concept_values: np.ndarray,
    descriptor_values: np.ndarray,
    concept_names: Sequence[str],
    descriptor_names: Sequence[str],
    mapped_pairs: Iterable[tuple[str, str]],
    *,
    draws: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate exactly ``draws`` random, unmatched concept--descriptor pairs."""

    concepts = np.asarray(concept_values, dtype=np.float64)
    descriptors = np.asarray(descriptor_values, dtype=np.float64)
    if concepts.ndim != 2 or descriptors.ndim != 2 or concepts.shape[0] != descriptors.shape[0]:
        raise ValueError("concept_values and descriptor_values must be [samples, features] with equal samples")
    if concepts.shape[1] != len(concept_names) or descriptors.shape[1] != len(descriptor_names):
        raise ValueError("feature matrix width does not match feature names")
    forbidden = {(str(left), str(right)) for left, right in mapped_pairs}
    candidates = [
        (concept_index, descriptor_index)
        for concept_index, concept in enumerate(concept_names)
        for descriptor_index, descriptor in enumerate(descriptor_names)
        if (str(concept), str(descriptor)) not in forbidden
    ]
    if not candidates or draws <= 0:
        raise ValueError("null requires at least one unmatched pair and positive draws")
    rng = np.random.default_rng(int(seed))
    selected = rng.integers(0, len(candidates), size=int(draws))
    result: list[dict[str, Any]] = []
    for pair_id, candidate_index in enumerate(selected):
        concept_index, descriptor_index = candidates[int(candidate_index)]
        left = concepts[:, concept_index]
        right = descriptors[:, descriptor_index]
        finite_n = int(np.isfinite(left).astype(np.int64).dot(np.isfinite(right).astype(np.int64)))
        result.append(
            {
                "pair_id": int(pair_id),
                "concept": str(concept_names[concept_index]),
                "descriptor": str(descriptor_names[descriptor_index]),
                "n": finite_n,
                "spearman_rho": spearman_rho(left, right),
            }
        )
    return result


def aggregate_controlled_edits(
    batches: Iterable[Mapping[str, Any]],
    descriptor_map: Mapping[str, Any],
    *,
    unmatched_draws: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare fixed mapped pairs with random unmatched pairs after edits.

    Each batch contains ``dataset``, ``edit``, ``concept_names``,
    ``descriptor_names``, ``concept_changes`` and ``descriptor_changes``.
    Change matrices are [samples, features] and must already be produced by
    the eventual controlled-edit inference job.  No map selection happens here.
    """

    mapping_by_dataset = {
        str(dataset): [item for item in descriptor_map["mappings"] if str(dataset) in item["datasets"]]
        for dataset in {str(dataset) for item in descriptor_map["mappings"] for dataset in item["datasets"]}
    }
    summaries: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        dataset = str(batch["dataset"])
        concept_names = [str(value) for value in batch["concept_names"]]
        descriptor_names = [str(value) for value in batch["descriptor_names"]]
        concept_changes = np.asarray(batch["concept_changes"], dtype=np.float64)
        descriptor_changes = np.asarray(batch["descriptor_changes"], dtype=np.float64)
        if concept_changes.ndim != 2 or descriptor_changes.ndim != 2:
            raise ValueError("controlled-edit changes must be two-dimensional")
        if concept_changes.shape[0] != descriptor_changes.shape[0]:
            raise ValueError("controlled-edit matrices must have equal sample counts")
        concept_index = {name: index for index, name in enumerate(concept_names)}
        descriptor_index = {name: index for index, name in enumerate(descriptor_names)}
        mapped = mapping_by_dataset.get(dataset, [])
        matched_values: list[float] = []
        matched_records: list[dict[str, Any]] = []
        forbidden: set[tuple[str, str]] = set()
        for item in mapped:
            concept = str(item["concept"])
            descriptor = str(item["descriptor"])
            if concept not in concept_index or descriptor not in descriptor_index:
                continue
            forbidden.add((concept, descriptor))
            rho = spearman_rho(
                concept_changes[:, concept_index[concept]],
                descriptor_changes[:, descriptor_index[descriptor]],
            )
            if rho is not None:
                matched_values.append(rho)
            matched_records.append({"concept": concept, "descriptor": descriptor, "rho": rho})
        candidates = [
            (left, right)
            for left in concept_names
            for right in descriptor_names
            if (left, right) not in forbidden
        ]
        if not candidates:
            unmatched_values: list[float] = []
        else:
            rng = np.random.default_rng(int(seed) + batch_index * 1009)
            selected = rng.integers(0, len(candidates), size=max(0, int(unmatched_draws)))
            unmatched_values = [
                rho
                for selected_index in selected
                for rho in [
                    spearman_rho(
                        concept_changes[:, concept_index[candidates[int(selected_index)][0]]],
                        descriptor_changes[:, descriptor_index[candidates[int(selected_index)][1]]],
                    )
                ]
                if rho is not None
            ]
        matched_mean = float(np.mean(matched_values)) if matched_values else None
        unmatched_mean = float(np.mean(unmatched_values)) if unmatched_values else None
        matched_ci = bootstrap_mean_ci(matched_values, seed=int(seed) + batch_index * 1009 + 1)
        unmatched_ci = bootstrap_mean_ci(unmatched_values, seed=int(seed) + batch_index * 1009 + 2)
        delta_values = (
            np.asarray(matched_values)[:, None] - np.asarray(unmatched_values)[None, :]
        ).reshape(-1) if matched_values and unmatched_values else []
        delta_ci = bootstrap_mean_ci(delta_values, seed=int(seed) + batch_index * 1009 + 3)
        summaries.append(
            {
                "dataset": dataset,
                "edit": str(batch["edit"]),
                "matched_pairs": matched_records,
                "matched_pair_count": len(matched_values),
                "unmatched_pair_count": len(unmatched_values),
                "matched_mean_rho": matched_mean,
                "unmatched_mean_rho": unmatched_mean,
                "matched_ci": list(matched_ci) if matched_ci else None,
                "unmatched_ci": list(unmatched_ci) if unmatched_ci else None,
                "matched_minus_unmatched": (
                    matched_mean - unmatched_mean
                    if matched_mean is not None and unmatched_mean is not None
                    else None
                ),
                "matched_minus_unmatched_ci": list(delta_ci) if delta_ci else None,
                "mapping_is_fixed": True,
            }
        )
    return summaries


def plot_controlled_edit_figure(summary_rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> None:
    """Plot matched versus unmatched controlled-edit correlations."""

    import matplotlib.pyplot as plt

    rows = list(summary_rows)
    if not rows:
        raise ValueError("cannot plot an empty controlled-edit summary")
    labels = [str(row["edit"]) for row in rows]
    matched = [row["matched_mean_rho"] if row["matched_mean_rho"] is not None else 0.0 for row in rows]
    unmatched = [row["unmatched_mean_rho"] if row["unmatched_mean_rho"] is not None else 0.0 for row in rows]
    positions = np.arange(len(rows), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(max(8.0, len(rows) * 1.3), 4.5))
    matched_errors = [
        [
            max(0.0, float(row["matched_mean_rho"]) - float(row["matched_ci"][0]))
            if row.get("matched_ci")
            else 0.0
            for row in rows
        ],
        [
            max(0.0, float(row["matched_ci"][1]) - float(row["matched_mean_rho"]))
            if row.get("matched_ci")
            else 0.0
            for row in rows
        ],
    ]
    unmatched_errors = [
        [
            max(0.0, float(row["unmatched_mean_rho"]) - float(row["unmatched_ci"][0]))
            if row.get("unmatched_ci")
            else 0.0
            for row in rows
        ],
        [
            max(0.0, float(row["unmatched_ci"][1]) - float(row["unmatched_mean_rho"]))
            if row.get("unmatched_ci")
            else 0.0
            for row in rows
        ],
    ]
    axis.bar(positions - 0.18, matched, width=0.36, yerr=matched_errors, capsize=3, label="mapped pairs")
    axis.bar(positions + 0.18, unmatched, width=0.36, yerr=unmatched_errors, capsize=3, label="unmatched null pairs")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Spearman rho of edit changes")
    axis.set_title("Controlled-edit concept–descriptor alignment")
    axis.legend(frameon=False)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _load_temporal_bundle(path: Path) -> dict[str, Any]:
    import torch

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    required = {"temporal_concepts", "sample_ids", "concepts", "dataset"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"temporal bundle missing {sorted(missing)}: {path}")
    return bundle


def _cache_descriptor_values(cache_root: Path, dataset: str, row: Mapping[str, Any]) -> dict[str, float]:
    with np.load(cache_root / str(row["clip_cache"]), allow_pickle=False) as archive:
        values = archive["values"].astype(np.float64)
        names = [str(value) for value in archive["names"].tolist()]
        result = dict(zip(names, values))
        if "egemaps_names" in archive:
            result.update(
                zip(
                    [str(value) for value in archive["egemaps_names"].tolist()],
                    archive["egemaps_values"].astype(np.float64),
                )
            )
    return result


def build_descriptor_observations(
    temporal_paths: Sequence[str | Path],
    descriptor_root: str | Path,
    descriptor_map: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Join saved temporal concepts with clip-level descriptor caches."""

    root = Path(descriptor_root)
    manifests: dict[tuple[str, str], dict[str, Any]] = {}
    for line in (root / "cache_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        manifests[(str(row["dataset"]), str(row["id"]))] = row
    observations: list[dict[str, Any]] = []
    bundle_matrices: dict[str, dict[str, Any]] = {}
    for temporal_path in temporal_paths:
        path = Path(temporal_path)
        bundle = _load_temporal_bundle(path)
        dataset = str(bundle["dataset"])
        temporal_value = bundle["temporal_concepts"]
        if hasattr(temporal_value, "detach"):
            temporal_value = temporal_value.detach().cpu().numpy()
        temporal = np.asarray(temporal_value, dtype=np.float64)
        if temporal.ndim != 3:
            raise ValueError(f"temporal concepts must be [samples, windows, concepts]: {path}")
        concepts = [str(value) for value in bundle["concepts"]]
        by_concept = {name: index for index, name in enumerate(concepts)}
        sample_ids = [str(value) for value in bundle["sample_ids"]]
        mapped = [item for item in descriptor_map["mappings"] if dataset in item["datasets"]]
        matrix_descriptor_names: list[str] = []
        matrix_descriptor_rows: list[list[float]] = []
        for sample_index, sample_id in enumerate(sample_ids):
            manifest_row = manifests.get((dataset, sample_id))
            if manifest_row is None:
                raise ValueError(f"no descriptor cache manifest row for {(dataset, sample_id)}")
            descriptor_values = _cache_descriptor_values(root, dataset, manifest_row)
            if not matrix_descriptor_names:
                matrix_descriptor_names = list(DESCRIPTOR_NAMES)
                if dataset == "cremad":
                    matrix_descriptor_names.extend(
                        sorted(name for name in descriptor_values if name not in matrix_descriptor_names)
                    )
            matrix_descriptor_rows.append([descriptor_values.get(name, np.nan) for name in matrix_descriptor_names])
            for item in mapped:
                concept = str(item["concept"])
                descriptor = str(item["descriptor"])
                if concept not in by_concept or descriptor not in descriptor_values:
                    continue
                observations.append(
                    {
                        "dataset": dataset,
                        "model": path.parent.name,
                        "sample_id": sample_id,
                        "concept": concept,
                        "descriptor": descriptor,
                        "concept_value": float(np.nanmean(temporal[sample_index, :, by_concept[concept]])),
                        "descriptor_value": descriptor_values[descriptor],
                    }
                )
        key = str(path)
        bundle_matrices[key] = {
            "dataset": dataset,
            "model": path.parent.name,
            "concept_names": concepts,
            "concept_values": np.nanmean(temporal, axis=1),
            "descriptor_names": matrix_descriptor_names,
            "descriptor_values": np.asarray(matrix_descriptor_rows, dtype=np.float64),
        }
    return observations, bundle_matrices


def run_descriptor_analysis(
    temporal_paths: Sequence[str | Path],
    descriptor_root: str | Path,
    descriptor_map_path: str | Path,
    *,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
    null_draws: int = 1000,
) -> dict[str, Any]:
    """Run fixed-map correlations and write-ready JSON-serializable results."""

    descriptor_map = load_descriptor_map(descriptor_map_path)
    observations, bundle_matrices = build_descriptor_observations(temporal_paths, descriptor_root, descriptor_map)
    model_concepts: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for path in temporal_paths:
        bundle = _load_temporal_bundle(Path(path))
        model_concepts[str(bundle["dataset"])][Path(path).parent.name] = set(bundle["concepts"])
    validation = validate_mapping_against_models(descriptor_map, model_concepts)
    correlations = aggregate_dataset_concept_correlations(
        observations,
        descriptor_map,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    null_pairings: list[dict[str, Any]] = []
    for bundle_index, matrix in enumerate(bundle_matrices.values()):
        dataset = str(matrix["dataset"])
        mapped_pairs = {
            (str(item["concept"]), str(item["descriptor"]))
            for item in descriptor_map["mappings"]
            if dataset in item["datasets"]
        }
        try:
            rows = random_concept_descriptor_null(
                matrix["concept_values"],
                matrix["descriptor_values"],
                matrix["concept_names"],
                matrix["descriptor_names"],
                mapped_pairs,
                draws=null_draws,
                seed=int(seed) + bundle_index * 1009,
            )
        except ValueError:
            rows = []
        for row in rows:
            null_pairings.append({"dataset": dataset, "model": matrix["model"], **row})
    return {
        "mapping_size": len(descriptor_map["mappings"]),
        "mapping_validation": validation,
        "correlations": correlations,
        "null_pairings": null_pairings,
        "null_pairings_per_model": int(null_draws),
        "null_definition": (
            "random unmatched concept-descriptor pairings per temporal model; "
            "mapping is excluded from the null"
        ),
        "controlled_edit_specs": list(CONTROLLED_EDIT_SPECS),
        "observation_count": len(observations),
        "seed": int(seed),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-map", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, default=Path("results/revision/t5_descriptors"))
    parser.add_argument("--temporal-concepts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--null-draws", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_descriptor_analysis(
        args.temporal_concepts,
        args.descriptor_root,
        args.descriptor_map,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        null_draws=args.null_draws,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("mapping_size", "observation_count")}, indent=2))


if __name__ == "__main__":
    main()
