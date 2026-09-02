#!/usr/bin/env python3
"""Build and verify the external artifact bundle used by the reported experiments.

The Git repository contains code and compact concept metadata. Trained CBM
weights and generated evaluation artifacts live outside Git. This script gathers
the canonical outputs into one directory suitable for upload to Google Drive and
writes SHA-256 checksums for every file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "release" / "google_drive_bundle"
ENVIRONMENTAL_SUMMARY = REPO_ROOT / "results/audio_concept_ablation/cbm/summary.json"
CREMAD_SUMMARY = REPO_ROOT / (
    "results/cremad_targeted_rerun_20260828/source_ablation_canonical/"
    "source_ablation_summary.json"
)
CREMAD_GENERATION = REPO_ROOT / (
    "results/cremad_targeted_rerun_20260828/generation/source_ablations_canonical"
)
MODEL_FILES = (
    "W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt",
    "concepts.txt", "metrics.txt", "args.txt",
)
VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing bundle after a complete staging build.")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing bundle against its manifest.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_model(source_dir: Path, destination_dir: Path) -> None:
    for name in MODEL_FILES:
        copy_file(source_dir / name, destination_dir / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(root: Path) -> list[dict]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return files


def model_record(dataset: str, run: dict, destination: Path, staging: Path) -> dict:
    test = run.get("test", {})
    return {
        "dataset": dataset,
        "variant": run["variant"],
        "checkpoint_dir": destination.relative_to(staging).as_posix(),
        "input_concepts": run["input_concepts"],
        "retained_concepts": run.get("retained_concepts") or test.get("retained_concepts"),
        "accuracy": run.get("test_accuracy") or test.get("accuracy"),
        "macro_f1": run.get("test_f1_macro") or test.get("macro_f1"),
        "sparsity": run.get("sparsity_fraction") or test.get("sparsity"),
    }


def build_bundle(staging: Path) -> list[dict]:
    environmental = read_json(ENVIRONMENTAL_SUMMARY)
    environmental_runs = [run for run in environmental["runs"] if run["dataset"] in {"esc50", "urbansound8k"}]
    expected = {(dataset, variant) for dataset in ("esc50", "urbansound8k") for variant in VARIANTS}
    observed = {(run["dataset"], run["variant"]) for run in environmental_runs}
    if observed != expected:
        raise ValueError("Environmental summary must contain the eight canonical runs")

    records = []
    for run in environmental_runs:
        dataset, variant = run["dataset"], run["variant"]
        destination = staging / "checkpoints" / dataset / variant
        copy_model(resolve_repo_path(run["model_dir"]), destination)
        copy_file(resolve_repo_path(run["concept_set"]), staging / "concept_sets" / dataset / f"{variant}.txt")
        records.append(model_record(dataset, run, destination, staging))

    cremad = read_json(CREMAD_SUMMARY)
    cremad_runs = {run["variant"]: run for run in cremad["results"]}
    if set(cremad_runs) != set(VARIANTS):
        raise ValueError("CREMA-D summary must contain the four canonical runs")
    generation_manifest = read_json(CREMAD_GENERATION / "manifest.json")
    for variant in VARIANTS:
        run = cremad_runs[variant]
        destination = staging / "checkpoints" / "cremad" / variant
        copy_model(resolve_repo_path(run["model_dir"]), destination)
        grounded_name = generation_manifest["variants"][variant]["grounded_file"]
        copy_file(CREMAD_GENERATION / grounded_name, staging / "concept_sets" / "cremad" / f"{variant}.txt")
        records.append(model_record("cremad", run, destination, staging))

    provenance_sources = {
        "cremad_generation_manifest.json": CREMAD_GENERATION / "manifest.json",
        "cremad_concepts_metadata.json": CREMAD_GENERATION.parent / "concepts_metadata.json",
        "esc50_concepts_metadata.json": REPO_ROOT / "data/concept_sets/esc50/concepts_metadata.json",
        "urbansound8k_concepts_metadata.json": REPO_ROOT / "data/concept_sets/urbansound8k/concepts_metadata.json",
    }
    for name, source in provenance_sources.items():
        copy_file(source, staging / "concept_sets" / "provenance" / name)

    report_sources = {
        "environmental_cbm_summary.json": ENVIRONMENTAL_SUMMARY,
        "environmental_comparison.csv": REPO_ROOT / "results/audio_concept_ablation/comparison.csv",
        "environmental_segmentation_summary.json": REPO_ROOT / "results/audio_concept_ablation/segmented/summary.json",
        "cremad_targeted_source_ablation.json": CREMAD_SUMMARY,
        "cremad_targeted_source_ablation.csv": CREMAD_SUMMARY.with_name("source_ablation.csv"),
        "cremad_targeted_segmentation.json": REPO_ROOT / ("results/cremad_targeted_rerun_20260828/segmented/cremad/" "targeted_grounded_i40_l15/segmented_metrics.json"),
    }
    for name, source in report_sources.items():
        copy_file(source, staging / "reports" / name)

    (staging / "README.md").write_text("""# Audio LF-CBM external artifacts

This directory contains the canonical CBM checkpoints, exact concept inputs, and
compact evaluation reports used by the reported experiments. Upload the whole directory to
Google Drive without changing its internal paths.

The AST backbones are intentionally not duplicated here; download them from the
three Hugging Face repositories listed in the project README. Dataset audio and
cached AST/CLAP activations are also excluded because the repository scripts can
download or regenerate them.

After download, run:

    python -m scripts.release.build_google_drive_bundle --output /path/to/bundle --verify-only

Each checkpoint directory contains the concept projection (`W_c.pt`), sparse
classifier (`W_g.pt`, `b_g.pt`), normalization tensors, retained concept order,
training arguments, and metrics. `manifest.json` records all sizes and SHA-256
checksums.
""", encoding="utf-8")
    return records


def verify_bundle(root: Path) -> None:
    manifest = read_json(root / "manifest.json")
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing: {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            failures.append(f"size mismatch: {item['path']}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"checksum mismatch: {item['path']}")
    if failures:
        raise RuntimeError("Bundle verification failed:\n" + "\n".join(failures))
    print(f"Verified {len(manifest['files'])} files ({manifest['total_size_bytes'] / 1024**2:.1f} MiB) in {root}")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if args.verify_only:
        verify_bundle(output)
        return
    if output.exists() and not args.force:
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio_lf_cbm_bundle_", dir=output.parent) as tmp:
        staging = Path(tmp) / output.name
        staging.mkdir()
        models = build_bundle(staging)
        files = manifest_files(staging)
        manifest = {
            "format_version": 1,
            "description": "Canonical artifacts for Audio LF-CBM experiment results",
            "models": models,
            "files": files,
            "total_size_bytes": sum(item["size_bytes"] for item in files),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        shutil.copytree(staging, output)
    verify_bundle(output)


if __name__ == "__main__":
    main()
