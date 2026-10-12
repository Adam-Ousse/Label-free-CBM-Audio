#!/usr/bin/env python3
"""Compute CPU descriptor caches for the revision test clips.

The default inventory is the existing CREMA-D test split and UrbanSound8K
fold-10 test split.  Existing cache files are never replaced; ``--resume``
allows already-complete items to be reused.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .audio_descriptors import (
    DESCRIPTOR_NAMES,
    compute_librosa_descriptors,
    compute_opensmile_egemaps,
    descriptor_vector,
    iter_fixed_windows,
    load_audio,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_id(row: dict[str, Any]) -> str:
    dataset = str(row["dataset"])
    return f"{dataset}__{row['id']}"


def _write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing asset: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing asset: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_one(
    row: dict[str, Any],
    repo_root: Path,
    output_root: Path,
    *,
    include_egemaps: bool,
    window_hop: float,
    resume: bool,
) -> dict[str, Any]:
    audio_path = repo_root / row["audio_path"]
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    item_id = _stable_id(row)
    dataset = str(row["dataset"])
    clip_path = output_root / "clips" / dataset / f"{item_id}.npz"
    window_path = output_root / "windows" / dataset / f"{item_id}.npz"
    if not resume and (clip_path.exists() or window_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing cache for {item_id}")
    if resume and clip_path.exists() and window_path.exists():
        with np.load(window_path, allow_pickle=False) as cached:
            window_count = int(cached["values"].shape[0])
        return {
            **row,
            "cache_id": item_id,
            "audio_sha256": _sha256(audio_path),
            "sample_rate_used": int(row.get("sample_rate", 0)),
            "clip_cache": str(clip_path.relative_to(output_root)),
            "window_cache": str(window_path.relative_to(output_root)),
            "window_count": window_count,
            "window_hop_seconds": window_hop,
            "has_egemaps": include_egemaps,
        }

    y, sr = load_audio(audio_path)
    librosa_values = compute_librosa_descriptors(y, sr)
    egemaps_values = compute_opensmile_egemaps(y, sr) if include_egemaps else {}
    clip_values = {**librosa_values, **egemaps_values}
    _write_npz(
        clip_path,
        values=descriptor_vector(clip_values),
        names=np.asarray(list(DESCRIPTOR_NAMES), dtype=str),
        egemaps_names=np.asarray(list(egemaps_values), dtype=str),
        egemaps_values=np.asarray(list(egemaps_values.values()), dtype=np.float32),
    )

    starts: list[float] = []
    ends: list[float] = []
    window_vectors: list[np.ndarray] = []
    window_egemaps: list[np.ndarray] = []
    egemap_names = list(egemaps_values)
    for start, end, segment in iter_fixed_windows(y, sr, hop_seconds=window_hop):
        starts.append(start)
        ends.append(end)
        values = compute_librosa_descriptors(segment, sr)
        window_vectors.append(descriptor_vector(values))
        if include_egemaps:
            window_values = compute_opensmile_egemaps(segment, sr)
            window_egemaps.append(np.asarray([window_values[name] for name in egemap_names], dtype=np.float32))
    _write_npz(
        window_path,
        values=np.stack(window_vectors) if window_vectors else np.empty((0, len(DESCRIPTOR_NAMES)), dtype=np.float32),
        starts=np.asarray(starts, dtype=np.float32),
        ends=np.asarray(ends, dtype=np.float32),
        names=np.asarray(list(DESCRIPTOR_NAMES), dtype=str),
        egemaps_values=np.stack(window_egemaps) if window_egemaps else np.empty((0, len(egemap_names)), dtype=np.float32),
        egemaps_names=np.asarray(egemap_names, dtype=str),
    )
    return {
        **row,
        "cache_id": item_id,
        "audio_sha256": _sha256(audio_path),
        "sample_rate_used": sr,
        "clip_cache": str(clip_path.relative_to(output_root)),
        "window_cache": str(window_path.relative_to(output_root)),
        "window_count": len(starts),
        "window_hop_seconds": window_hop,
        "has_egemaps": include_egemaps,
    }


def _work_item(args: tuple[dict[str, Any], str, str, bool, float, bool]) -> dict[str, Any]:
    row, repo_root, output_root, include_egemaps, window_hop, resume = args
    return _cache_one(
        row,
        Path(repo_root),
        Path(output_root),
        include_egemaps=include_egemaps,
        window_hop=window_hop,
        resume=resume,
    )


def build_cache(
    repo_root: Path,
    output_root: Path,
    *,
    resume: bool = False,
    window_hop: float = 0.5,
    include_egemaps: bool = True,
    workers: int = 1,
) -> dict[str, Any]:
    inventories = {
        "esc50": repo_root / "data/esc50/manifests/test.jsonl",
        "cremad": repo_root / "data/cremad/manifests/test.jsonl",
        "urbansound8k": repo_root / "data/urbansound8k/manifests/fold10_test.jsonl",
    }
    rows = []
    for path in inventories.values():
        rows.extend(_read_jsonl(path))
    output_root.mkdir(parents=True, exist_ok=True)
    if int(workers) <= 1:
        cached_rows = [
            _work_item(
                (
                    row,
                    str(repo_root),
                    str(output_root),
                    include_egemaps and row["dataset"] == "cremad",
                    window_hop,
                    resume,
                )
            )
            for row in rows
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(workers)) as executor:
            jobs = [
                (
                    row,
                    str(repo_root),
                    str(output_root),
                    include_egemaps and row["dataset"] == "cremad",
                    window_hop,
                    resume,
                )
                for row in rows
            ]
            cached_rows = list(executor.map(_work_item, jobs))
    manifest_path = output_root / "cache_manifest.jsonl"
    if not resume or not manifest_path.exists():
        _write_jsonl(manifest_path, cached_rows)
    summary = {
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "datasets": {dataset: sum(row["dataset"] == dataset for row in cached_rows) for dataset in inventories},
        "clip_cache_count": len(cached_rows),
        "window_cache_count": sum(int(row["window_count"]) for row in cached_rows),
        "window_seconds": 1.0,
        "window_hop_seconds": window_hop,
        "opensmile": {"feature_set": "eGeMAPSv02", "dataset": "cremad" if include_egemaps else None},
        "source_manifests": {k: str(v.relative_to(repo_root)) for k, v in inventories.items()},
    }
    summary_path = output_root / "summary.json"
    if not resume or not summary_path.exists():
        _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--output-root", type=Path, default=Path("results/revision/t5_descriptors"))
    parser.add_argument("--window-hop", type=float, default=0.5)
    parser.add_argument("--no-egemaps", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    summary = build_cache(
        args.repo_root.resolve(),
        args.output_root.resolve(),
        resume=args.resume,
        window_hop=args.window_hop,
        include_egemaps=not args.no_egemaps,
        workers=args.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
