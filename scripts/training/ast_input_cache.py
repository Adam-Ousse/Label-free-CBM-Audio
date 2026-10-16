"""Exact CPU AST input caches and cached-input datasets.

The cache stores the output of the established AST feature extractor after the
existing audio decode, resample, mono conversion, and fixed-duration padding
path.  It is deliberately a CPU-side execution optimization: model inputs,
labels, ordering, and all training hyperparameters remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

import data_utils


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _batch_to_inputs(batch: dict[str, Any], feature_extractor: Any) -> torch.Tensor:
    audio = batch["audio"]
    if audio.dim() == 3 and audio.shape[1] == 1:
        audio = audio.squeeze(1)
    elif audio.dim() != 2:
        raise ValueError("Unexpected audio batch shape: {}".format(tuple(audio.shape)))

    waveforms = [audio[index].cpu().numpy() for index in range(audio.shape[0])]
    sample_rates = batch["sr"]
    if isinstance(sample_rates, torch.Tensor):
        sample_rates = sample_rates.cpu().tolist()
    unique_sr = sorted(set(int(sr) for sr in sample_rates))
    if len(unique_sr) != 1:
        raise ValueError("Mixed sample rates in batch: {}".format(unique_sr))

    inputs = feature_extractor(
        waveforms,
        sampling_rate=int(unique_sr[0]),
        return_tensors="pt",
        padding=True,
    )
    values = inputs.get("input_values")
    if not isinstance(values, torch.Tensor):
        raise ValueError("AST feature extractor did not return input_values")
    return values.detach().cpu().contiguous().float()


def build_cache(
    *,
    dataset: str,
    manifest: Path,
    output: Path,
    base_model_id: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    if output.exists():
        try:
            payload = torch.load(output, map_location="cpu")
            values = payload["input_values"]
            rows = _read_rows(manifest)
            if tuple(values.shape) == (len(rows), 1024, 128):
                print("reusing verified cache: {}".format(output), flush=True)
                return payload["metadata"]
        except Exception:
            pass
        raise FileExistsError("Existing AST input cache is invalid; refusing to overwrite: {}".format(output))

    rows = _read_rows(manifest)
    if not rows:
        raise ValueError("Manifest is empty: {}".format(manifest))
    ids = [str(row.get("id", index)) for index, row in enumerate(rows)]
    if len(set(ids)) != len(ids):
        raise ValueError("Manifest contains duplicate sample IDs: {}".format(manifest))

    dataset_obj = data_utils.get_audio_dataset(dataset, split="cache", manifest_path=str(manifest))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": False,
        "collate_fn": data_utils.collate_audio_batch,
    }
    if int(num_workers) > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset_obj, **loader_kwargs)
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_id)

    cached_values: torch.Tensor | None = None
    cached_labels = torch.empty(len(rows), dtype=torch.long)
    cursor = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            values = _batch_to_inputs(batch, feature_extractor)
            if values.ndim != 3 or tuple(values.shape[1:]) != (1024, 128):
                raise ValueError("Unexpected AST input shape: {}".format(tuple(values.shape)))
            if cached_values is None:
                cached_values = torch.empty((len(rows), 1024, 128), dtype=torch.float32)
            end = cursor + int(values.shape[0])
            cached_values[cursor:end].copy_(values)
            cached_labels[cursor:end].copy_(batch["target"].long().cpu())
            cursor = end
            if batch_number == 1 or batch_number % 25 == 0 or cursor == len(rows):
                print("cached {}/{} samples".format(cursor, len(rows)), flush=True)

    if cached_values is None or cursor != len(rows):
        raise RuntimeError("Cache construction ended at {}/{} samples".format(cursor, len(rows)))

    metadata = {
        "dataset": dataset,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256_file(manifest),
        "base_model_id": base_model_id,
        "feature_extractor": "AutoFeatureExtractor output input_values",
        "shape": list(cached_values.shape),
        "dtype": str(cached_values.dtype).replace("torch.", ""),
        "sample_count": len(rows),
        "ids_sha256": _ids_digest(ids),
    }
    payload = {
        "input_values": cached_values,
        "labels": cached_labels,
        "ids": ids,
        "metadata": metadata,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-{}".format(os.getpid()))
    torch.save(payload, temporary)
    os.replace(temporary, output)
    print("wrote exact AST input cache: {}".format(output), flush=True)
    return metadata


class CachedASTInputDataset(Dataset):
    """Manifest-ordered view over one global exact AST-input cache."""

    def __init__(self, cache_path: str | Path, manifest_path: str | Path):
        payload = torch.load(cache_path, map_location="cpu")
        values = payload.get("input_values")
        labels = payload.get("labels")
        ids = payload.get("ids")
        if not isinstance(values, torch.Tensor) or tuple(values.shape[1:]) != (1024, 128):
            raise ValueError("Invalid cached AST input tensor: {}".format(cache_path))
        if not isinstance(labels, torch.Tensor) or not isinstance(ids, list):
            raise ValueError("Invalid cached AST metadata: {}".format(cache_path))
        if len(ids) != values.shape[0] or labels.shape[0] != values.shape[0]:
            raise ValueError("Cached AST input metadata length mismatch: {}".format(cache_path))
        by_id = {str(sample_id): index for index, sample_id in enumerate(ids)}
        rows = _read_rows(Path(manifest_path))
        self.indices: list[int] = []
        self.labels: list[int] = []
        self.ids: list[str] = []
        for row in rows:
            sample_id = str(row["id"])
            if sample_id not in by_id:
                raise ValueError("Manifest sample missing from AST cache: {}".format(sample_id))
            cache_index = by_id[sample_id]
            label = int(row["label_idx"])
            cached_label = int(labels[cache_index].item())
            if cached_label != label:
                raise ValueError("AST cache label mismatch for {}".format(sample_id))
            self.indices.append(cache_index)
            self.labels.append(label)
            self.ids.append(sample_id)
        self.input_values = values.float().contiguous()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache_index = self.indices[index]
        return {
            "id": self.ids[index],
            "input_values": self.input_values[cache_index],
            "target": self.labels[index],
        }


def collate_cached_ast_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "input_values": torch.stack([item["input_values"] for item in batch], dim=0),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("esc50", "urbansound8k"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model-id", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    metadata = build_cache(
        dataset=args.dataset,
        manifest=args.manifest,
        output=args.output,
        base_model_id=args.base_model_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
