#!/usr/bin/env python3
"""Inference-only Task-5 controlled acoustic edits.

The runner reuses the retained segmented AST backbones and CBM checkpoints.
It never fits a model.  Original model activations are checked against the
cached temporal bundles before edited activations are written.  Audio edits
are applied to the existing held-out/test manifests only.

The controlled-edit protocol is intentionally explicit here because the
preparation README specified the edit magnitudes but did not contain an
inference implementation.  Pitch shifting and time stretching use librosa;
the low-pass uses a fourth-order zero-phase Butterworth filter; gain is a
linear amplitude multiplier; and reverb uses a deterministic exponentially
decaying synthetic impulse response whose envelope is -60 dB at RT60.
Edited model inputs are returned to the established dataset duration after
the edit, while descriptor measurements retain the natural edited duration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.stats import rankdata
from torch.utils.data import DataLoader

import data_utils
from features.audio_descriptors import (
    DESCRIPTOR_NAMES,
    compute_librosa_descriptors,
    compute_opensmile_egemaps,
    descriptor_vector,
    load_audio,
)
from scripts.evaluation.run_segmented_audio_ablation import split_to_segments
from scripts.evaluation.task5_descriptor_analysis import (
    CONTROLLED_EDIT_SPECS,
    aggregate_controlled_edits,
    bootstrap_mean_ci,
    bootstrap_spearman_ci,
    load_descriptor_map,
    plot_controlled_edit_figure,
    random_concept_descriptor_null,
    spearman_rho,
)
from models.ast_backbone import build_ast_backbone


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/revision/t5_descriptors/controlled_edits"
DESCRIPTOR_ROOT = ROOT / "results/revision/t5_descriptors"
CONTROLLED_ARTIFACT_ROOT = OUTPUT
MAP_PATH = DESCRIPTOR_ROOT / "concept_descriptor_map.json"
TEMPORAL_ROOT = ROOT / "results/revision/t0_baseline/segmented"
CHECKPOINT_ROOT = ROOT / "release/google_drive_bundle/checkpoints"
SEED = 42
WINDOW_SEC = 1.0
HOP_SEC = 0.5
DATASETS: dict[str, dict[str, Any]] = {
    "esc50": {"split": "fold1_test", "backbone": "ast_esc50", "duration_sec": 5.0, "manifest": "data/esc50/manifests/test.jsonl"},
    "urbansound8k": {"split": "fold10_test", "backbone": "ast_urbansound8k", "duration_sec": 4.0, "manifest": "data/urbansound8k/manifests/fold10_test.jsonl"},
    "cremad": {"split": "test", "backbone": "ast_hf__Adam-ousse__ast-cremad-finetuned", "duration_sec": 4.0, "manifest": "data/cremad/manifests/test.jsonl"},
}
VARIANTS = ("lf", "lf_broad", "lf_contrastive", "full")


class EgemapsRunner:
    """Reuse one exact eGeMAPSv02 OpenSMILE object for all CREMA clips."""

    def __init__(self) -> None:
        import opensmile

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

    def __call__(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        frame = self.smile.process_signal(np.asarray(audio, dtype=np.float32), int(sample_rate))
        if frame.shape[0] != 1:
            raise ValueError(f"Expected one OpenSMILE functional row, got shape {frame.shape}")
        return {str(key): float(value) for key, value in frame.iloc[0].items()}


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tensor_numpy(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    """Convert an inference tensor safely without changing its numerical values."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def torch_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fix_length(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size >= length:
        return audio[:length].copy()
    return np.pad(audio, (0, length - audio.size)).astype(np.float32, copy=False)


def stable_rng(dataset: str, sample_id: str, edit: str) -> np.random.Generator:
    token = f"{SEED}:{dataset}:{sample_id}:{edit}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(value)


def synthetic_reverb(audio: np.ndarray, sample_rate: int, rt60: float, rng: np.random.Generator) -> np.ndarray:
    """Apply deterministic dry+wet synthetic reverb with a specified RT60."""

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    ir_length = max(2, int(round(float(sample_rate) * float(rt60))) + 1)
    times = np.arange(ir_length, dtype=np.float64) / float(sample_rate)
    envelope = 10.0 ** (-3.0 * times / float(rt60))
    impulse = rng.standard_normal(ir_length).astype(np.float64) * envelope
    impulse[0] += 1.0
    impulse /= max(float(np.linalg.norm(impulse)), 1e-12)
    wet = signal.fftconvolve(y.astype(np.float64), impulse, mode="full")
    dry_rms = float(np.sqrt(np.mean(np.square(y.astype(np.float64)))))
    wet_rms = float(np.sqrt(np.mean(np.square(wet))))
    if wet_rms > 0.0 and dry_rms > 0.0:
        wet *= dry_rms / wet_rms
    # A fixed 35% wet mix makes the edit audible while retaining the source.
    result = np.concatenate([y.astype(np.float64), np.zeros(ir_length - 1)]) * 0.65
    result[: wet.size] += wet * 0.35
    return np.nan_to_num(result.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def apply_edit(audio: np.ndarray, sample_rate: int, edit: Mapping[str, Any], *, dataset: str, sample_id: str, target_length: int | None = None) -> np.ndarray:
    """Apply one preregistered edit, optionally returning a model-length clip."""

    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    family = str(edit["family"])
    if family == "pitch_shift":
        output = librosa.effects.pitch_shift(y=y, sr=int(sample_rate), n_steps=float(edit["semitones"]))
    elif family == "low_pass":
        cutoff = float(edit["cutoff_hz"])
        sos = signal.butter(4, cutoff, btype="lowpass", fs=float(sample_rate), output="sos")
        try:
            output = signal.sosfiltfilt(sos, y).astype(np.float32)
        except ValueError:
            output = signal.sosfilt(sos, y).astype(np.float32)
    elif family == "gain":
        output = y * float(10.0 ** (float(edit["gain_db"]) / 20.0))
    elif family == "reverb":
        output = synthetic_reverb(y, int(sample_rate), float(edit["rt60_seconds"]), stable_rng(dataset, sample_id, str(edit["name"])))
    elif family == "time_stretch":
        rate = float(edit["rate"])
        try:
            output = librosa.effects.time_stretch(y, rate=rate)
        except librosa.util.exceptions.ParameterError as exc:
            # Some librosa versions let np.arange include the STFT endpoint
            # for particular frame counts/rates. Reuse the same phase-vocoder
            # definition while clipping that numerically invalid endpoint.
            if "t_out values must be in the range" not in str(exc):
                raise
            stft = librosa.stft(y)
            t_out = np.arange(0.0, float(stft.shape[-1]), rate)
            t_out = np.minimum(t_out, np.nextafter(float(stft.shape[-1]), -np.inf))
            stretched = librosa.phase_vocoder(stft, t_out=t_out)
            output = librosa.istft(
                stretched,
                dtype=y.dtype,
                length=round(y.shape[-1] / rate),
            )
    else:
        raise ValueError(f"unknown controlled edit family: {family}")
    output = np.nan_to_num(np.asarray(output, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if target_length is not None:
        output = fix_length(output, int(target_length))
    return output


def edit_validation() -> dict[str, Any]:
    """Technical checks on synthetic signals, independent of model outputs."""

    sample_rate = 16000
    duration = 2.0
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    sine = 0.6 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    plus = apply_edit(sine, sample_rate, CONTROLLED_EDIT_SPECS[1], dataset="validation", sample_id="sine", target_length=None)
    minus = apply_edit(sine, sample_rate, CONTROLLED_EDIT_SPECS[0], dataset="validation", sample_id="sine", target_length=None)
    plus_f0 = float(np.nanmedian(librosa.yin(plus, fmin=200, fmax=1000, sr=sample_rate)))
    minus_f0 = float(np.nanmedian(librosa.yin(minus, fmin=100, fmax=800, sr=sample_rate)))
    original_rms = float(np.sqrt(np.mean(np.square(sine))))
    gain = apply_edit(sine, sample_rate, CONTROLLED_EDIT_SPECS[3], dataset="validation", sample_id="sine", target_length=None)
    gain_ratio = float(np.sqrt(np.mean(np.square(gain))) / original_rms)
    lowpass = apply_edit(sine + 0.4 * np.sin(2 * np.pi * 6000 * t), sample_rate, CONTROLLED_EDIT_SPECS[2], dataset="validation", sample_id="sine", target_length=None)
    fft_original = np.abs(np.fft.rfft(sine + 0.4 * np.sin(2 * np.pi * 6000 * t))) ** 2
    fft_lowpass = np.abs(np.fft.rfft(lowpass)) ** 2
    frequencies = np.fft.rfftfreq(sine.size, 1.0 / sample_rate)
    high = frequencies > 2000.0
    high_suppression_db = float(10.0 * np.log10((fft_lowpass[high].sum() + 1e-12) / (fft_original[high].sum() + 1e-12)))
    reverb = apply_edit(sine, sample_rate, CONTROLLED_EDIT_SPECS[4], dataset="validation", sample_id="sine", target_length=None)
    stretch = apply_edit(sine, sample_rate, CONTROLLED_EDIT_SPECS[5], dataset="validation", sample_id="sine", target_length=None)
    result = {
        "sample_rate": sample_rate,
        "pitch_original_hz": 440.0,
        "pitch_plus4_hz": plus_f0,
        "pitch_minus4_hz": minus_f0,
        "pitch_plus4_expected_hz": 440.0 * 2.0 ** (4.0 / 12.0),
        "pitch_minus4_expected_hz": 440.0 * 2.0 ** (-4.0 / 12.0),
        "gain_rms_ratio": gain_ratio,
        "gain_expected_ratio": 0.1,
        "lowpass_high_band_suppression_db": high_suppression_db,
        "reverb_input_samples": int(sine.size),
        "reverb_output_samples": int(reverb.size),
        "stretch_input_samples": int(sine.size),
        "stretch_output_samples": int(stretch.size),
        "stretch_expected_output_samples": int(round(sine.size / 0.7)),
        "finite": bool(np.isfinite([plus_f0, minus_f0, gain_ratio, high_suppression_db]).all()),
    }
    checks = {
        "pitch_direction": plus_f0 > 440.0 and minus_f0 < 440.0,
        "gain_minus20db": abs(gain_ratio - 0.1) < 1e-5,
        "lowpass_above2khz_suppressed": high_suppression_db < -20.0,
        "reverb_adds_tail": reverb.size > sine.size,
        "stretch_rate_0p7": abs(stretch.size / sine.size - (1.0 / 0.7)) < 0.01,
    }
    result["checks"] = checks
    result["all_checks_pass"] = bool(all(checks.values()))
    if not result["all_checks_pass"]:
        raise RuntimeError(f"controlled-edit technical validation failed: {result}")
    return result


def load_cbm_variant(dataset: str, variant: str, device: str) -> dict[str, Any]:
    root = CHECKPOINT_ROOT / dataset / variant
    required = [root / name for name in ("W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing canonical CBM files: {missing}")
    tensors = {name: torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True).float().to(device) for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")}
    concepts = [line.strip() for line in (root / "concepts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if tensors["W_c"].shape[0] != len(concepts) or tensors["W_g"].shape[1] != len(concepts):
        raise ValueError(f"CBM concept order/count mismatch for {dataset}/{variant}")
    return {"dataset": dataset, "variant": variant, "root": root, "tensors": tensors, "concepts": concepts, "checkpoint_sha256": sha256(root / "W_c.pt")}


def project_features(features: torch.Tensor, model: Mapping[str, Any]) -> torch.Tensor:
    tensors = model["tensors"]
    concept = F.linear(features.float(), tensors["W_c"])
    return (concept - tensors["proj_mean"]) / tensors["proj_std"].clamp_min(1e-12)


def raw_descriptor_vector(audio: np.ndarray, sample_rate: int, dataset: str, include_egemaps: bool, reference_names: list[str], egemaps_runner: EgemapsRunner | None = None) -> tuple[dict[str, float], list[str]]:
    values = compute_librosa_descriptors(audio, int(sample_rate))
    if include_egemaps:
        if egemaps_runner is None:
            values.update(compute_opensmile_egemaps(audio, int(sample_rate)))
        else:
            values.update(egemaps_runner(audio, int(sample_rate)))
    names = list(reference_names)
    return values, names


def load_original_descriptor_values(dataset: str, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    manifest = {}
    for line in (DESCRIPTOR_ROOT / "cache_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if str(row["dataset"]) == dataset:
            manifest[str(row["id"])] = row
    rows, names = [], None
    for sample_id in sample_ids:
        row = manifest[sample_id]
        with np.load(DESCRIPTOR_ROOT / str(row["clip_cache"]), allow_pickle=False) as archive:
            if names is None:
                names = list(archive["names"].astype(str)) + list(archive.get("egemaps_names", np.asarray([], dtype=str)).astype(str))
            values = list(archive["values"].astype(np.float64)) + list(archive.get("egemaps_values", np.asarray([], dtype=np.float64)).astype(np.float64))
            rows.append(values)
    if names is None:
        raise ValueError(f"no descriptor caches for {dataset}")
    return np.asarray(rows, dtype=np.float64), names


def infer_backbone(backbone: torch.nn.Module, audio: torch.Tensor, sample_rate: int, device: str, segment_batch_size: int) -> torch.Tensor:
    segments, _ = split_to_segments(audio, sample_rate, WINDOW_SEC, HOP_SEC)
    flat = segments.reshape(-1, segments.shape[-1])
    chunks = []
    with torch.inference_mode():
        for start in range(0, flat.shape[0], int(segment_batch_size)):
            current = flat[start : start + int(segment_batch_size)].to(device)
            rates = torch.full((current.shape[0],), int(sample_rate), dtype=torch.long, device=device)
            chunks.append(backbone(current, sample_rates=rates).detach())
    return torch.cat(chunks, dim=0).reshape(segments.shape[0], segments.shape[1], -1)


def existing_temporal(dataset: str, variant: str) -> tuple[torch.Tensor, list[str], list[str]]:
    path = TEMPORAL_ROOT / dataset / variant / "test_temporal_concepts.pt"
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    values = bundle["temporal_concepts"].detach().cpu().float()
    return values, [str(x) for x in bundle["sample_ids"]], [str(x) for x in bundle["concepts"]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_ids(value: Any) -> list[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(item) for item in value]


def reconcile_existing_artifacts(datasets: Sequence[str]) -> dict[str, Any]:
    """Audit source/cache/inference IDs and explain natural-vs-fixed windows."""

    report: dict[str, Any] = {
        "status": "valid_expected_protocol_difference",
        "classification": "A_expected_by_protocol",
        "summary": (
            "UrbanSound8K descriptor windows are natural-duration padded windows; "
            "AST/CBM windows are seven fixed windows after 4-second model-input normalization. "
            "Final Task-5 pairings are clip-level and use descriptor clip caches."
        ),
        "datasets": {},
    }
    for dataset in datasets:
        source_rows = _read_jsonl(ROOT / str(DATASETS[dataset]["manifest"]))
        source_ids = [str(row["id"]) for row in source_rows]
        cache_rows = [
            json.loads(line)
            for line in (DESCRIPTOR_ROOT / "cache_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["dataset"] == dataset
        ]
        cache_ids = [str(row["id"]) for row in cache_rows]
        cache_windows = []
        cache_window_index_checks = []
        for row in cache_rows:
            with np.load(DESCRIPTOR_ROOT / str(row["window_cache"]), allow_pickle=False) as archive:
                count = int(archive["values"].shape[0])
                starts = np.asarray(archive["starts"])
                ends = np.asarray(archive["ends"])
                cache_windows.append(count)
                cache_window_index_checks.append(
                    count == int(row["window_count"])
                    and starts.shape == (count,)
                    and ends.shape == (count,)
                    and bool(np.all(np.isfinite(starts)))
                    and bool(np.all(np.isfinite(ends)))
                )

        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            baseline, baseline_ids, baseline_concepts = existing_temporal(dataset, variant)
            controlled_rows: dict[str, Any] = {}
            for edit in CONTROLLED_EDIT_SPECS:
                edit_name = str(edit["name"])
                artifact = CONTROLLED_ARTIFACT_ROOT / dataset / variant / f"{edit_name}.pt"
                bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                controlled_ids = _artifact_ids(bundle["sample_ids"])
                temporal_delta = tensor_numpy(bundle["temporal_concept_changes"], dtype=np.float32)
                concept_delta = tensor_numpy(bundle["concept_changes"], dtype=np.float64)
                controlled_rows[edit_name] = {
                    "artifact": str(artifact),
                    "sample_count": len(controlled_ids),
                    "ids_match_source": controlled_ids == source_ids,
                    "duplicate_ids": len(controlled_ids) - len(set(controlled_ids)),
                    "temporal_shape": list(temporal_delta.shape),
                    "concept_shape": list(concept_delta.shape),
                    "finite_concept_change": bool(np.isfinite(concept_delta).all()),
                }
            variants[variant] = {
                "baseline_sample_count": len(baseline_ids),
                "baseline_ids_match_source": baseline_ids == source_ids,
                "baseline_window_count": int(baseline.shape[1]),
                "concept_count": len(baseline_concepts),
                "controlled": controlled_rows,
            }

        descriptor_edits: dict[str, Any] = {}
        for edit in CONTROLLED_EDIT_SPECS:
            edit_name = str(edit["name"])
            path = CONTROLLED_ARTIFACT_ROOT / dataset / f"{edit_name}_descriptors.npz"
            with np.load(path, allow_pickle=False) as archive:
                ids = _artifact_ids(archive["sample_ids"])
                descriptor_edits[edit_name] = {
                    "sample_count": len(ids),
                    "ids_match_source": ids == source_ids,
                    "duplicate_ids": len(ids) - len(set(ids)),
                    "shape": list(archive["changes"].shape),
                    "names": [str(x) for x in archive["names"].tolist()],
                    "undefined_descriptor_values_preserved": bool(np.isnan(archive["changes"]).any()),
                }

        missing_audio = [str(row["audio_path"]) for row in source_rows if not (ROOT / str(row["audio_path"])).is_file()]
        dataset_report = {
            "source_manifest": str(ROOT / str(DATASETS[dataset]["manifest"])),
            "source_clip_count": len(source_ids),
            "source_unique_clip_count": len(set(source_ids)),
            "source_duplicate_clip_count": len(source_ids) - len(set(source_ids)),
            "source_fold_or_split": sorted({str(row.get("fold", row.get("split", ""))) for row in source_rows}),
            "cache_clip_count": len(cache_rows),
            "cache_ids_match_source": cache_ids == source_ids,
            "cache_missing_ids": sorted(set(source_ids) - set(cache_ids)),
            "cache_extra_ids": sorted(set(cache_ids) - set(source_ids)),
            "cache_duplicate_ids": len(cache_ids) - len(set(cache_ids)),
            "cache_window_count": int(sum(cache_windows)),
            "cache_window_count_by_clip": {str(k): int(v) for k, v in __import__("collections").Counter(cache_windows).items()},
            "cache_window_indices_valid": bool(all(cache_window_index_checks)),
            "missing_audio_records": missing_audio,
            "variants": variants,
            "descriptor_edits": descriptor_edits,
        }
        if dataset == "urbansound8k":
            dataset_report.update(
                {
                    "fixed_model_window_count_per_clip": 7,
                    "fixed_model_window_count_total": len(source_ids) * 7,
                    "window_count_explanation": (
                        "The descriptor cache iterates natural clip lengths with pad_end=True, "
                        "yielding 1--8 windows per clip (1553 total). The model path pads/truncates "
                        "to 4 seconds before split_to_segments, yielding 7 windows per clip (5859 total)."
                    ),
                    "final_pairing_uses_clip_level_descriptor_cache": True,
                }
            )
        report["datasets"][dataset] = dataset_report

    for dataset_report in report["datasets"].values():
        if dataset_report["missing_audio_records"]:
            raise ValueError(f"missing source audio records in Task-5 manifest: {dataset_report['missing_audio_records'][:3]}")
        if not dataset_report["cache_ids_match_source"] or dataset_report["cache_duplicate_ids"]:
            raise ValueError(f"descriptor cache/source alignment invalid for {dataset_report['source_manifest']}")
        for variant_report in dataset_report["variants"].values():
            if not variant_report["baseline_ids_match_source"]:
                raise ValueError("temporal baseline/source alignment invalid")
            for edit_report in variant_report["controlled"].values():
                if not edit_report["ids_match_source"] or edit_report["duplicate_ids"] or not edit_report["finite_concept_change"]:
                    raise ValueError("controlled-edit/source alignment or finiteness check failed")
        for edit_report in dataset_report["descriptor_edits"].values():
            if not edit_report["ids_match_source"] or edit_report["duplicate_ids"]:
                raise ValueError("descriptor-edit/source alignment check failed")
    return report


def load_existing_dataset_artifacts(dataset: str) -> dict[str, Any]:
    """Load inference outputs and reconstruct clip-paired matrices on CPU only."""

    temporal_by_variant: dict[str, tuple[torch.Tensor, list[str], list[str]]] = {
        variant: existing_temporal(dataset, variant) for variant in VARIANTS
    }
    sample_ids = temporal_by_variant[VARIANTS[0]][1]
    for variant in VARIANTS:
        if temporal_by_variant[variant][1] != sample_ids:
            raise ValueError(f"baseline sample ordering mismatch for {dataset}/{variant}")
    descriptor_changes: dict[str, np.ndarray] = {}
    descriptor_names: list[str] | None = None
    descriptor_original: np.ndarray | None = None
    descriptor_edited: dict[str, np.ndarray] = {}
    for edit in CONTROLLED_EDIT_SPECS:
        edit_name = str(edit["name"])
        path = CONTROLLED_ARTIFACT_ROOT / dataset / f"{edit_name}_descriptors.npz"
        with np.load(path, allow_pickle=False) as archive:
            ids = _artifact_ids(archive["sample_ids"])
            if ids != sample_ids:
                raise ValueError(f"descriptor edit sample ordering mismatch for {dataset}/{edit_name}")
            names = [str(x) for x in archive["names"].tolist()]
            original = np.asarray(archive["original_values"], dtype=np.float64)
            edited = np.asarray(archive["edited_values"], dtype=np.float64)
            changes = np.asarray(archive["changes"], dtype=np.float64)
            if descriptor_names is None:
                descriptor_names = names
                descriptor_original = original
            if names != descriptor_names or original.shape != descriptor_original.shape:
                raise ValueError(f"descriptor layout mismatch for {dataset}/{edit_name}")
            # The three arrays were each serialized as float32.  At the largest
            # eGeMAPS values, recomputing edited-original after deserialization
            # can differ by a few 1e-3 solely through float32 quantization.
            if not np.allclose(edited - original, changes, equal_nan=True, atol=1e-2, rtol=1e-6):
                raise ValueError(f"stored descriptor change mismatch for {dataset}/{edit_name}")
            descriptor_changes[edit_name] = changes
            descriptor_edited[edit_name] = edited
    if descriptor_names is None or descriptor_original is None:
        raise ValueError(f"no descriptor edit artifacts for {dataset}")

    paired_paths: dict[tuple[str, str], Path] = {}
    for variant in VARIANTS:
        baseline_tensor, _, concepts = temporal_by_variant[variant]
        baseline_pooled = tensor_numpy(baseline_tensor.mean(dim=1), dtype=np.float64)
        for edit in CONTROLLED_EDIT_SPECS:
            edit_name = str(edit["name"])
            path = CONTROLLED_ARTIFACT_ROOT / dataset / variant / f"{edit_name}.pt"
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            ids = _artifact_ids(bundle["sample_ids"])
            artifact_concepts = [str(x) for x in bundle["concepts"]]
            temporal_delta = tensor_numpy(bundle["temporal_concept_changes"], dtype=np.float64)
            concept_delta = tensor_numpy(bundle["concept_changes"], dtype=np.float64)
            if ids != sample_ids or artifact_concepts != concepts:
                raise ValueError(f"controlled-edit provenance mismatch for {dataset}/{variant}/{edit_name}")
            if temporal_delta.shape != baseline_tensor.shape or concept_delta.shape != baseline_pooled.shape:
                raise ValueError(f"controlled-edit shape mismatch for {dataset}/{variant}/{edit_name}")
            pooled_from_temporal = np.mean(temporal_delta, axis=1)
            if not np.allclose(pooled_from_temporal, concept_delta, atol=2e-5, rtol=2e-5):
                raise ValueError(f"controlled-edit pooling mismatch for {dataset}/{variant}/{edit_name}")
            if not np.isfinite(concept_delta).all():
                raise FloatingPointError(f"non-finite concept change for {dataset}/{variant}/{edit_name}")
            edited_concepts = baseline_pooled + concept_delta
            output_path = OUTPUT / "paired_matrices" / dataset / variant / f"{edit_name}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                sample_ids=np.asarray(sample_ids),
                concept_names=np.asarray(concepts),
                original_concept_values=baseline_pooled.astype(np.float32),
                edited_concept_values=edited_concepts.astype(np.float32),
                concept_changes=concept_delta.astype(np.float32),
                descriptor_names=np.asarray(descriptor_names),
                original_descriptor_values=descriptor_original.astype(np.float32),
                edited_descriptor_values=descriptor_edited[edit_name].astype(np.float32),
                descriptor_changes=descriptor_changes[edit_name].astype(np.float32),
            )
            paired_paths[(variant, edit_name)] = output_path
    return {
        "dataset": dataset,
        "sample_ids": sample_ids,
        "descriptor_names": descriptor_names,
        "original_descriptor_values": descriptor_original,
        "descriptor_changes": descriptor_changes,
        "models": {variant: {"concepts": temporal_by_variant[variant][2]} for variant in VARIANTS},
        "paired_paths": paired_paths,
    }


def process_dataset(dataset: str, args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    spec = DATASETS[dataset]
    split = str(spec["split"])
    dataset_obj = data_utils.get_audio_dataset(dataset, split)
    loader = DataLoader(dataset_obj, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=data_utils.collate_audio_batch)
    device = str(args.device)
    backbone = build_ast_backbone(str(spec["backbone"]), device=device)
    models = {variant: load_cbm_variant(dataset, variant, device) for variant in VARIANTS}
    expected_temporal = {variant: existing_temporal(dataset, variant) for variant in VARIANTS}
    sample_ids = expected_temporal[VARIANTS[0]][1]
    manifest_ids = [str(row["id"]) for row in dataset_obj.samples]
    if len(sample_ids) != len(dataset_obj) or sample_ids != manifest_ids:
        raise ValueError(f"sample order mismatch for {dataset}")
    original_descriptor_values, descriptor_names = load_original_descriptor_values(dataset, sample_ids)
    include_egemaps = dataset == "cremad"
    egemaps_runner = EgemapsRunner() if include_egemaps else None
    edited_descriptor_values: dict[str, list[np.ndarray]] = {str(edit["name"]): [] for edit in CONTROLLED_EDIT_SPECS}
    original_chunks: dict[str, list[torch.Tensor]] = {variant: [] for variant in VARIANTS}
    edit_chunks: dict[tuple[str, str], list[torch.Tensor]] = {(variant, str(edit["name"])): [] for variant in VARIANTS for edit in CONTROLLED_EDIT_SPECS}
    processed = 0
    for batch in loader:
        batch_ids = [str(x) for x in batch["id"]]
        audio = batch["audio"].float()
        original_features = infer_backbone(backbone, audio, 16000, device, int(args.segment_batch_size))
        original_temporal: dict[str, torch.Tensor] = {}
        for variant in VARIANTS:
            original_temporal[variant] = project_features(original_features, models[variant])
            original_chunks[variant].append(original_temporal[variant].cpu())
        for edit in CONTROLLED_EDIT_SPECS:
            name = str(edit["name"])
            edited_model_audio = torch.stack([
                torch.from_numpy(apply_edit(audio[index].detach().cpu().numpy(), 16000, edit, dataset=dataset, sample_id=batch_ids[index], target_length=audio.shape[-1]))
                for index in range(audio.shape[0])
            ])
            edited_features = infer_backbone(backbone, edited_model_audio, 16000, device, int(args.segment_batch_size))
            for variant in VARIANTS:
                edited_temporal = project_features(edited_features, models[variant])
                delta = edited_temporal - original_temporal[variant]
                if not bool(torch.isfinite(delta).all()):
                    raise FloatingPointError(f"non-finite concept change in {dataset}/{variant}/{name}")
                edit_chunks[(variant, name)].append(delta.cpu().half())
            for index, sample_id in enumerate(batch_ids):
                row = dataset_obj.samples[processed + index]
                raw, raw_sr = load_audio(ROOT / str(row["audio_path"]))
                edited_raw = apply_edit(raw, raw_sr, edit, dataset=dataset, sample_id=sample_id, target_length=None)
                values, names = raw_descriptor_vector(edited_raw, raw_sr, dataset, include_egemaps, descriptor_names, egemaps_runner)
                if names != descriptor_names:
                    raise ValueError(f"descriptor names changed for {dataset}/{name}: {names} != {descriptor_names}")
                edited_descriptor_values[name].append(np.asarray([values.get(key, np.nan) for key in descriptor_names], dtype=np.float64))
        processed += len(batch_ids)
        print(f"{dataset}: processed {processed}/{len(dataset_obj)}", flush=True)
    original_full = {variant: torch.cat(original_chunks[variant], dim=0) for variant in VARIANTS}
    for variant in VARIANTS:
        expected, expected_ids, expected_names = expected_temporal[variant]
        if expected_ids != sample_ids or expected_names != models[variant]["concepts"]:
            raise ValueError(f"temporal bundle provenance mismatch for {dataset}/{variant}")
        max_abs = float((original_full[variant] - expected).abs().max())
        validation.setdefault("baseline_activation_max_abs_error", {})[f"{dataset}/{variant}"] = max_abs
        if max_abs > 0.02:
            raise ValueError(f"baseline activation mismatch for {dataset}/{variant}: {max_abs}")
    dataset_out = OUTPUT / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)
    descriptor_changes: dict[str, np.ndarray] = {}
    for edit in CONTROLLED_EDIT_SPECS:
        name = str(edit["name"])
        edited_values = np.asarray(edited_descriptor_values[name], dtype=np.float64)
        if edited_values.shape != original_descriptor_values.shape:
            raise ValueError(f"descriptor shape mismatch for {dataset}/{name}: {edited_values.shape}")
        changes = edited_values - original_descriptor_values
        np.savez_compressed(dataset_out / f"{name}_descriptors.npz", sample_ids=np.asarray(sample_ids), names=np.asarray(descriptor_names), original_values=original_descriptor_values.astype(np.float32), edited_values=edited_values.astype(np.float32), changes=changes.astype(np.float32))
        descriptor_changes[name] = changes
    for variant in VARIANTS:
        variant_out = dataset_out / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        for edit in CONTROLLED_EDIT_SPECS:
            name = str(edit["name"])
            temporal_delta = torch.cat(edit_chunks[(variant, name)], dim=0)
            pooled_delta = temporal_delta.float().mean(dim=1)
            torch_dump(variant_out / f"{name}.pt", {"dataset": dataset, "variant": variant, "edit": name, "sample_ids": sample_ids, "concepts": models[variant]["concepts"], "temporal_concept_changes": temporal_delta, "concept_changes": pooled_delta, "tensor_layout": ["sample", "window", "concept"], "pool": "mean", "checkpoint_dir": str(models[variant]["root"]), "checkpoint_sha256_W_c": models[variant]["checkpoint_sha256"]})
        json_dump(variant_out / "metadata.json", {"dataset": dataset, "variant": variant, "sample_count": len(sample_ids), "concept_count": len(models[variant]["concepts"]), "concepts": models[variant]["concepts"], "checkpoint_dir": str(models[variant]["root"]), "checkpoint_sha256_W_c": models[variant]["checkpoint_sha256"]})
    return {"dataset": dataset, "sample_ids": sample_ids, "descriptor_names": descriptor_names, "original_descriptor_values": original_descriptor_values, "descriptor_changes": descriptor_changes, "models": models}


def bootstrap_rho_ci(x: np.ndarray, y: np.ndarray, seed: int, draws: int) -> tuple[float, float] | None:
    """Vectorized equivalent of the established paired Spearman bootstrap."""

    left, right = np.asarray(x, dtype=np.float64).reshape(-1), np.asarray(y, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"paired inputs must have equal shape, got {left.shape} and {right.shape}")
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or draws <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, left.size, size=(int(draws), left.size))
    # rankdata(method='average') is the vectorized form of the established
    # tie-aware average_ranks implementation.  The resampling RNG, statistic,
    # undefined zero-variance handling, and percentile CI are unchanged.
    left_ranks = rankdata(left[indices], axis=1, method="average")
    right_ranks = rankdata(right[indices], axis=1, method="average")
    left_centered = left_ranks - left_ranks.mean(axis=1, keepdims=True)
    right_centered = right_ranks - right_ranks.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(left_centered, axis=1) * np.linalg.norm(right_centered, axis=1)
    values = np.divide(
        np.sum(left_centered * right_centered, axis=1),
        denominator,
        out=np.full(denominator.shape, np.nan, dtype=np.float64),
        where=denominator != 0.0,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    alpha = 0.025
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def make_final_analysis(dataset_results: Mapping[str, Any], args: argparse.Namespace, validation: dict[str, Any]) -> dict[str, Any]:
    descriptor_map = load_descriptor_map(MAP_PATH)
    descriptor_analysis_path = DESCRIPTOR_ROOT / "analysis.json"
    if not descriptor_analysis_path.is_file():
        raise FileNotFoundError(f"established descriptor analysis is missing: {descriptor_analysis_path}")
    descriptor_analysis = json.loads(descriptor_analysis_path.read_text(encoding="utf-8"))
    if int(len(descriptor_analysis.get("null_pairings", []))) != 12000:
        raise ValueError("established descriptor null analysis does not contain exactly 12,000 pairings")
    rows: list[dict[str, Any]] = []
    summary_batches: list[dict[str, Any]] = []
    for dataset, result in dataset_results.items():
        mappings = [item for item in descriptor_map["mappings"] if dataset in item["datasets"]]
        for variant in VARIANTS:
            concepts = result["models"][variant]["concepts"]
            concept_index = {name: index for index, name in enumerate(concepts)}
            for edit in CONTROLLED_EDIT_SPECS:
                edit_name = str(edit["name"])
                paired_path = Path(result["paired_paths"][(variant, edit_name)])
                with np.load(paired_path, allow_pickle=False) as paired:
                    concept_changes = np.asarray(paired["concept_changes"], dtype=np.float64)
                descriptor_changes = result["descriptor_changes"][edit_name]
                descriptor_index = {name: index for index, name in enumerate(result["descriptor_names"])}
                mapped_pairs = {(str(item["concept"]), str(item["descriptor"])) for item in mappings}
                null_rows = random_concept_descriptor_null(concept_changes, descriptor_changes, concepts, result["descriptor_names"], mapped_pairs, draws=int(args.null_draws), seed=SEED + len(rows) * 17)
                null_rhos = np.asarray([row["spearman_rho"] for row in null_rows if row["spearman_rho"] is not None], dtype=np.float64)
                for mapping in mappings:
                    concept = str(mapping["concept"])
                    descriptor = str(mapping["descriptor"])
                    if concept not in concept_index or descriptor not in descriptor_index:
                        continue
                    x = concept_changes[:, concept_index[concept]]
                    y = descriptor_changes[:, descriptor_index[descriptor]]
                    rho = spearman_rho(x, y)
                    rho_ci = bootstrap_rho_ci(x, y, SEED + len(rows) * 19, int(args.bootstrap_resamples))
                    effect_ci = bootstrap_mean_ci(x, seed=SEED + len(rows) * 23, draws=int(args.bootstrap_resamples))
                    descriptor_ci = bootstrap_mean_ci(y, seed=SEED + len(rows) * 29, draws=int(args.bootstrap_resamples))
                    finite = np.isfinite(x) & np.isfinite(y)
                    observed_sign = int(np.sign(np.nanmean(x[finite]))) if finite.any() and np.nanmean(x[finite]) != 0 else 0
                    expected_sign = int(mapping["expected_sign"])
                    null_p = float((np.sum(np.abs(null_rhos) >= abs(rho)) + 1) / (null_rhos.size + 1)) if rho is not None and null_rhos.size else None
                    rows.append({"dataset": dataset, "variant": variant, "edit": edit_name, "concept": concept, "descriptor": descriptor, "expected_sign": expected_sign, "observed_concept_change_mean": float(np.nanmean(x)), "observed_concept_change_ci_low": effect_ci[0] if effect_ci else None, "observed_concept_change_ci_high": effect_ci[1] if effect_ci else None, "descriptor_change_mean": float(np.nanmean(y)) if np.isfinite(y).any() else None, "descriptor_change_ci_low": descriptor_ci[0] if descriptor_ci else None, "descriptor_change_ci_high": descriptor_ci[1] if descriptor_ci else None, "observed_concept_sign": observed_sign, "concept_expected_sign_match": bool(observed_sign == expected_sign), "spearman_rho": rho, "spearman_ci_low": rho_ci[0] if rho_ci else None, "spearman_ci_high": rho_ci[1] if rho_ci else None, "null_pair_count": int(null_rhos.size), "null_abs_rho_p": null_p, "finite_pair_count": int(finite.sum()), "sample_count": int(x.size), "mapping_is_fixed": True})
                summary_batches.append({"dataset": dataset, "variant": variant, "edit": edit_name, "concept_names": concepts, "descriptor_names": result["descriptor_names"], "concept_changes": concept_changes, "descriptor_changes": descriptor_changes})
    summaries = aggregate_controlled_edits(summary_batches, descriptor_map, unmatched_draws=int(args.null_draws), seed=SEED)
    dataset_summaries: list[dict[str, Any]] = []
    for dataset in sorted(dataset_results):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        finite_rows = [row for row in dataset_rows if row["spearman_rho"] is not None]
        sign_rows = [row for row in dataset_rows if row["concept_expected_sign_match"] is not None]
        dataset_summaries.append(
            {
                "dataset": dataset,
                "clip_count": len(dataset_results[dataset]["sample_ids"]),
                "variant_count": len(VARIANTS),
                "intervention_count": len(CONTROLLED_EDIT_SPECS),
                "mapped_test_count": len(dataset_rows),
                "finite_mapped_correlation_count": len(finite_rows),
                "expected_sign_success_count": int(sum(bool(row["concept_expected_sign_match"]) for row in sign_rows)),
                "expected_sign_success_rate": float(sum(bool(row["concept_expected_sign_match"]) for row in sign_rows) / len(sign_rows)) if sign_rows else None,
                "all_activation_effects_finite": bool(
                    all(
                        np.isfinite(row["observed_concept_change_mean"])
                        and (row["observed_concept_change_ci_low"] is None or np.isfinite(row["observed_concept_change_ci_low"]))
                        and (row["observed_concept_change_ci_high"] is None or np.isfinite(row["observed_concept_change_ci_high"]))
                        for row in dataset_rows
                    )
                ),
            }
        )
    null_comparison = {
        "descriptor_analysis_null_pairing_count": int(len(descriptor_analysis["null_pairings"])),
        "descriptor_analysis_null_pairings_per_model": int(descriptor_analysis.get("null_pairings_per_model", 0)),
        "descriptor_analysis_null_definition": descriptor_analysis.get("null_definition"),
        "controlled_edit_null_draws_per_batch": int(args.null_draws),
        "controlled_edit_batches": len(summary_batches),
        "controlled_edit_unmatched_pair_count": int(sum(row["unmatched_pair_count"] for row in summaries)),
        "controlled_edit_summaries": summaries,
    }
    output = {
        "protocol": {"seed": SEED, "datasets": DATASETS, "variants": list(VARIANTS), "window_sec": WINDOW_SEC, "hop_sec": HOP_SEC, "edits": list(CONTROLLED_EDIT_SPECS), "model_checkpoint_root": str(CHECKPOINT_ROOT), "descriptor_root": str(DESCRIPTOR_ROOT), "descriptor_map": str(MAP_PATH), "descriptor_missing_values_preserved": True, "reverb_wet_mix": 0.35, "reverb_ir_envelope": "10^(-3*t/RT60)", "bootstrap_resamples": int(args.bootstrap_resamples), "null_draws": int(args.null_draws), "completion_mode": "cpu_only_artifact_postprocessing", "inference_rerun": False},
        "validation": validation,
        "descriptor_associations": descriptor_analysis["correlations"],
        "descriptor_null_pairings": descriptor_analysis["null_pairings"],
        "rows": rows,
        "intervention_effects": rows,
        "dataset_summaries": dataset_summaries,
        "null_comparison": null_comparison,
        "edit_summaries": summaries,
    }
    json_dump(OUTPUT / "controlled_edit_analysis.json", output)
    with (OUTPUT / "controlled_edit_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0]) if rows else ["dataset", "variant", "edit"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "descriptor_correlations.csv").open("w", encoding="utf-8", newline="") as handle:
        descriptor_rows = descriptor_analysis["correlations"]
        fields = list(descriptor_rows[0]) if descriptor_rows else ["dataset", "concept", "descriptor"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(descriptor_rows)
    with (OUTPUT / "dataset_summaries.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(dataset_summaries[0]) if dataset_summaries else ["dataset"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dataset_summaries)
    with (OUTPUT / "null_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "variant", "edit", "matched_pair_count", "unmatched_pair_count", "matched_mean_rho", "unmatched_mean_rho", "matched_minus_unmatched", "matched_ci", "unmatched_ci", "matched_minus_unmatched_ci"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field) for field in fields})
    with (OUTPUT / "controlled_edit_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llllrrrr}\n\\toprule\nDataset & Variant & Edit & Concept & $\\Delta c$ & 95\\% CI & $\\rho$ & Null $p$ \\\\\n\\midrule\n")
        for row in rows:
            ci = "[{:.3f}, {:.3f}]".format(row["observed_concept_change_ci_low"], row["observed_concept_change_ci_high"]) if row["observed_concept_change_ci_low"] is not None else "--"
            handle.write("{} & {} & {} & {} & {:.3f} & {} & {} & {} \\\\\n".format(row["dataset"], row["variant"], row["edit"].replace("_", "\\_"), row["concept"].replace("_", "\\_"), row["observed_concept_change_mean"], ci, "{:.3f}".format(row["spearman_rho"]) if row["spearman_rho"] is not None else "--", "{:.3f}".format(row["null_abs_rho_p"]) if row["null_abs_rho_p"] is not None else "--"))
        handle.write("\\bottomrule\n\\end{tabular}\n")
    write_task5_tables(rows, descriptor_analysis["correlations"], dataset_summaries)
    make_heatmap(rows, OUTPUT / "controlled_edit_heatmap.png", OUTPUT / "controlled_edit_heatmap.pdf")
    make_association_figure(rows, OUTPUT / "controlled_edit_associations.png", OUTPUT / "controlled_edit_associations.pdf")
    for dataset in sorted({str(row["dataset"]) for row in summaries}):
        subset = [row for row in summaries if str(row["dataset"]) == dataset]
        plot_controlled_edit_figure(subset, OUTPUT / f"{dataset}_matched_vs_null.png")
        plot_controlled_edit_figure(subset, OUTPUT / f"{dataset}_matched_vs_null.pdf")
    return output


def write_task5_tables(
    rows: Sequence[Mapping[str, Any]],
    descriptor_rows: Sequence[Mapping[str, Any]],
    dataset_summaries: Sequence[Mapping[str, Any]],
) -> None:
    """Write compact main and detailed supplementary LaTeX tables."""

    with (OUTPUT / "task5_main_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lrrrr}\n\\toprule\n")
        handle.write("Dataset & Clips & Mapped tests & Expected-sign success & Finite effects \\\\\n\\midrule\n")
        for summary in dataset_summaries:
            rate = summary["expected_sign_success_rate"]
            handle.write(
                "{} & {} & {} & {}/{} ({:.1f}\\%) & {} \\\\\n".format(
                    str(summary["dataset"]).replace("_", "\\_"),
                    summary["clip_count"],
                    summary["mapped_test_count"],
                    summary["expected_sign_success_count"],
                    summary["mapped_test_count"],
                    100.0 * float(rate) if rate is not None else float("nan"),
                    "yes" if summary["all_activation_effects_finite"] else "no",
                )
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")

    with (OUTPUT / "task5_supplementary_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{llllrrrrr}\n\\toprule\n")
        handle.write("Dataset & Variant & Intervention & Concept & Descriptor & $\\Delta c$ & 95\\% CI & $\\rho$ & Null $p$ \\\\\n\\midrule\n")
        for row in rows:
            ci = "[{:.3f}, {:.3f}]".format(row["observed_concept_change_ci_low"], row["observed_concept_change_ci_high"]) if row["observed_concept_change_ci_low"] is not None else "--"
            handle.write(
                "{} & {} & {} & {} & {} & {:.3f} & {} & {} & {} \\\\\n".format(
                    str(row["dataset"]).replace("_", "\\_"),
                    str(row["variant"]).replace("_", "\\_"),
                    str(row["edit"]).replace("_", "\\_"),
                    str(row["concept"]).replace("_", "\\_"),
                    str(row["descriptor"]).replace("_", "\\_"),
                    row["observed_concept_change_mean"],
                    ci,
                    "{:.3f}".format(row["spearman_rho"]) if row["spearman_rho"] is not None else "--",
                    "{:.3f}".format(row["null_abs_rho_p"]) if row["null_abs_rho_p"] is not None else "--",
                )
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")

    with (OUTPUT / "descriptor_associations_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{lllrrrr}\n\\toprule\n")
        handle.write("Dataset & Concept & Descriptor & $n$ & $\\rho$ & 95\\% CI & Sign \\\\\n\\midrule\n")
        for row in descriptor_rows:
            ci = "[{:.3f}, {:.3f}]".format(row["ci_low"], row["ci_high"]) if row["ci_low"] is not None else "--"
            handle.write(
                "{} & {} & {} & {} & {} & {} & {} \\\\\n".format(
                    str(row["dataset"]).replace("_", "\\_"),
                    str(row["concept"]).replace("_", "\\_"),
                    str(row["descriptor"]).replace("_", "\\_"),
                    row["n"],
                    "{:.3f}".format(row["spearman_rho"]) if row["spearman_rho"] is not None else "--",
                    ci,
                    "yes" if row.get("expected_sign_match") else "no",
                )
            )
        handle.write("\\bottomrule\n\\end{tabular}\n")


def make_heatmap(rows: list[dict[str, Any]], png: Path, pdf: Path) -> None:
    keys = sorted({(row["dataset"], row["variant"], row["concept"]) for row in rows})
    edits = [str(item["name"]) for item in CONTROLLED_EDIT_SPECS]
    matrix = np.full((len(keys), len(edits)), np.nan)
    for i, key in enumerate(keys):
        for j, edit in enumerate(edits):
            values = [row["observed_concept_change_mean"] for row in rows if (row["dataset"], row["variant"], row["concept"]) == key and row["edit"] == edit]
            if values:
                matrix[i, j] = values[0]
    height = max(5.0, 0.22 * len(keys))
    fig, ax = plt.subplots(figsize=(11, height))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_xticks(range(len(edits)), [x.replace("_", " ") for x in edits], rotation=40, ha="right")
    ax.set_yticks(range(len(keys)), ["/".join(key) for key in keys], fontsize=6)
    ax.set_title("Task-5 controlled-edit concept responses")
    fig.colorbar(image, ax=ax, label="mean concept activation change")
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)


def make_association_figure(rows: list[dict[str, Any]], png: Path, pdf: Path) -> None:
    groups = sorted({(row["dataset"], row["variant"]) for row in rows})
    values = []
    labels = []
    for dataset, variant in groups:
        rhos = [row["spearman_rho"] for row in rows if row["dataset"] == dataset and row["variant"] == variant and row["spearman_rho"] is not None]
        values.append(float(np.mean(rhos)) if rhos else np.nan)
        labels.append(f"{dataset}/{variant}")
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.65), 5.0))
    ax.bar(np.arange(len(labels)), values, color="#0072B2")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean mapped Spearman rho")
    ax.set_title("Mapped concept–descriptor association under controlled edits")
    fig.tight_layout()
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--descriptor-root", type=Path, default=DESCRIPTOR_ROOT)
    parser.add_argument("--descriptor-map", type=Path, default=MAP_PATH)
    parser.add_argument("--controlled-artifact-root", type=Path, default=OUTPUT)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global CONTROLLED_ARTIFACT_ROOT, DESCRIPTOR_ROOT, MAP_PATH, OUTPUT
    OUTPUT = args.output.resolve()
    DESCRIPTOR_ROOT = args.descriptor_root.resolve()
    MAP_PATH = args.descriptor_map.resolve()
    CONTROLLED_ARTIFACT_ROOT = args.controlled_artifact_root.resolve()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.skip_inference:
        validation_path = CONTROLLED_ARTIFACT_ROOT / "validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {"existing_validation_missing": True}
        reconciliation = reconcile_existing_artifacts(args.datasets)
        validation["artifact_reconciliation"] = reconciliation
        json_dump(OUTPUT / "validation.json", validation)
        results = {dataset: load_existing_dataset_artifacts(dataset) for dataset in args.datasets}
        analysis = make_final_analysis(results, args, validation)
        analysis["runtime_seconds"] = 0.0
        analysis["validation"]["artifact_reconciliation"] = reconciliation
        json_dump(OUTPUT / "controlled_edit_analysis.json", analysis)
        json_dump(OUTPUT / "task5_metadata.json", {"protocol": analysis["protocol"], "artifact_reconciliation": reconciliation, "validation": validation})
        print(json.dumps({"datasets": args.datasets, "mode": "cpu_only_artifact_postprocessing", "rows": len(analysis["rows"]), "descriptor_null_pairings": len(analysis["descriptor_null_pairings"])}, indent=2, sort_keys=True), flush=True)
        return
    validation = edit_validation()
    json_dump(OUTPUT / "validation.json", validation)
    descriptor_analysis = DESCRIPTOR_ROOT / "analysis.json"
    if not descriptor_analysis.exists():
        print("Descriptor analysis is missing; run the established Task-5 analysis command first.", flush=True)
    start = time.time()
    results = {}
    for dataset in args.datasets:
        results[dataset] = process_dataset(dataset, args, validation)
    analysis = make_final_analysis(results, args, validation)
    analysis["runtime_seconds"] = time.time() - start
    json_dump(OUTPUT / "controlled_edit_analysis.json", analysis)
    print(json.dumps({"datasets": args.datasets, "runtime_seconds": analysis["runtime_seconds"], "rows": len(analysis["rows"]), "validation": validation}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
