#!/usr/bin/env python3
"""GPU-only score generation for the persisted Task-6 URBAN-SED protocol.

This worker deliberately consumes the prepared event-window manifests.  It does
not construct a second windowing protocol: the final, partially out-of-range
windows in those manifests are zero padded to one second, as required by the
preparation record.  The worker writes one finite score matrix per scorer and
hop and is resumable at the file level.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import data_utils
from models.ast_classifier import build_ast_classifier
from scripts.evaluation.task6_urbansed_analysis import load_event_window_rows


SCORERS = (
    "cbm_positive_contribution",
    "cbm_silence_baseline",
    "ast_window_logit",
    "ast_occlusion_drop",
    "ast_attention_rollout",
)
HOPS = (0.5, 0.25)
CLASS_NAMES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_preparation(preparation: Path, bundle: Path) -> dict[str, Any]:
    summary = json.loads((preparation / "summary.json").read_text(encoding="utf-8"))
    retained = _read_jsonl(preparation / "retained_soundscapes.jsonl")
    provenance = _read_jsonl(preparation / "provenance_audit.jsonl")
    retained_ids = [str(row["id"]) for row in retained]
    if len(retained_ids) != len(set(retained_ids)):
        raise ValueError("retained soundscape manifest contains duplicate IDs")
    if len(retained_ids) != int(summary["counts"]["retained"]):
        raise ValueError("retained manifest and preparation summary disagree")
    if any(str(row.get("split")) != "test" for row in retained):
        raise ValueError("Task-6 retained soundscapes are not all from the persisted test subset")
    if any(not Path(row["audio_path"]).is_file() for row in retained):
        raise FileNotFoundError("one or more retained URBAN-SED audio paths are missing")

    source_ids: set[str] = set()
    source_folds: set[int] = set()
    source_event_count = 0
    for row in provenance:
        if row.get("status") != "retained":
            continue
        for event in row.get("events", []):
            if event.get("provenance", {}).get("status") != "resolved":
                continue
            source_event_count += 1
            source_ids.update(str(value) for value in event["provenance"].get("source_ids", []))
            source = event["provenance"].get("source", {})
            if source.get("fold") is not None:
                source_folds.add(int(source["fold"]))
    reference_folds = {int(value) for value in summary["urbansound8k_reference_training_folds"]}
    if source_folds & reference_folds:
        raise ValueError(f"retained URBAN-SED source overlap with reference training folds: {sorted(source_folds & reference_folds)}")
    if source_folds != {9, 10}:
        raise ValueError(f"persisted retained source folds changed: {sorted(source_folds)}")

    windows: dict[str, Any] = {}
    for hop in HOPS:
        path = preparation / f"event_windows_1s_hop_{hop:g}.jsonl"
        rows = load_event_window_rows(path, hop_seconds=hop)
        ids = [str(row["soundscape_id"]) for row in rows]
        counts = {sid: ids.count(sid) for sid in dict.fromkeys(ids)}
        expected = 20 if hop == 0.5 else 40
        if len(rows) != len(retained) * expected or len(counts) != len(retained):
            raise ValueError(f"prepared {hop:g}s event-window count is not {len(retained)} x {expected}")
        if any(value != expected for value in counts.values()):
            raise ValueError(f"prepared {hop:g}s event-window rows are not balanced per soundscape")
        if set(counts) != set(retained_ids):
            raise ValueError(f"prepared {hop:g}s soundscape IDs do not match retained manifest")
        starts_by_id: dict[str, list[float]] = {}
        for row in rows:
            starts_by_id.setdefault(str(row["soundscape_id"]), []).append(float(row["window_start"]))
        expected_starts = [round(index * hop, 6) for index in range(expected)]
        if any(values != expected_starts for values in starts_by_id.values()):
            raise ValueError(f"prepared {hop:g}s timestamps do not match the persisted padded-window protocol")
        windows[str(hop)] = {
            "path": str(path),
            "rows": len(rows),
            "soundscapes": len(counts),
            "windows_per_soundscape": expected,
            "window_seconds": 1.0,
            "hop_seconds": hop,
            "starts": expected_starts,
        }

    idx_to_label = json.loads(Path("data/urbansound8k/idx_to_label.json").read_text(encoding="utf-8"))
    class_names = [str(idx_to_label[str(index)]) for index in range(len(idx_to_label))]
    if class_names != CLASS_NAMES:
        raise ValueError(f"UrbanSound8K class mapping changed: {class_names}")
    bundle_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    full_entry = next(
        (item for item in bundle_manifest["models"] if item.get("dataset") == "urbansound8k" and item.get("variant") == "full"),
        None,
    )
    if full_entry is None:
        raise ValueError("canonical release bundle has no UrbanSound8K full checkpoint")
    checkpoint = bundle / full_entry["checkpoint_dir"]
    required = ["W_c.pt", "W_g.pt", "b_g.pt", "proj_mean.pt", "proj_std.pt", "concepts.txt", "args.txt"]
    missing = [name for name in required if not (checkpoint / name).is_file()
               ]
    if missing:
        raise FileNotFoundError(f"canonical checkpoint is incomplete: {missing}")
    hashes = {name: _sha256(checkpoint / name) for name in required}
    checkpoint_args = json.loads((checkpoint / "args.txt").read_text(encoding="utf-8"))
    if checkpoint_args.get("backbone") != "ast_urbansound8k" or checkpoint_args.get("train_split") != "fold10_train":
        raise ValueError("canonical checkpoint is not the persisted fold-10 UrbanSound8K AST configuration")
    stale_zero_byte = [str(path) for path in Path("results/revision/t6_urbansed").rglob("*") if path.is_file() and path.stat().st_size == 0]
    return {
        "protocol": "Task 6 URBAN-SED temporal localization",
        "dataset": "URBAN-SED v2.0",
        "retained_soundscapes": len(retained),
        "retained_soundscape_ids_unique": True,
        "retained_split": sorted({str(row["split"]) for row in retained}),
        "reference_training_folds": sorted(reference_folds),
        "retained_source_folds": sorted(source_folds),
        "source_event_count": source_event_count,
        "unique_source_clip_count": len(source_ids),
        "source_overlap_with_reference_training_folds": [],
        "source_clip_count_is_not_soundscape_count": {
            "status": "expected_by_protocol",
            "explanation": "URBAN-SED soundscapes are synthetic mixtures; source clips can recur across soundscapes and events.",
        },
        "class_names": class_names,
        "windows": windows,
        "window_alignment": "prepared event-window rows are authoritative; final windows are zero padded to 1.0 s",
        "source_archive_md5": summary["source_archive"]["md5"],
        "checkpoint": {
            "bundle_manifest": str(bundle / "manifest.json"),
            "variant": "full",
            "path": str(checkpoint),
            "model": "Adam-ousse/ast-urbansound8k-finetuned-fold10",
            "args": checkpoint_args,
            "release_manifest_entry": full_entry,
            "sha256": hashes,
            "selection": "canonical release checkpoint; no URBAN-SED result used for selection",
        },
        "stale_zero_byte_artifacts_before_inference": stale_zero_byte,
        "validation": "passed",
    }


def _load_audio(path: str, sample_rate: int = 16000) -> torch.Tensor:
    audio, actual_rate = data_utils._load_wav_audio(path, target_sample_rate=sample_rate, mono=True)
    if actual_rate != sample_rate:
        raise ValueError(f"audio loader returned unexpected sample rate {actual_rate} for {path}")
    audio = audio.squeeze(0).float()
    target = int(round(10.0 * sample_rate))
    if audio.numel() > target:
        audio = audio[:target]
    elif audio.numel() < target:
        audio = F.pad(audio, (0, target - audio.numel()))
    if audio.numel() != target or not torch.isfinite(audio).all():
        raise ValueError(f"invalid 10-second waveform for {path}")
    return audio


def _window_audio(audio: torch.Tensor, starts: Iterable[float], sample_rate: int = 16000) -> torch.Tensor:
    window = int(round(sample_rate))
    result = []
    for value in starts:
        start = int(round(float(value) * sample_rate))
        segment = audio[start : start + window]
        if segment.numel() < window:
            segment = F.pad(segment, (0, window - segment.numel()))
        result.append(segment)
    return torch.stack(result)


def _features_and_logits(model: torch.nn.Module, waveforms: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    def extract_one(waveform: np.ndarray) -> torch.Tensor:
        return model.feature_extractor(
            [waveform],
            sampling_rate=model.default_sample_rate,
            return_tensors="pt",
            padding=True,
        )["input_values"][0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        input_values = torch.stack(list(executor.map(extract_one, waveforms)))
    input_values = input_values.to(device)
    outputs = model.model.audio_spectrogram_transformer(input_values=input_values)
    features = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
    return features.float(), model.model.classifier(features).float()


def _batched_features_and_logits(model: torch.nn.Module, waveforms: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for start in range(0, waveforms.shape[0], batch_size):
        batch = waveforms[start : start + batch_size]
        feature, logit = _features_and_logits(model, [row.detach().cpu().numpy() for row in batch], device)
        features.append(feature.detach())
        logits.append(logit.detach())
    return torch.cat(features), torch.cat(logits)


def _attention_rollout_batch(model: torch.nn.Module, waveforms: list[torch.Tensor], device: torch.device) -> np.ndarray:
    def extract_one(waveform: torch.Tensor) -> torch.Tensor:
        return model.feature_extractor(
            [waveform.detach().cpu().numpy()],
            sampling_rate=model.default_sample_rate,
            return_tensors="pt",
            padding=True,
        )["input_values"][0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        input_values = torch.stack(list(executor.map(extract_one, waveforms)))
    captured: list[torch.Tensor] = []
    hooks = []

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        captured.append(output[1])

    encoder_layers = model.model.audio_spectrogram_transformer.encoder.layer
    for layer in encoder_layers:
        hooks.append(layer.attention.attention.register_forward_hook(capture))
    ast_config = model.model.audio_spectrogram_transformer.config
    original_attention_implementation = getattr(ast_config, "_attn_implementation", "eager")
    ast_config._attn_implementation = "eager"
    try:
        with torch.inference_mode():
            model.model.audio_spectrogram_transformer(input_values=input_values.to(device))
    finally:
        ast_config._attn_implementation = original_attention_implementation
        for hook in hooks:
            hook.remove()
    attentions = captured
    if not attentions:
        raise RuntimeError("AST did not return attentions for the persisted attention-rollout baseline")
    # This AST checkpoint uses two leading special tokens (CLS and the
    # distillation token), followed by the time-frequency patch grid.
    special_tokens = 2
    token_count = int(attentions[-1].shape[-1]) - special_tokens
    frequency_patches = (128 - 16) // 10 + 1
    if token_count % frequency_patches != 0:
        raise ValueError(f"unexpected AST patch-token count {token_count}")
    time_patches = token_count // frequency_patches
    joint = torch.eye(int(attentions[-1].shape[-1]), device=device, dtype=torch.float32).unsqueeze(0).expand(len(waveforms), -1, -1).clone()
    for attention in attentions:
        matrix = attention.float().mean(dim=1)
        matrix = matrix + torch.eye(matrix.shape[-1], device=device, dtype=matrix.dtype).unsqueeze(0)
        matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = matrix @ joint
    patch_mass = joint[:, 0, special_tokens:].reshape(len(waveforms), time_patches, frequency_patches).sum(dim=2)
    time_rollout = patch_mass.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(time_rollout).all():
        raise ValueError("attention rollout contains non-finite values")
    return time_rollout


def _attention_to_windows(rollout: np.ndarray, starts: list[float], window_seconds: float = 1.0) -> np.ndarray:
    time_patches = int(rollout.shape[0])
    token_intervals = np.asarray(
        [[index * 0.1, (index * 0.1) + 0.16] for index in range(time_patches)], dtype=np.float64
    )
    values = []
    for start in starts:
        end = float(start) + window_seconds
        overlap = np.maximum(0.0, np.minimum(end, token_intervals[:, 1]) - np.maximum(float(start), token_intervals[:, 0]))
        duration = np.maximum(token_intervals[:, 1] - token_intervals[:, 0], 0.0)
        values.append(float(np.sum(rollout * np.divide(overlap, duration, out=np.zeros_like(overlap), where=duration > 0))))
    return np.asarray(values, dtype=np.float32)


def _load_cbm(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = {name: torch.load(checkpoint / f"{name}.pt", map_location=device, weights_only=True).float() for name in ("W_c", "W_g", "b_g", "proj_mean", "proj_std")}
    if tensors["W_c"].shape[0] != tensors["W_g"].shape[1] or tensors["W_c"].shape[1] != 768:
        raise ValueError("canonical CBM projection/classifier dimensions do not match the AST backbone")
    if tensors["proj_mean"].shape != (1, tensors["W_c"].shape[0]) or tensors["proj_std"].shape != tensors["proj_mean"].shape:
        raise ValueError("canonical CBM normalization dimensions do not match the concept projection")
    if not torch.isfinite(torch.cat([value.reshape(-1) for value in tensors.values()])).all():
        raise ValueError("canonical CBM checkpoint contains non-finite tensors")
    return tensors


def _save_score(path: Path, score: np.ndarray, rows: list[dict[str, Any]], class_names: list[str], metadata: dict[str, Any]) -> None:
    score = np.asarray(score, dtype=np.float32)
    if score.ndim != 2 or score.shape != (len(rows), len(class_names)) or not np.isfinite(score).all():
        raise ValueError(f"invalid score shape or non-finite values for {path}: {score.shape}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        scores=score,
        soundscape_ids=np.asarray([str(row["soundscape_id"]) for row in rows]),
        window_starts=np.asarray([float(row["window_start"]) for row in rows], dtype=np.float32),
        window_ends=np.asarray([float(row["window_end"]) for row in rows], dtype=np.float32),
        class_names=np.asarray(class_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def _group_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        groups.setdefault(str(row["soundscape_id"]), []).append(row)
    return list(groups.items())


def run(args: argparse.Namespace) -> None:
    if args.device == "cpu":
        raise ValueError("Task 6 score generation requires the explicitly permitted GPU path; use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to substitute a different inference path")
    device = torch.device(args.device)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    prep = args.preparation
    bundle = args.bundle
    validation = _validate_preparation(prep, bundle)
    validation["device"] = str(device)
    validation["score_generation_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_dump(root / "provenance_validation.json", validation)

    checkpoint = bundle / "checkpoints/urbansound8k/full"
    cbm = _load_cbm(checkpoint, device)
    classifier = build_ast_classifier("ast_urbansound8k", device)
    if classifier.num_labels != len(CLASS_NAMES) or [str(classifier.id2label.get(i, classifier.id2label.get(str(i)))) for i in range(classifier.num_labels)] != CLASS_NAMES:
        raise ValueError("fine-tuned AST class mapping does not match UrbanSound8K/URBAN-SED")
    classifier.eval()
    silence_window = torch.zeros((1, 16000), dtype=torch.float32)
    with torch.inference_mode():
        silence_features, _ = _batched_features_and_logits(classifier, silence_window, 1, device)
        silence_concepts = (silence_features @ cbm["W_c"].T - cbm["proj_mean"]) / cbm["proj_std"].clamp_min(1e-12)
        silence_contributions = silence_concepts[0, None, :] * cbm["W_g"]
    # Shape is [classes, concepts]; this is the fixed Task-1 one-second silence reference.
    silence_contributions = silence_contributions.detach()
    validation["checkpoint"]["ast_label_order"] = CLASS_NAMES
    validation["checkpoint"]["concept_count"] = int(cbm["W_c"].shape[0])
    validation["checkpoint"]["silence_baseline"] = "AST representation of a zero 1-second waveform"
    validation["validation"] = "passed_before_score_generation"
    _json_dump(root / "provenance_validation.json", validation)

    retained = {str(row["id"]): row for row in _read_jsonl(prep / "retained_soundscapes.jsonl")}
    rows_by_hop = {
        hop: load_event_window_rows(prep / f"event_windows_1s_hop_{hop:g}.jsonl", hop_seconds=hop)
        for hop in HOPS
    }
    groups_by_hop = {hop: _group_rows(rows_by_hop[hop]) for hop in HOPS}
    if [sid for sid, _ in groups_by_hop[0.5]] != [sid for sid, _ in groups_by_hop[0.25]]:
        raise ValueError("the persisted hop manifests do not have identical soundscape ordering")
    if args.max_soundscapes is not None:
        for hop in HOPS:
            groups_by_hop[hop] = groups_by_hop[hop][: int(args.max_soundscapes)]
            rows_by_hop[hop] = [row for _, group_rows in groups_by_hop[hop] for row in group_rows]
    output_paths = {
        hop: {scorer: root / f"scores_1s_hop_{hop:g}_{scorer}.npz" for scorer in SCORERS}
        for hop in HOPS
    }
    complete = {
        hop: all(path.is_file() and path.stat().st_size > 0 for path in output_paths[hop].values()) and not args.force
        for hop in HOPS
    }
    scores = {hop: {scorer: [] for scorer in SCORERS} for hop in HOPS}
    pending = {hop for hop in HOPS if not complete[hop]}
    if pending:
        # The 0.25-s manifest is the superset: its even-indexed rows are the
        # persisted 0.5-s windows.  Shared model calls avoid recomputing the
        # same full clip and attention rollout for the two reporting hops.
        attention_records: list[dict[str, Any]] = []

        def flush_attention_records() -> None:
            if not attention_records:
                return
            rollouts = _attention_rollout_batch(classifier, [record["audio"] for record in attention_records], device)
            for record, rollout in zip(attention_records, rollouts):
                fine_attention = _attention_to_windows(rollout, record["fine_starts"])
                fine_scores = record["fine_scores"]
                fine_scores["ast_attention_rollout"] = np.repeat(fine_attention[:, None], len(CLASS_NAMES), axis=1)
                for hop in pending:
                    indices = record["coarse_indices"] if hop == 0.5 else list(range(len(record["fine_rows"])))
                    for scorer in SCORERS:
                        scores[hop][scorer].append(fine_scores[scorer][indices])
            attention_records.clear()

        for soundscape_index, (soundscape_id, fine_rows) in enumerate(
            tqdm(groups_by_hop[0.25], desc="Task6 shared GPU inference", unit="soundscape")
        ):
            if soundscape_id not in retained:
                raise ValueError(f"prepared window references unknown soundscape {soundscape_id}")
            coarse_rows = groups_by_hop[0.5][soundscape_index][1]
            audio = _load_audio(retained[soundscape_id]["audio_path"])
            fine_starts = [float(row["window_start"]) for row in fine_rows]
            coarse_indices = list(range(0, len(fine_rows), 2))
            if [fine_starts[index] for index in coarse_indices] != [float(row["window_start"]) for row in coarse_rows]:
                raise ValueError(f"prepared hop alignment failed for {soundscape_id}")
            windows = _window_audio(audio, fine_starts)
            occluded = audio.unsqueeze(0).repeat(len(fine_starts), 1)
            for index, start in enumerate(fine_starts):
                left = int(round(start * 16000))
                right = min(left + 16000, occluded.shape[1])
                if left < occluded.shape[1]:
                    occluded[index, left:right] = 0.0
            with torch.inference_mode():
                window_features, window_logits = _batched_features_and_logits(classifier, windows, args.batch_size, device)
                full_logits = _batched_features_and_logits(classifier, audio.unsqueeze(0), 1, device)[1]
                occluded_logits = _batched_features_and_logits(classifier, occluded, args.batch_size, device)[1]
                concepts = (window_features @ cbm["W_c"].T - cbm["proj_mean"]) / cbm["proj_std"].clamp_min(1e-12)
                contributions = concepts[:, None, :] * cbm["W_g"][None, :, :]
                positive = torch.clamp(contributions, min=0.0).sum(dim=-1)
                silence_centered = contributions - silence_contributions[None, :, :]
                silence_score = torch.clamp(silence_centered, min=0.0).sum(dim=-1)
            attention_records.append({
                "audio": audio,
                "fine_rows": fine_rows,
                "fine_starts": fine_starts,
                "coarse_indices": coarse_indices,
                "fine_scores": {
                    "cbm_positive_contribution": positive.detach().cpu().numpy(),
                    "cbm_silence_baseline": silence_score.detach().cpu().numpy(),
                    "ast_window_logit": window_logits.detach().cpu().numpy(),
                    "ast_occlusion_drop": (full_logits[0][None, :] - occluded_logits).detach().cpu().numpy(),
                },
            })
            if len(attention_records) >= args.attention_batch_size:
                flush_attention_records()
        flush_attention_records()
        for hop in pending:
            metadata = {
                "protocol": "Task 6 URBAN-SED temporal localization",
                "hop_seconds": hop,
                "window_seconds": 1.0,
                "checkpoint_variant": "full",
                "checkpoint_dir": str(checkpoint),
                "ast_model": "Adam-ousse/ast-urbansound8k-finetuned-fold10",
                "class_names": CLASS_NAMES,
                "device": str(device),
                "window_construction": "prepared rows; zero pad final out-of-range samples",
                "attention_rollout": "12-layer residual attention rollout, frequency-patch mass summed into 100-ms temporal patches",
                "silence_baseline": "zero 1-second waveform representation",
            }
            for scorer in SCORERS:
                score = np.concatenate(scores[hop][scorer], axis=0)
                _save_score(output_paths[hop][scorer], score, rows_by_hop[hop], CLASS_NAMES, metadata | {"scorer": scorer})
                print(json.dumps({"hop": hop, "scorer": scorer, "rows": len(rows_by_hop[hop]), "path": str(output_paths[hop][scorer])}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, default=Path("results/revision/t6_urbansed/preparation"))
    parser.add_argument("--bundle", type=Path, default=Path("release/google_drive_bundle"))
    parser.add_argument("--output-root", type=Path, default=Path("results/revision/t6_urbansed/inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--attention-batch-size", type=int, default=2)
    parser.add_argument("--max-soundscapes", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
