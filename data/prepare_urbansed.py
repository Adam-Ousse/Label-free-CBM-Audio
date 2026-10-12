#!/usr/bin/env python3
"""Download, verify, and prepare URBAN-SED v2.0 for leakage-safe windows.

URBAN-SED v2.0 is synthetic audio whose foreground events come from
UrbanSound8K.  This script does not infer provenance from the public train /
validation / test names alone: it parses each JAMS event's ``source_file`` and
cross-checks that source against the local UrbanSound8K metadata.  Any missing
or conflicting source identity is retained in the audit log and excluded from
the leakage-safe set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

URBANSED_URL = "https://zenodo.org/record/1324404/files/URBAN-SED_v2.0.0.tar.gz?download=1"
URBANSED_MD5 = "a2d24a2148ece7c021fcc079ee87c2dc"
URBANSED_INDEX_URL = "https://zenodo.org/records/11176925/files/urbansed_index_2.0.json?download=1"
REFERENCE_TRAIN_FOLDS = frozenset(range(1, 9))
US8K_ID_RE = re.compile(r"(?<!\d)(\d{2,8}-\d+-\d+-\d+)(?!\d)")


def _write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing asset: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing asset: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(destination: Path, *, url: str = URBANSED_URL, md5: str = URBANSED_MD5) -> dict[str, Any]:
    """Download the official archive to a new path and verify its MD5."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _md5(destination)
        if actual != md5:
            raise ValueError(f"Existing URBAN-SED archive has MD5 {actual}, expected {md5}: {destination}")
        return {"path": str(destination), "bytes": destination.stat().st_size, "md5": actual, "reused": True}

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        raise FileExistsError(
            f"A partial download exists at {partial}; remove it only after checking it is disposable, then retry."
        )
    request = Request(url, headers={"User-Agent": "Label-free-CBM-Audio/Task6"})
    with urlopen(request) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(16 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    actual = _md5(partial)
    if actual != md5:
        raise ValueError(f"Downloaded URBAN-SED archive has MD5 {actual}, expected {md5}")
    os.replace(partial, destination)
    return {"path": str(destination), "bytes": destination.stat().st_size, "md5": actual, "reused": False}


def _safe_member_path(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return target


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extract once, refusing to overwrite any existing member."""

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name and member.name != "."]
        for member in members:
            target = _safe_member_path(destination, member.name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member type: {member.name}")
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with target.open("xb") as output:
                while chunk := source.read(16 * 1024 * 1024):
                    output.write(chunk)
    candidates = [destination / "URBAN-SED_v2.0.0", destination]
    for candidate in candidates:
        if (candidate / "audio").exists() and (candidate / "annotations").exists():
            return candidate
    raise FileNotFoundError(f"Could not find extracted URBAN-SED audio/annotations under {destination}")


def load_urbansound_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Map source clip stem to all matching UrbanSound8K metadata rows."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        stem = Path(row["slice_file_name"]).stem
        result.setdefault(stem, []).append(
            {
                "id": stem,
                "fold": int(row["fold"]),
                "class": row["class"].strip(),
                "slice_file_name": row["slice_file_name"].strip(),
            }
        )
    return result


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _event_values(jams: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for annotation in jams.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        for event in annotation.get("data", []):
            if isinstance(event, dict):
                value = event.get("value", {})
                if isinstance(value, dict):
                    values.append({"time": event.get("time"), "duration": event.get("duration"), **value})
    return values


def _source_candidates(event: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("source_file", "source_filename", "filename", "file"):
        value = event.get(key)
        if isinstance(value, str):
            candidates.append(value)
    # Some older Scaper JAMS variants nest source_file in a sandbox object.
    for value in _iter_strings(event.get("sandbox", {})):
        if US8K_ID_RE.search(value):
            candidates.append(value)
    return sorted(set(candidates))


def resolve_event_source(event: dict[str, Any], source_index: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    candidates = _source_candidates(event)
    if str(event.get("role", "")).lower() == "background" or any(
        "/background/" in value.lower() or "brownian" in value.lower() for value in candidates
    ):
        return {
            "source_file_candidates": candidates,
            "source_ids": [],
            "status": "background",
            "reason": "synthetic_background_not_an_urbansound8k_source_clip",
        }
    ids = sorted({match.group(1) for value in candidates for match in US8K_ID_RE.finditer(value)})
    if len(ids) != 1:
        return {
            "source_file_candidates": candidates,
            "source_ids": ids,
            "status": "ambiguous",
            "reason": "missing_or_nonunique_source_clip_id",
        }
    source_id = ids[0]
    rows = source_index.get(source_id, [])
    if len(rows) != 1:
        return {
            "source_file_candidates": candidates,
            "source_ids": ids,
            "status": "ambiguous",
            "reason": "source_clip_id_not_unique_in_urbansound8k_manifest",
        }
    source = rows[0]
    fold_numbers = sorted({int(match.group(1)) for value in candidates for match in re.finditer(r"fold(\d+)", value, re.I)})
    if fold_numbers and fold_numbers != [source["fold"]]:
        return {
            "source_file_candidates": candidates,
            "source_ids": ids,
            "status": "ambiguous",
            "reason": "source_path_fold_conflicts_with_manifest",
            "manifest_source": source,
            "path_folds": fold_numbers,
        }
    return {
        "source_file_candidates": candidates,
        "source_ids": ids,
        "status": "resolved",
        "source": source,
        "source_event": event,
    }


def read_soundscape_events(jams_path: Path, source_index: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jams = json.loads(jams_path.read_text(encoding="utf-8"))
    events = _event_values(jams)
    resolved: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for event in events:
        source = resolve_event_source(event, source_index)
        if source["status"] == "ambiguous":
            ambiguities.append(source)
        resolved.append({"event": event, "provenance": source})
    return resolved, ambiguities


def _soundscape_rows(dataset_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validate", "test"):
        audio_dir = dataset_root / "audio" / split
        jams_dir = dataset_root / "annotations" / split
        for audio_path in sorted(audio_dir.glob("*.wav")):
            # The official tarball contains macOS Finder resource forks next
            # to some real files; they are not dataset clips and have no JAMS.
            if audio_path.name.startswith("._"):
                continue
            jams_path = jams_dir / f"{audio_path.stem}.jams"
            if not jams_path.exists():
                raise FileNotFoundError(f"Missing JAMS for {audio_path}")
            rows.append({"id": audio_path.stem, "split": split, "audio_path": str(audio_path), "jams_path": str(jams_path)})
    return rows


def _event_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    try:
        start = float(event.get("event_time", event.get("time")))
        duration = float(event.get("event_duration", event.get("duration")))
    except (TypeError, ValueError):
        return None
    if duration <= 0.0:
        return None
    return start, start + duration


def _window_rows(soundscape: dict[str, Any], events: list[dict[str, Any]], hop: float) -> Iterable[dict[str, Any]]:
    duration = 10.0
    window = 1.0
    start = 0.0
    while start < duration:
        end = start + window
        active = []
        for item in events:
            event = item["event"]
            if item["provenance"].get("status") != "resolved":
                continue
            interval = _event_interval(event)
            if interval is None:
                continue
            event_start, event_end = interval
            if event_start < end and event_end > start:
                active.append(
                    {
                        "label": event.get("label"),
                        "event_start": event_start,
                        "event_end": event_end,
                        "overlap_start": max(start, event_start),
                        "overlap_end": min(end, event_end),
                        "source_id": item["provenance"].get("source_ids", [None])[0],
                    }
                )
        yield {
            "soundscape_id": soundscape["id"],
            "split": soundscape["split"],
            "audio_path": soundscape["audio_path"],
            "window_start": round(start, 6),
            "window_end": round(end, 6),
            "window_seconds": window,
            "hop_seconds": hop,
            "labels": sorted({str(item["label"]) for item in active if item.get("label")}),
            "events": active,
        }
        start += hop


def prepare(
    dataset_root: Path,
    urbansound_manifest: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_index = load_urbansound_manifest(urbansound_manifest)
    all_rows = _soundscape_rows(dataset_root)
    provenance_rows: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    ambiguity_rows: list[dict[str, Any]] = []
    for soundscape in all_rows:
        resolved_events, ambiguities = read_soundscape_events(Path(soundscape["jams_path"]), source_index)
        foreground_events = [
            item for item in resolved_events if item["provenance"].get("status") == "resolved"
        ]
        source_folds = sorted(
            {
                int(item["provenance"]["source"]["fold"])
                for item in resolved_events
                if item["provenance"]["status"] == "resolved"
            }
        )
        reasons: list[str] = []
        if ambiguities:
            reasons.append("provenance_ambiguous")
        if any(fold in REFERENCE_TRAIN_FOLDS for fold in source_folds):
            reasons.append("source_clip_in_reference_training_folds_1_8")
        if not foreground_events:
            reasons.append("no_scaper_events_found")
        if not reasons:
            status = "retained"
            retained_rows.append(soundscape)
        else:
            status = "excluded"
            excluded_rows.append({**soundscape, "exclusion_reasons": reasons})
        record = {
            **soundscape,
            "status": status,
            "source_folds": source_folds,
            "source_clip_ids": sorted(
                {
                    source_id
                    for item in resolved_events
                    for source_id in item["provenance"].get("source_ids", [])
                }
            ),
            "events": resolved_events,
            "exclusion_reasons": reasons,
        }
        provenance_rows.append(record)
        for ambiguity in ambiguities:
            ambiguity_rows.append({"soundscape_id": soundscape["id"], "split": soundscape["split"], **ambiguity})

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "provenance_audit.jsonl", provenance_rows, overwrite=overwrite)
    _write_jsonl(output_root / "retained_soundscapes.jsonl", retained_rows, overwrite=overwrite)
    _write_jsonl(output_root / "excluded_soundscapes.jsonl", excluded_rows, overwrite=overwrite)
    _write_jsonl(output_root / "provenance_ambiguities.jsonl", ambiguity_rows, overwrite=overwrite)

    for hop in (0.5, 0.25):
        rows = []
        retained_by_id = {row["id"]: row for row in provenance_rows if row["status"] == "retained"}
        for soundscape in retained_rows:
            rows.extend(_window_rows(soundscape, retained_by_id[soundscape["id"]]["events"], hop))
        _write_jsonl(output_root / f"event_windows_1s_hop_{hop:g}.jsonl", rows, overwrite=overwrite)

    summary = {
        "dataset": "URBAN-SED",
        "version": "2.0.0",
        "source_archive": {"url": URBANSED_URL, "md5": URBANSED_MD5},
        "urbansound8k_reference_training_folds": sorted(REFERENCE_TRAIN_FOLDS),
        "counts": {
            "before_filtering": len(all_rows),
            "retained": len(retained_rows),
            "excluded": len(excluded_rows),
            "provenance_ambiguous_soundscapes": len({row["soundscape_id"] for row in ambiguity_rows}),
            "retained_by_split": {
                split: sum(row["split"] == split for row in retained_rows)
                for split in ("train", "validate", "test")
            },
            "excluded_reason_counts": {
                reason: sum(reason in row.get("exclusion_reasons", []) for row in excluded_rows)
                for reason in sorted({reason for row in excluded_rows for reason in row.get("exclusion_reasons", [])})
            },
        },
        "event_window_files": {
            "1.0s_hop_0.5s": "event_windows_1s_hop_0.5.jsonl",
            "1.0s_hop_0.25s": "event_windows_1s_hop_0.25.jsonl",
        },
        "ambiguity_policy": "exclude any soundscape with missing, nonunique, or conflicting source clip provenance",
    }
    _write_json(output_root / "summary.json", summary, overwrite=overwrite)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/urbansed/URBAN-SED_v2.0.0.tar.gz"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/urbansed/raw"))
    parser.add_argument("--urbansound-manifest", type=Path, default=Path("data/urbansound8k/raw/UrbanSound8K.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results/revision/t6_urbansed/preparation"))
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.no_download:
        print(json.dumps(download_archive(args.archive), indent=2, sort_keys=True))
    if not args.no_extract:
        dataset_root = extract_archive(args.archive, args.dataset_root)
    else:
        dataset_root = args.dataset_root
    summary = prepare(
        dataset_root,
        args.urbansound_manifest,
        args.output_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
