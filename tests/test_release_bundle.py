import hashlib
import json

import pytest

from scripts.evaluation import run_segmented_audio_ablation as segmented
from scripts.release import build_google_drive_bundle as bundle


def test_manifest_files_records_relative_path_size_and_checksum(tmp_path):
    artifact = tmp_path / "checkpoints" / "model.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"canonical-model")

    records = bundle.manifest_files(tmp_path)

    assert records == [
        {
            "path": "checkpoints/model.pt",
            "size_bytes": len(b"canonical-model"),
            "sha256": hashlib.sha256(b"canonical-model").hexdigest(),
        }
    ]


def test_verify_bundle_detects_modified_artifact(tmp_path):
    artifact = tmp_path / "concepts.txt"
    artifact.write_text("high-pitched\n", encoding="utf-8")
    files = bundle.manifest_files(tmp_path)
    manifest = {
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle.verify_bundle(tmp_path)

    artifact.write_text("low--pitched\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        bundle.verify_bundle(tmp_path)


def test_segmented_runner_loads_models_from_release_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "cremad" / "full"
    checkpoint.mkdir(parents=True)
    manifest = {
        "models": [
            {
                "dataset": "cremad",
                "variant": "full",
                "checkpoint_dir": "checkpoints/cremad/full",
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    args = type("Args", (), {"artifact_bundle": tmp_path, "cbm_summary": None})()
    runs = segmented.load_cbm_runs(args)

    assert runs == [
        {
            "dataset": "cremad",
            "variant": "full",
            "model_dir": str(checkpoint),
        }
    ]
