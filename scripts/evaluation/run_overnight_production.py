#!/usr/bin/env python3
"""Unattended, conservative ESC/Urban production schedule.

The supervisor runs at most one GPU-heavy child at a time. It never resumes
from partial AST checkpoints and never treats the aborted Urban attempt as an
official fold.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATUS = ROOT / "results/revision/OVERNIGHT_STATUS.md"
PROD = ROOT / "results/revision/overnight_production_20261015"
CACHE = PROD / "cache/ast_inputs"
HF_HOME = Path("/workspace/.hf_home_production")
CLAP_MODEL = "laion/clap-htsat-unfused"
ESC5_RUN_ID = "esc50_compressed_gpu1_fold5_concurrent"
ESC4_RUN_ID = "esc50_compressed_gpu1_overnight_exactcache_20261015"
URBAN_RUN_ID = "urbansound8k_official_cv_overnight_exactcache_20261015"
ESC5_COMPLETE = ROOT / "results/revision" / ESC5_RUN_ID / "fold5/fold_complete.json"
ESC5_LOG = ROOT / "results/revision" / ESC5_RUN_ID / "ast/esc50/fold5/train.log"
ESC4_COMPLETE = ROOT / "results/revision" / ESC4_RUN_ID / "fold4/fold_complete.json"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def epoch_from_log(path: Path) -> str:
    if not path.exists():
        return "unknown"
    matches = re.findall(r"epoch\s+(\d+)/30", path.read_text(encoding="utf-8", errors="replace"))
    return (matches[-1] + "/30") if matches else "not logged"


def urban_complete_folds() -> list[int]:
    result = []
    for fold in range(1, 11):
        marker = ROOT / "results/revision/t4_cv" / URBAN_RUN_ID / ("fold{}/fold_complete.json".format(fold))
        if marker.exists():
            result.append(fold)
    return result


def urban_metrics() -> tuple[float | None, float | None]:
    ast_values = []
    cbm_values = []
    for fold in urban_complete_folds():
        marker = ROOT / "results/revision/t4_cv" / URBAN_RUN_ID / ("fold{}/fold_complete.json".format(fold))
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            ast_values.append(float(payload["ast_test_metrics"]["accuracy"]))
            cbm_values.append(float(payload["cbm_test_metrics"]["accuracy"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return (
        (sum(ast_values) / len(ast_values)) if ast_values else None,
        (sum(cbm_values) / len(cbm_values)) if cbm_values else None,
    )


def urban_progress(stage: str, complete: list[int]) -> tuple[str, str, str]:
    match = re.search(r"Urban fold (\d+)", stage)
    if not match:
        return ("none", "not running", "pending first fold")
    fold = int(match.group(1))
    train_log = ROOT / "results/revision/t4_cv" / URBAN_RUN_ID / ("ast/urbansound8k/fold{}/train.log".format(fold))
    remaining = len([f for f in range(1, 11) if f not in complete])
    runtimes = []
    for done_fold in complete:
        marker = ROOT / "results/revision/t4_cv" / URBAN_RUN_ID / ("fold{}/fold_complete.json".format(done_fold))
        try:
            runtimes.append(float(json.loads(marker.read_text(encoding="utf-8"))["runtime_seconds"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    if runtimes:
        eta = "~{} remaining".format(time.strftime("%H:%M", time.gmtime(max(0, int((remaining - 1) * (sum(runtimes) / len(runtimes)))))))
    else:
        eta = "pending first fold timing"
    return (str(fold), epoch_from_log(train_log), eta)


def write_status(stage: str, detail: str = "") -> None:
    complete = urban_complete_folds()
    ast_mean, cbm_mean = urban_metrics()
    current_fold, current_epoch, urban_eta = urban_progress(stage, complete)
    lines = [
        "# Overnight production status",
        "",
        "Last update: {}".format(now()),
        "",
        "## Current state",
        "",
        "- Stage: {}".format(stage),
        "- Detail: {}".format(detail or "none"),
        "- Supervisor PID: `{}`; persistent HF cache: `{}`.".format(os.getpid(), HF_HOME),
        "- ESC fold 5: run `{}`; latest completed AST epoch `{}`; fold-complete marker: `{}`.".format(ESC5_RUN_ID, epoch_from_log(ESC5_LOG), ESC5_COMPLETE.exists()),
        "- ESC fold 4: run `{}`; fold-complete marker: `{}`.".format(ESC4_RUN_ID, ESC4_COMPLETE.exists()),
        "- ESC official aggregation: `{}`.".format((PROD / "esc50_official_cv/summary.json").exists()),
        "- Urban run: `{}`; completed official folds: `{}`; remaining: `{}`.".format(URBAN_RUN_ID, complete or "none", [f for f in range(1, 11) if f not in complete]),
        "- Urban current fold/epoch: `{}/{}`; ETA estimate: `{}`.".format(current_fold, current_epoch, urban_eta),
        "- Urban completed mean accuracy: AST `{}`; LF+broad `{}`.".format("n/a" if ast_mean is None else "{:.4f}".format(ast_mean), "n/a" if cbm_mean is None else "{:.4f}".format(cbm_mean)),
        "- Errors/retries: `{}`; see supervisor and per-fold logs.".format("none recorded" if "failed" not in detail.lower() and "retry" not in detail.lower() else detail),
        "- Aborted Urban attempt: preserved and excluded at `results/revision/t4_cv/urbansound8k_official_cv_overnight_20261015/ABORTED_NOT_USED.md`.",
        "",
        "## Exact execution configuration",
        "",
        "- ESC cache: `cache/ast_inputs/esc50_input_values.pt`, global exact float32 `(2000, 1024, 128)`.",
        "- Urban cache: `cache/ast_inputs/urbansound8k_input_values.pt`, global exact float32 `(8732, 1024, 128)`.",
        "- ESC fold 4: test fold 4, validation fold 5, batch 8, accumulation 1, 30 epochs, seed 42, original optimizer/LR/scheduler.",
        "- Urban replacement CV: test fold 1→10, next cyclic validation, physical batch 16 × accumulation 2 = effective batch 32, 30 epochs, seed 42, original optimizer/LR/scheduler.",
        "- No AMP/BF16/FP16, TF32, compile, alternative attention, fused optimizer, early stopping, or checkpoint-resume semantics.",
        "",
        "## Logs",
        "",
        "- Supervisor: `overnight_supervisor.log`",
        "- ESC fold 4: `esc_fold4.log`",
        "- Urban per-fold logs: `urban_foldN.log`",
    ]
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": str(HF_HOME),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    return env


def run_logged(label: str, command: list[str], log_path: Path, *, status_detail: str, retries: int = 0) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n=== {} attempt {}/{} at {} ===\n$ {}\n".format(label, attempt, attempts, now(), " ".join(command)))
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=base_env(), stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                write_status(label, "{}; attempt {}/{}".format(status_detail, attempt, attempts))
                time.sleep(30)
            code = process.returncode
            log.write("=== {} exit code {} at {} ===\n".format(label, code, now()))
        if code == 0:
            return
        if attempt < attempts:
            write_status(label, "failed with code {}; retrying cleanly".format(code))
            time.sleep(5)
    raise RuntimeError("{} failed after {} attempt(s); see {}".format(label, attempts, log_path))


def preflight() -> None:
    """Fail before launching multi-hour work if exact offline inputs are absent."""

    required = [
        CACHE / "esc50_input_values.pt",
        CACHE / "urbansound8k_input_values.pt",
        ESC5_COMPLETE,
        HF_HOME / "hub/models--laion--clap-htsat-unfused/snapshots/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a/config.json",
        HF_HOME / "hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/f826b80d28226b62986cc218e5cec390b1096902/config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("overnight preflight missing: {}".format(", ".join(missing)))
    smoke = [
        sys.executable, "-c",
        "from clap.core import load_clap_model; from transformers import ASTForAudioClassification; "
        "load_clap_model('" + CLAP_MODEL + "', 'cpu'); "
        "ASTForAudioClassification.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593', local_files_only=True); "
        "print('offline model preflight passed')",
    ]
    subprocess.run(smoke, cwd=ROOT, env=base_env(), check=True)


def main() -> None:
    PROD.mkdir(parents=True, exist_ok=True)
    write_status("supervisor started", "preflighting exact persistent caches and completed ESC fold 5")
    preflight()
    write_status("ESC fold 5 complete", "starting clean ESC fold 4 with exact cached AST inputs")

    fold4_command = [
        sys.executable, str(ROOT / "scripts/evaluation/run_esc50_compressed.py"),
        "--repo-root", str(ROOT), "--run-id", ESC4_RUN_ID, "--fold", "4",
        "--device", "cuda", "--num-workers", "1", "--ast-batch-size", "8", "--feature-batch-size", "4",
        "--ast-input-cache", str(CACHE / "esc50_input_values.pt"),
    ]
    if not ESC4_COMPLETE.exists():
        run_logged("ESC fold 4", fold4_command, PROD / "esc_fold4.log", status_detail="clean epoch-1 restart with exact cache", retries=0)

    aggregate_command = [
        sys.executable, str(ROOT / "scripts/evaluation/aggregate_esc50_overnight.py"),
        "--repo-root", str(ROOT), "--output-dir", str(PROD / "esc50_official_cv"),
        "--fold4-run-id", ESC4_RUN_ID, "--fold5-run-id", ESC5_RUN_ID,
    ]
    if not (PROD / "esc50_official_cv/summary.json").exists():
        run_logged("ESC official aggregation", aggregate_command, PROD / "esc_aggregate.log", status_detail="verifying all five official folds", retries=0)
    else:
        write_status("ESC complete", "preserving existing verified ESC aggregation; no ESC work rerun")

    smoke_log = PROD / "urban_batch16_smoke.log"
    smoke_command = [
        sys.executable, str(ROOT / "scripts/training/smoke_cached_ast_batch.py"),
        "--cache", str(CACHE / "urbansound8k_input_values.pt"),
        "--manifest", str(ROOT / "results/revision/t4_cv/urbansound8k_official_cv/manifests/fold1_train.jsonl"),
        "--batch-size", "16",
    ]
    smoke_failed = False
    try:
        run_logged("Urban batch-16 smoke", smoke_command, smoke_log, status_detail="one optimizer step only", retries=0)
    except RuntimeError:
        smoke_failed = True
        write_status("Urban smoke failed", "batch-16 optimizer-step smoke failed; preserving completed ESC outputs")

    if smoke_failed:
        write_status("ESC complete; Urban not launched", "batch-16 smoke failed; preserving ESC outputs for review")
        return

    urban_base = [
        sys.executable, str(ROOT / "scripts/evaluation/run_urbansound8k_compressed.py"),
        "--repo-root", str(ROOT), "--run-id", URBAN_RUN_ID,
        "--device", "cuda", "--num-workers", "2", "--ast-batch-size", "16",
        "--gradient-accumulation-steps", "2", "--feature-batch-size", "16",
        "--ast-input-cache", str(CACHE / "urbansound8k_input_values.pt"),
    ]
    for fold in range(1, 11):
        marker = ROOT / "results/revision/t4_cv" / URBAN_RUN_ID / ("fold{}/fold_complete.json".format(fold))
        if marker.exists():
            continue
        command = urban_base + ["--fold", str(fold)]
        run_logged("Urban fold {}".format(fold), command, PROD / ("urban_fold{}.log".format(fold)), status_detail="AST + LF+broad/CBM downstream", retries=1)

    urban_aggregate = [
        sys.executable, str(ROOT / "scripts/evaluation/aggregate_urban_overnight.py"),
        "--run-dir", str(ROOT / "results/revision/t4_cv" / URBAN_RUN_ID),
        "--output-dir", str(PROD / "urban_official_cv"),
    ]
    run_logged("Urban official aggregation", urban_aggregate, PROD / "urban_aggregate.log", status_detail="verifying all ten official folds", retries=0)
    write_status("all scheduled work complete", "Urban folds completed: {}".format(urban_complete_folds()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("FAILED", "{}: {}".format(type(exc).__name__, exc))
        raise
