# Task 1 — Silence and padding in segmented inference

Status: PASS. This directory contains the complete Task 1 run only. `paper/main.tex` was not modified, and Task 0 outputs were read but not overwritten.

## Reproduction command

The repository-local alpha-CBM revision brief was not present in this checkout; the Task 1 protocol below follows the wave specification supplied for this run.

Exact environment/test commands:

```bash
cd /workspace/Label-free-CBM-Audio
source /venv/main/bin/activate
PYTHONPATH=. pytest -q tests/test_silence_padding.py
python -m py_compile data_utils.py scripts/evaluation/silence_padding.py scripts/evaluation/run_silence_padding_ablation.py
```

Exact complete inference command:

```bash
cd /workspace/Label-free-CBM-Audio
source /venv/main/bin/activate
time -p env PYTHONPATH=. python -m scripts.evaluation.run_silence_padding_ablation \
  --artifact-bundle release/google_drive_bundle \
  --output-root results/revision/t1_silence \
  --cache-root saved_activations/revision/t1_silence \
  --baseline-summary results/revision/t0_baseline/segmented/summary.json \
  --datasets esc50 urbansound8k cremad \
  --window-sec 1.0 --hop-sec 0.5 --lme-alpha 5.0 \
  --device cuda --batch-size 8 --segment-batch-size 64 \
  --num-workers 2 --seed 42
```

Seed: `42`. Windowing: 1.0 s windows, 0.5 s hop, LME alpha 5.0. Release artifact bundle: `release/google_drive_bundle`; its canonical manifest/checkpoint paths were used without modification.

## Environment and runtime

- Git commit for this Task 1 implementation: `b61bd8c222d31b2842ce0803de6a2d14e6da183e`; parent Task 0 baseline commit: `4154ab0dbd1ecc1a827e3456fd4211bd509094b7`.
- Python 3.12.14; PyTorch 2.11.0+cu128; CUDA 12.8; transformers 5.19.0; scikit-learn 1.9.1; NumPy 2.5.3; datasets 5.1.0.
- GPU: NVIDIA GeForce RTX 3090 Ti, driver 610.57.04, 24564 MiB.
- Cache construction GPU wall time: 1051.61 s (17m31.61s).
- Final cached evaluation/diagnostic pass: 13.41 s.
- Successful artifact-producing GPU-worker total: 1065.02 s (17m45.02s); the cached evaluation pass itself uses CUDA for projection/pooling.

## Protocol implemented

`data_utils.py` now preserves each manifest `duration` as `duration_sec` through collation. Window validity is computed from true duration, not from the padded tensor length. The pure functions in `scripts/evaluation/silence_padding.py` implement:

- padding mask: retain windows with nonzero overlap with the true duration;
- silence mask: compute RMS dBFS only over true, non-padding samples and mask windows below the threshold;
- energy-weighted pooling: normalized RMS energy multiplied by the true-duration fraction;
- silence-baseline: replace masked windows with the AST representation of a zero 1-second waveform before pooling;
- all-masked fallback: retain the highest-RMS window.

Both `-50 dBFS` and `-60 dBFS` are evaluated explicitly. Threshold selection uses validation macro-F1 averaged over AST, all four CBM variants (`lf`, `lf_broad`, `lf_contrastive`, `full`), and mean/max/LME pools only. Test labels are not used for threshold selection. Validation selected `-50 dBFS` for ESC-50, UrbanSound8K, and CREMA-D.

The zero-waveform diagnostic records the 1-second all-zero waveform through `AST -> W_c -> standardization -> W_F` for all 12 release models in `zero_waveform_diagnostic.json`.

## Key test numbers

The table reports mean-pool accuracy averaged over the AST and four CBM variants for each dataset. `energy_weighted` has one energy-weighted pool per model and is shown separately. Values are percentages; the complete per-model/per-pool table is in `segmented/summary.csv`.

| Dataset | Unmasked | Padding mask | Silence mask -50 | Silence mask -60 | Silence baseline -50 | Silence baseline -60 | Energy-weighted |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESC-50 | 81.150 | 81.150 | 85.700 | 84.750 | 80.950 | 81.100 | 87.500 |
| UrbanSound8K | 83.250 | 87.599 | 87.790 | 87.742 | 81.553 | 82.700 | 87.957 |
| CREMA-D | 59.610 | 62.364 | 62.431 | 62.485 | 59.570 | 59.610 | 67.011 |

CREMA-D mean-pool accuracy by strategy is also reflected in macro-F1: unmasked `25.017%`, padding mask `29.977%`, silence mask `-50/-60 dBFS` `30.266/30.157%`, silence baseline `-50/-60 dBFS` `25.005/25.017%`, and energy-weighted `40.546%`.

## CREMA-D interpretation

**Does padding/silence masking substantially close the CREMA-D segmented-performance gap?** No, not by itself. The corresponding global mean accuracy across AST and four CBMs is `69.805%`. Unmasked segmented mean-pool accuracy is `59.610%`, leaving a `10.195` percentage-point gap. Padding masking leaves `7.441` points; silence masking leaves `7.374` points at `-50 dBFS` and `7.320` points at `-60 dBFS`. Thus masking closes roughly 27–28% of the gap, not a majority. Silence-baseline does not improve the result. Energy-weighted pooling separately narrows the gap to `2.794` points, but that gain is from weighting rather than evidence that binary padding/silence masking explains the original CREMA-D gap.

## Baseline preservation and artifacts

The unmasked rows were compared with every corresponding Task 0 segmented row: 180 metric checks across AST/CBM models, pools, and accuracy/macro-F1/weighted-F1/balanced-accuracy. `baseline_preservation.json` reports `pass: true` and maximum absolute delta `0.0`.

Machine-readable outputs:

- `validation/summary.json` and `validation/summary.csv`: validation-only strategy results.
- `validation/threshold_selection.json`: selected thresholds and selection scores.
- `segmented/summary.json` and `segmented/summary.csv`: 375 test rows covering AST plus all four CBM variants, explicit thresholds, selected thresholds, and all required pools.
- `baseline_preservation.json`: exact unmasked comparison against Task 0.
- `zero_waveform_diagnostic.json`: 12 AST → `W_c` → standardization → `W_F` diagnostics.
- `diagnostics/`: 285 confusion matrices and 285 predicted-class histograms, one CSV pair per saved model/method/pool diagnostic path.
- `saved_activations/revision/t1_silence/`: six segmented feature/AST-logit caches for validation and test splits.

Unit tests: `tests/test_silence_padding.py` — 8 passed. Warnings: the transformers AST backbone loader reports the classifier head as unexpected because the classifier head is intentionally loaded separately from `ASTForAudioClassification`; this is expected and does not change the reproduced unmasked baseline. No canonical checkpoint or Task 0 result was changed.
