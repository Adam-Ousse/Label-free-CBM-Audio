# Task 2 retraining: leakage-clean vocabularies

This directory contains the Task 2 retraining outputs for ESC-50, UrbanSound8K,
and CREMA-D. Only the audit-produced vocabularies under
`data/concept_sets/<dataset>/clean/` were used. The canonical AST checkpoints
were frozen and reused; AST was not retrained. No test split was used for
vocabulary selection or hyperparameter tuning.

## Reproduction

Repository: `/workspace/Label-free-CBM-Audio`

Source git commit used before the Task 2 changes: `236010d49a3ae041f01d8952b56dab97c0000a1a`.

Environment:

```text
Python 3.12.14
PyTorch 2.11.0+cu128
CUDA 12.8
GPU NVIDIA RTX 3090 Ti
transformers 5.19.0
scikit-learn 1.9.1
numpy 2.5.3
datasets 5.1.0
```

The fixed seed was `42` for every dataset and variant. Canonical settings were
read from `release/google_drive_bundle/checkpoints/<dataset>/<variant>/args.txt`.
Only the clean concept-set path, revision activation/cache paths, and revision
output paths were changed. The canonical AST backbones, split names, batch
sizes, projection steps (`1000`), SAGA settings, regularization, cutoffs, and
similarity objective were preserved.

Exact commands:

```bash
cd /workspace/Label-free-CBM-Audio
source /venv/main/bin/activate

python -m py_compile scripts/training/run_t2_clean_retraining.py
PYTHONPATH=. pytest -q tests/test_t2_clean_retraining.py tests/test_leakage_audit.py

time -p env PYTHONPATH=. python -m scripts.training.run_t2_clean_retraining \
  --datasets esc50 urbansound8k cremad \
  --variants lf lf_broad lf_contrastive full \
  --device cuda \
  --output-root results/revision/t2_leakage/retrained \
  --cache-root saved_activations/revision/t2_leakage
```

The first invocation stopped after the first ESC-50 fit because the new
validator had an internal weighted-F1 key mismatch. That revision-only output
was resumed after fixing the validator; the successful second invocation took
`5109.75 s` wall time. Including the failed first invocation (`337.20 s`),
total GPU-worker wall time was `5446.95 s` (`90.78 min`). The final runner
metadata records `5101.617 s` for the successful invocation.

## Original versus clean results

Accuracy and macro-F1 are percentages. `V` is vocabulary size; `retained` is
the number of concepts retained in the trained projection; `sparsity` is the
fraction of zero classifier weights; `nonzero/class` is the mean nonzero
classifier weights per class.

| Dataset | Variant | Accuracy original → clean (Δ pp) | Macro-F1 original → clean (Δ pp) | V original → clean | Retained original → clean | Sparsity original → clean | Nonzero/class original → clean |
|---|---|---:|---:|---:|---:|---:|---:|
| ESC-50 | LF | 93.00 → 93.25 (+0.25) | 92.97 → 93.10 (+0.13) | 603 → 501 | 596 → 495 | 91.78% → 89.86% | 48.98 → 50.20 |
| ESC-50 | LF+broad | 93.50 → 93.75 (+0.25) | 93.39 → 93.64 (+0.25) | 683 → 581 | 671 → 571 | 92.58% → 90.71% | 49.80 → 53.02 |
| ESC-50 | LF+contrastive | 93.25 → 93.75 (+0.50) | 93.01 → 93.56 (+0.55) | 700 → 595 | 689 → 584 | 92.82% → 90.84% | 49.50 → 53.48 |
| ESC-50 | full | 92.75 → 91.75 (-1.00) | 92.70 → 91.52 (-1.18) | 777 → 673 | 761 → 662 | 93.13% → 91.72% | 52.26 → 54.84 |
| UrbanSound8K | LF | 89.37 → 89.25 (-0.12) | 90.60 → 90.25 (-0.35) | 140 → 127 | 133 → 122 | 81.88% → 75.41% | 24.10 → 30.00 |
| UrbanSound8K | LF+broad | 89.84 → 89.61 (-0.24) | 90.94 → 90.58 (-0.35) | 224 → 211 | 212 → 199 | 86.04% → 83.87% | 29.60 → 32.10 |
| UrbanSound8K | LF+contrastive | 89.73 → 89.01 (-0.72) | 90.76 → 90.20 (-0.56) | 185 → 172 | 178 → 167 | 86.07% → 82.10% | 24.80 → 29.90 |
| UrbanSound8K | full | 89.84 → 89.01 (-0.84) | 90.78 → 90.18 (-0.60) | 269 → 256 | 255 → 245 | 87.69% → 86.94% | 31.40 → 32.00 |
| CREMA-D | LF | 69.44 → 65.08 (-4.37) | 50.65 → 40.01 (-10.63) | 52 → 52 | 311 → 16 | 88.00% → 25.00% | 37.33 → 12.00 |
| CREMA-D | LF+broad | 69.78 → 68.64 (-1.14) | 50.69 → 47.43 (-3.25) | 136 → 136 | 481 → 49 | 91.34% → 46.26% | 41.67 → 26.33 |
| CREMA-D | LF+contrastive | 69.51 → 66.69 (-2.82) | 49.86 → 45.45 (-4.41) | 79 → 79 | 90.07% → 32.64% | 39.33 → 16.17 |
| CREMA-D | full | 70.25 → 67.63 (-2.62) | 51.58 → 47.43 (-4.15) | 162 → 162 | 548 → 51 | 92.79% → 49.35% | 39.50 → 25.83 |

Machine-readable versions are in:

- `comparison.csv` and `comparison.json`
- `checkpoint_validation.json` (`pass: true`)
- `run_metadata.json`

## CREMA-D equivalence and interpretation

For all four CREMA-D variants, the clean vocabulary is content-equivalent to
the audit original vocabulary (`52`, `136`, `79`, and `162` concepts); the only
byte difference is the clean file's terminal-newline formatting. However, the
canonical release checkpoints are not equivalent to those audit-original
vocabularies: their `args.txt` concept-set sources are the separate
`cremad_targeted_rerun_20260828/generation/source_ablations_canonical` files,
and their checkpoint concept lists contain `311`, `481`, `396`, and `548`
retained concepts with zero overlap with the audit originals. Therefore the
canonical checkpoint input equivalence check correctly failed and CREMA-D was
retrained rather than incorrectly reusing a mismatched release checkpoint.

The CREMA-D clean-vs-release deltas above are consequently not a pure
leakage-removal effect. They reflect a provenance mismatch between the release
checkpoint and the audit vocabulary, despite the clean and audit-original
vocabulary files being content-identical. No concepts were restored and no
audit threshold was changed.

## Validation, preservation, and warnings

All 12 resulting checkpoints passed required-file checks, tensor-shape checks,
direct frozen-test inference, and recorded/direct metric agreement; see
`checkpoint_validation.json`. New checkpoints are under
`results/revision/t2_leakage/retrained/<dataset>/<variant>/models/`, and new
frozen-feature caches are under `saved_activations/revision/t2_leakage/`.

The canonical `release/google_drive_bundle/`, historical result directories,
and `paper/main.tex` were not modified. The run emitted expected Hugging Face
unauthenticated-request warnings and expected AST loader reports about the
classifier head being unused while loading the frozen backbone; neither affects
the frozen AST feature path. No test-set tuning was performed.

**Does the leakage-cleaned models remain competitive?** Yes for ESC-50 and
UrbanSound8K: changes are small (at most -1.00 accuracy points and -1.18
macro-F1 points for ESC-50 full; at most -0.84/-0.60 for UrbanSound8K full),
and the models remain in the same performance range. CREMA-D is not a valid
leakage-cleaning comparison because of the checkpoint-input provenance failure
described above; its large apparent drops must not be interpreted as evidence
that leakage removal caused them.
