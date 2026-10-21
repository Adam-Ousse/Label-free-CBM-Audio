# Instance backup manifest

Read-only destruction-readiness inventory. Nothing was uploaded, moved, or deleted.

## MUST_BACK_UP_BEFORE_DESTROYING_INSTANCE

- release/, including ignored release/google_drive_bundle/ checkpoints and bundle manifest; required by T0, T2, T5, and Task 6 provenance.
- Canonical results/revision/ trees: t0_baseline, t1_silence, t2_leakage, t3_baselines/t3_gpu_01_fix, t4_cv, overnight_production_20261015, t5_descriptors, t6_urbansed, and t7_cremad.
- Ignored dataset manifests and prepared data under data/*/manifests/ and data/urbansed/.
- Ignored saved_activations/ and Task 5 descriptor caches, including results/revision/t5_descriptors_corrected/.
- Task 4 official CV fold checkpoints/features under results/revision/t4_cv/urbansound8k_official_cv_overnight_exactcache_20261015/.
- Task 7 speaker-disjoint manifests, features, checkpoint, and summary.
- Task 6 inference score NPZs, score manifest, provenance/QC JSON, tables, and figures.
- Audit reports in this directory.

## NICE_TO_BACK_UP

- results.stale/, historical interrupted/aborted logs, and forensic copies.
- Environment/version manifests, package files, and git history.
- Uncommitted final-stage scripts: scripts/evaluation/task5_controlled_edits.py, task6_urbansed_inference.py, and task6_urbansed_finalize.py.
- All qualitative figures and PDF/PNG pairs.

## SAFE_TO_LOSE

- __pycache__, .pytest_cache, zero-byte queue/launcher logs, and regenerable temporary logs, after the canonical artifacts above are secured.
- Historical prose-only claims not referenced by the canonical inventory.

## Uncommitted / unbacked-up warning

git status shows modified features/audio_descriptors.py, untracked release/, untracked results.stale/, and untracked final-stage Task 5/Task 6 scripts. Ignored canonical results/data are not represented by the git commit and must be copied independently.
