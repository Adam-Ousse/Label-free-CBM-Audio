# Checkpoint audit

This inventory classifies reproducibility dependencies without deleting or modifying any checkpoint.

## KEEP_FOR_BACKUP

- `release/google_drive_bundle/checkpoints/esc50/{lf,lf_broad,lf_contrastive,full}/` — canonical T0/T2/T5 classification and controlled-edit checkpoints; approximately 1.9–2.5 MB per directory.
- `release/google_drive_bundle/checkpoints/urbansound8k/{lf,lf_broad,lf_contrastive,full}/` — canonical T0/T2/T5 and Task 6 α-CBM checkpoint family; approximately 0.44–0.80 MB per directory. Task 6 specifically uses `full`, 255 retained concepts, seed 42, fold-10 test protocol.
- `release/google_drive_bundle/checkpoints/cremad/{lf,lf_broad,lf_contrastive,full}/` — canonical T0/T2/T5 checkpoint family; approximately 1.0–1.7 MB per directory.
- `results/revision/t4_cv/urbansound8k_official_cv_overnight_exactcache_20261015/ast/urbansound8k/fold{1..10}/fold*/best_model/` — official UrbanSound8K CV AST checkpoints, about 3.6 GB total. Fold 7 is a valid retained best checkpoint from epochs 1–24 after interruption and must be preserved with its disclosure.
- `results/revision/t2_leakage/retrained/{esc50,urbansound8k,cremad}/{lf,lf_broad,lf_contrastive,full}/models/` — 12 leakage-cleaned retrained CBM checkpoints required to substantiate T2; approximately 80 KB–2.2 MB per model directory.
- `results/revision/t7_cremad/speaker_disjoint/ast/test/best_model/` — speaker-disjoint AST checkpoint, approximately 329 MB.
- `results/revision/t7_cremad/speaker_disjoint/cbm/full.pt` plus `selection_validation.json`, manifest verification, and speaker-disjoint manifests — targeted-full speaker-disjoint α-CBM, approximately 1.5 MB.
- Task 3 control checkpoints and feature/text caches under `results/revision/t3_baselines/t3_gpu_01_fix/` — required for the random-word/random-direction and CLAP control claims; preserve the whole directory rather than selecting individual files.
- The external AST provenance for Task 6, `Adam-ousse/ast-urbansound8k-finetuned-fold10`, together with the local Hugging Face cache if present — required to reproduce Task 6 AST-logit, occlusion, and attention-rollout baselines. The Task 6 provenance JSON records the exact model identifier and class order.

## OPTIONAL

- Per-fold features and prediction tensors under the official CV trees after the fold checkpoints and consolidated summaries are secured.
- Redundant T4 principal-seed model copies once `principal_seed_summary.json` and its per-seed result directories are backed up.
- T2 intermediate caches once the retrained checkpoints, comparison JSON, run metadata, and checkpoint validation are backed up.

## SAFE_TO_IGNORE

- No checkpoint is recommended for deletion during this audit. Only explicitly aborted/partial run copies may be treated as non-paper sources, and they should be retained as forensic artifacts until independent backup is complete.

## Provenance warnings

- `release/` is untracked and its canonical bundle is ignored by git.
- Task 4 fold-7 status is `SALVAGED_TRUNCATED_24_OF_30`, not a normal 30-epoch completion.
- Task 6 did not select any checkpoint using URBAN-SED results; its provenance explicitly records canonical release selection.
