# Overnight production status

Last update: 2026-10-20 09:06:38 UTC

## Current state

- Stage: all scheduled Urban work complete
- Detail: fold 7 salvaged; folds 8-10 complete; 10-fold aggregation written
- Supervisor PID: `22408`; detached Urban continuation; one GPU-heavy fold at a time.
- ESC folds 4 and 5: preserved complete; ESC aggregation: `True`.
- Urban run: `urbansound8k_official_cv_overnight_exactcache_20261015`; completed folds: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`; remaining: `none`.
- Urban current fold/epoch: `none/none`; ETA estimate: `complete`.
- Fold 7 status: `SALVAGED_TRUNCATED_24_OF_30` once its marker exists; not represented as a normal 30-epoch completion.
- Failures/retries: `none recorded by resumed supervisor`; persistent per-fold logs are authoritative.
- Fold 7 disclosure: `Fold 7 training was interrupted after epoch 24; its best validation checkpoint among epochs 1–24 was retained. Completed folds showed no meaningful late-epoch improvement after epoch 24.`

## Exact execution configuration

- Urban cache: `/workspace/Label-free-CBM-Audio/results/revision/overnight_production_20261015/cache/ast_inputs/urbansound8k_input_values.pt`; exact cached float32 AST inputs.
- Urban official CV: test fold f, validation fold f+1 cyclic, seed 42, 30 AST epochs, physical batch 16 × accumulation 2 = effective batch 32, num_workers 2, feature batch 16.
- AST base model: `MIT/ast-finetuned-audioset-10-10-0.4593`; optimizer/LR/scheduler and validation-only selection unchanged.
- No AMP/BF16/FP16, TF32, compile, alternative attention, fused optimizer, early stopping, or checkpoint resume.

## Logs and outputs

- Continuation log: `results/revision/overnight_production_20261015/urban_resume_supervisor.log`.
- Fold AST logs: `results/revision/t4_cv/urbansound8k_official_cv_overnight_exactcache_20261015/ast/urbansound8k/foldN/foldN/train.log`.
- Aggregation output: `results/revision/overnight_production_20261015/urban_official_cv/`.
