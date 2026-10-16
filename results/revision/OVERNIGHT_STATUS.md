# Overnight production status

Last update: 2026-10-16 21:47:04 UTC

## Current state

- Stage: Urban fold 3
- Detail: AST + LF+broad/CBM downstream; attempt 1/2
- Supervisor PID: `93320`; persistent HF cache: `/workspace/.hf_home_production`.
- ESC fold 5: run `esc50_compressed_gpu1_fold5_concurrent`; latest completed AST epoch `30/30`; fold-complete marker: `True`.
- ESC fold 4: run `esc50_compressed_gpu1_overnight_exactcache_20261015`; fold-complete marker: `True`.
- ESC official aggregation: `True`.
- Urban run: `urbansound8k_official_cv_overnight_exactcache_20261015`; completed official folds: `[1, 2]`; remaining: `[3, 4, 5, 6, 7, 8, 9, 10]`.
- Urban current fold/epoch: `3/2/30`; ETA estimate: `~03:24 remaining`.
- Urban completed mean accuracy: AST `0.8162`; LF+broad `0.8275`.
- Errors/retries: `none recorded`; see supervisor and per-fold logs.
- Aborted Urban attempt: preserved and excluded at `results/revision/t4_cv/urbansound8k_official_cv_overnight_20261015/ABORTED_NOT_USED.md`.

## Exact execution configuration

- ESC cache: `cache/ast_inputs/esc50_input_values.pt`, global exact float32 `(2000, 1024, 128)`.
- Urban cache: `cache/ast_inputs/urbansound8k_input_values.pt`, global exact float32 `(8732, 1024, 128)`.
- ESC fold 4: test fold 4, validation fold 5, batch 8, accumulation 1, 30 epochs, seed 42, original optimizer/LR/scheduler.
- Urban replacement CV: test fold 1→10, next cyclic validation, physical batch 16 × accumulation 2 = effective batch 32, 30 epochs, seed 42, original optimizer/LR/scheduler.
- No AMP/BF16/FP16, TF32, compile, alternative attention, fused optimizer, early stopping, or checkpoint-resume semantics.

## Logs

- Supervisor: `overnight_supervisor.log`
- ESC fold 4: `esc_fold4.log`
- Urban per-fold logs: `urban_foldN.log`
