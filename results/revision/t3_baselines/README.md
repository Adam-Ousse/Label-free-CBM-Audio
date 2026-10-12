# Task 3 baseline matrix

This directory contains the reproducible Task 3 configuration. The matrix is
guarded: `scripts/launch_task3_matrix.sh` prints the plan unless `--execute` is
provided, and every executable run requires a new `--run-id` (or receives a UTC
timestamped one). Existing runs are never overwritten.

The runner reuses the repository's canonical splits:

- ESC-50: `fold1_train`, `fold1_val`, `fold1_test`
- UrbanSound8K: `fold10_train`, `fold10_val`, `fold10_test`
- CREMA-D: `train`, `val`, `test`

It supports CLAP class-prompt zero-shot evaluation, CLAP logistic probes,
elastic-net AST probes, no-class-conditioned-LF CBMs, deterministic random-word
CBMs, and deterministic Gaussian random-direction probes. CBM and random-word
heads are evaluated at `full`, `NEC 25`, `NEC 10`, and `NEC 5`. NEC selection
walks the `glm_saga` path from low to high lambda and takes the first entry with
mean nonzeros per class at or below the requested level. All path and model
selection is validation-only; test is loaded only after selection.

The four no-class-conditioned vocabularies are constructed from the repository's
`concepts_broad.txt` and `concepts_contrastive.txt`. The clean variants retain
only terms present in the Task 2 clean full vocabulary. The exact generated
vocabularies, random seeds, path entries, nonzero counts, and model artifacts
are copied into each timestamped run directory.

## t+3 launch

From the repository root, after the GPU is available:

```bash
bash scripts/launch_task3_matrix.sh --execute --run-id t3_gpu_01
```

This will generate fresh AST/CLAP split features in the run-local cache when
they are not already present. It does not modify canonical release checkpoints.
