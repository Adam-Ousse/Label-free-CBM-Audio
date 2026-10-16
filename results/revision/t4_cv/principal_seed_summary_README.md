# Task 4 principal fixed-split seed robustness

Generated 2026-10-14T17:01:47Z at git commit `34ee4c825fb9b3cdecf467aea6e39b185a98fb73`. This bundle contains the minimal principal-model robustness matrix: five random seeds on each dataset, reusing fixed AST features/checkpoints and varying only the CBM projection/sparse-head seed.

## Principal configurations

- ESC-50: `lf_broad`, the environmental reference vocabulary.
- UrbanSound8K: `lf_broad`, the environmental reference vocabulary.
- CREMA-D: `full`, using the targeted release vocabulary at `release/google_drive_bundle/concept_sets/cremad/full.txt` (701 nonempty concepts). This is not the provenance-confounded CREMA clean-vocabulary comparison.

All 15 runs are valid and contain `full.json`, `mcnemar.json`, `nec25.json`, `nec10.json`, and `nec5.json`, plus the corresponding tensor artifacts where produced. Machine-readable run-level provenance is in `principal_seed_summary.csv` and `.json`; the compact paper table is in `.tex`.

## Summary (mean +/- sample standard deviation)

- esc50 / lf_broad: accuracy 94.25 +/- 0.35%; macro-F1 94.13 +/- 0.39%; seeds 0,1,2,42,123.\n- urbansound8k / lf_broad: accuracy 90.11 +/- 0.53%; macro-F1 91.25 +/- 0.47%; seeds 0,1,2,42,123.\n- cremad / full: accuracy 68.77 +/- 0.61%; macro-F1 46.37 +/- 3.01%; seeds 0,1,2,42,123.\n
The existing ESC-50 runs predate the added per-seed metadata fields, so their exact original launcher command, GPU assignment, and runtime are marked unavailable rather than inferred. New UrbanSound8K and CREMA-D runs record command, seed, git commit, AST checkpoint ID, vocabulary path, device, CUDA visibility, and runtime in each `run_metadata.json`.

## Commands used for new runs

Smoke validation (completed first):

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/.hf_home_task4 HF_HUB_DISABLE_XET=1 bash scripts/launch_task4_cv.sh --execute --resume --run-id t4_principal_urban --part seeds --dataset urbansound8k --variant lf_broad --seed 0
```

Remaining UrbanSound8K seeds (GPU 0), sequentially:

```bash
for seed in 1 2 42 123; do CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/.hf_home_task4 HF_HUB_DISABLE_XET=1 bash scripts/launch_task4_cv.sh --execute --resume --run-id t4_principal_urban --part seeds --dataset urbansound8k --variant lf_broad --seed "$seed"; done
```

CREMA-D seeds (GPU 1), sequentially (the first invocation initializes the run; later invocations resume it):

```bash
for seed in 0 1 2 42 123; do
  if [ "$seed" -eq 0 ]; then resume=""; else resume="--resume"; fi
  CUDA_VISIBLE_DEVICES=1 HF_HOME=/workspace/.hf_home_task4 HF_HUB_DISABLE_XET=1 bash scripts/launch_task4_cv.sh --execute $resume --run-id t4_principal_cremad --part seeds --dataset cremad --variant full --seed "$seed"
done
```

The actual per-seed commands and wall times are preserved in `t4_principal_urban_job_logs/` and `t4_principal_cremad_job_logs/`. No official-fold CV, CREMA speaker-disjoint training, Task 5, or Task 6 jobs were launched in this completion pass.

The migrated CLAP cache had a stale NFS file handle. It was left untouched; the identical canonical CLAP snapshot and the three canonical AST snapshots were populated under `/workspace/.hf_home_task4` and verified with local-only snapshot resolution before the smoke run.
