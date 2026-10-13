# Task 3 baseline matrix

The completed run is `t3_gpu_01_fix`. The authoritative machine-readable table
is [table.csv](table.csv), with a LaTeX rendering in [table.tex](table.tex).
All per-run JSONs, checkpoints, caches, exact generated vocabularies, and seeds
are under [t3_gpu_01_fix/](t3_gpu_01_fix/).

## Reproduction

Fresh complete run:

```bash
cd /workspace/Label-free-CBM-Audio
bash scripts/launch_task3_matrix.sh --execute --run-id t3_gpu_01
```

Continuation of an interrupted run:

```bash
bash scripts/launch_task3_matrix.sh --execute \
  --run-id t3_gpu_01_fix --resume --only-dataset cremad
```

The runner uses canonical splits: ESC-50 `fold1_train/fold1_val/fold1_test`,
UrbanSound8K `fold10_train/fold10_val/fold10_test`, and CREMA-D
`train/val/test`. Every lambda, path, and model choice uses validation data;
test data is evaluated only after selection.

Prompts are exactly `the sound of {class}` for ESC-50 and UrbanSound8K, and
`a person speaking in a {class} voice` for CREMA-D. Random words use seed 2028;
random directions use Gaussian seed 2027. Exact vocabularies are recorded as
`.txt` and `.json` files in each run.

NEC selection follows the VLG-CBM-style rule: traverse the glm_saga path toward
increasing regularization and select the first entry whose actual mean number of
nonzero head weights per class is at most the requested NEC. The table reports
the actual mean reached, not just the target.

## Held-out results

Accuracy / macro-F1 for full-capacity baseline rows:

| Dataset | CLAP zero-shot | CLAP linear | Sparse AST | Random direction |
|---|---:|---:|---:|---:|
| ESC-50 | 80.25 / 78.06 | **95.75 / 95.57** | 93.75 / 93.51 | 95.50 / 95.34 |
| UrbanSound8K | 75.87 / 73.74 | 88.65 / 89.36 | **89.84 / 91.07** | 89.84 / 90.95 |
| CREMA-D | 17.39 / 7.47 | 68.84 / 42.52 | **70.45 / 48.83** | 69.31 / 43.82 |

Full-capacity no-class-conditioned α-CBM rows:

| Dataset | Broad-only original | Broad-only clean | Broad+contrastive original | Broad+contrastive clean |
|---|---:|---:|---:|---:|
| ESC-50 | 92.25 / 92.17 | 92.25 / 91.96 | **92.50 / 92.29** | 92.25 / 91.97 |
| UrbanSound8K | 87.10 / 88.26 | 87.10 / 88.26 | **89.37 / 90.58** | 89.01 / 90.05 |
| CREMA-D | 69.51 / 45.06 | **69.51 / 45.06** | 68.91 / 38.39 | 68.91 / 38.39 |

Values are `accuracy / macro-F1`; the complete table includes every NEC row,
actual mean NNZ/class, and bottleneck/vocabulary dimensions.

## Required conclusions

1. **α-CBM versus learned baselines.** The no-class α-CBMs remain competitive
   with CLAP zero-shot and clearly outperform it on every dataset. They are
   below the strongest CLAP linear probe on ESC-50 and below sparse AST on
   CREMA-D, while broad+contrastive α-CBM is competitive with sparse AST on
   UrbanSound8K (89.37% vs 89.84%). Thus α-CBM is competitive with the learned
   baselines on UrbanSound8K, but does not dominate CLAP linear or sparse AST.

2. **Random controls.** Random-direction is strong: it matches sparse AST
   accuracy on UrbanSound8K and is close to CLAP linear on ESC-50, while below
   sparse AST on CREMA-D. Random-word full models are also competitive (93.75%
   ESC-50, 89.49% UrbanSound8K, 70.32% CREMA-D), so bottleneck width and the
   shared projection/head pipeline explain substantial full-capacity accuracy.
   Their NEC rows degrade substantially, providing the capacity-controlled
   comparison.

3. **NEC capacity control.** Performance generally falls as requested NEC
   decreases. The actual values are in `table.csv`; for example,
   broad+contrastive-clean α-CBM reaches 6.58/24.00/24.00 NNZ per class on
   ESC-50/UrbanSound8K/CREMA-D at NEC25, and 6.58/0.90/10.00 at NEC10. α-CBM
   does not retain full-capacity accuracy uniformly under aggressive NEC
   control; this negative result is reported unchanged.

4. **No class-conditioned LF concepts.** Broad-only models remain competitive
   with CLAP zero-shot and are close to broad+contrastive on ESC-50. On
   UrbanSound8K, broad+contrastive is better than broad-only by about 2.3
   accuracy points. On CREMA-D, broad-only is slightly better in this run. No
   class-conditioned LF vocabulary is required for a useful model, although
   contrastive additions help on UrbanSound8K.

5. **Task 4 CV candidate.** The fixed carry-forward rule is the equal-dataset
   mean macro-F1 across `full` and `NEC25`, with `NEC10` and `NEC5` as tie-breaks.
   Under this rule, `broad_plus_contrastive_clean` is the best Task 3(f)
   candidate (primary mean macro-F1 0.6190). This is one cross-dataset rule;
   no dataset-specific or test-set tweaking was used.

## Scientific warning

Task 2 established that the canonical CREMA-D release checkpoint vocabulary has
unaligned provenance relative to the audit vocabulary. CREMA-D original-vs-clean
rows are therefore **provenance-confounded** and must not be interpreted as a
clean leakage-removal effect. ESC-50 and UrbanSound8K provenance was aligned.

The first attempted run `t3_gpu_01` was excluded after a dataset-cache namespace
collision reused an ESC-50 random-word text embedding for UrbanSound8K. The
corrected run uses dataset-scoped caches and is the only run summarized here.

The run directory timestamps span 15:27:44 wall-clock, including the intentional
overnight pause. Exact per-run files and logs are retained for auditability.
