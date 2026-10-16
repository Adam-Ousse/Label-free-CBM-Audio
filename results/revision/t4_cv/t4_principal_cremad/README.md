# Task 4 predictive robustness and official-fold CV

This run is predictive robustness only. The strong Task 3 random-word and
random-direction controls mean that CV accuracy is not evidence of semantic
interpretability.

## Reproduction and resume

```bash
cd /workspace/Label-free-CBM-Audio
bash scripts/launch_task4_cv.sh --execute --run-id t4_principal_cremad
# resume an interrupted run:
bash scripts/launch_task4_cv.sh --execute --run-id t4_principal_cremad --resume
```

Seeds: `0, 1, 2, 42, 123`. Fixed-split robustness reuses the Task 3 AST/CLAP feature
caches read-only. Official ESC-50 folds are test folds 1--5 with the next
cyclic fold as validation. Official UrbanSound8K folds are test folds 1--10
with the next cyclic fold as validation. Training and checkpoint selection use
validation only. AST settings are the existing training-script defaults:
30 epochs, batch size 32, AdamW learning rate 5e-5, weight decay 1e-4,
warmup ratio 0.1, seed 42; UrbanSound8K epochs: 30.
CBM settings reuse Task 3: projection steps 1000, interpretability cutoff
0.45, activation cutoff 0.25, elastic-net alpha 0.99, 25 glm_saga path
points, 1000 SAGA iterations, and validation-selected full-path entry.

Task 2 clean candidate: `broad_plus_contrastive_clean` uses
`data/concept_sets/<dataset>/clean/concepts_ablation_full.txt`. CREMA-D
original-vs-clean interpretation remains provenance-confounded as documented
in Task 2; this run does not claim leakage-removal causality for it.

The per-seed and per-fold tables, aligned predictions, and exact model/cache
paths are machine-readable in this directory. Failed or retried jobs remain
in their isolated directory's `train.log`; existing complete artifacts are
validated before being skipped.

Rows generated so far: 5 fixed-split seed rows and 0
official-fold CBM rows.

## Aggregate results

| Dataset | Variant | Analysis | Accuracy mean ± std | Macro-F1 mean ± std |
|---|---|---|---:|---:|
| cremad | full | 5 seeds | 0.6877 ± 0.0061 | 0.4637 ± 0.0301 |
| _pending_ | _pending_ | official folds | _pending_ | _pending_ |

## Required interpretation answers

1. **α-CBM versus CLAP zero-shot, CLAP linear probe, and sparse AST probe:**
the fixed-split comparison is the Task 3 predeclared comparison in
`results/revision/t3_baselines/README.md` and its machine-readable table. This
Task 4 run adds predictive variance and does not retune or replace those
baselines. Any claim of outperforming or remaining competitive must use that
fixed-split table together with these uncertainty results.

2. **Random controls:** Task 3 showed strong random-word and random-direction
controls. They are therefore treated as competitive predictive controls, not
semantic evidence; this Task 4 CV report does not reinterpret them.

3. **NEC capacity control:** Task 3 contains full/NEC25/NEC10/NEC5 results and
actual mean nonzeros/class. The Task 4 CBM fits use the same validation-only
glm_saga path and report their selected full-path sparsity here; no test tuning
was introduced.

4. **No-class-conditioned LF:** Task 3 is the predeclared broad-only and
broad+contrastive comparison. Task 4 evaluates the selected broad-plus-
contrastive clean candidate in official folds; it is predictive robustness,
not an interpretability claim.

5. **Task 4(f) candidate for Task 4 CV:** `broad_plus_contrastive_clean` was
fixed before this CV run by the Task 3 evaluation. No post-hoc fold result was
used to select it.

The original-vs-clean CREMA-D comparison is provenance-confounded because the
release checkpoint vocabulary provenance does not align with the audit
vocabulary. It must not be described as an isolated leakage-removal effect.
