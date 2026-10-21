# Paper asset inventory

## MAIN PAPER TABLE CANDIDATES

- results/revision/overnight_production_20261015/esc50_official_cv/summary.csv — official 5-fold ESC-50 results.
- results/revision/overnight_production_20261015/urban_official_cv/summary.tex and summary.csv — official 10-fold UrbanSound8K results; retain fold-7 disclosure.
- results/revision/t5_descriptors/controlled_edits/task5_main_table.tex — Task 5 mapped summary.
- results/revision/t6_urbansed/tables/task6_main_quantitative.tex — Task 6 macro comparison.
- results/revision/t7_cremad/speaker_disjoint/table.tex — speaker-disjoint CREMA-D compact comparison.

## MAIN PAPER FIGURE CANDIDATES

- results/revision/t5_descriptors/controlled_edits/controlled_edit_associations.{png,pdf} — Task 5 association summary.
- results/revision/t6_urbansed/figures/task6_quantitative_comparison.{png,pdf} — Task 6 quantitative comparison.
- results/revision/t6_urbansed/figures/task6_concept_response_heatmap.{png,pdf} — Task 6 method/class heatmap.

## APPENDIX TABLE CANDIDATES

- results/revision/t5_descriptors/controlled_edits/descriptor_associations_table.tex — all 21 mapped descriptor tests.
- results/revision/t5_descriptors/controlled_edits/controlled_edit_table.tex — detailed controlled-edit rows.
- results/revision/t5_descriptors/controlled_edits/task5_supplementary_table.tex — Task 5 supplementary detail.
- results/revision/t6_urbansed/tables/task6_supplementary_per_class.tex — Task 6 per-class metrics.
- results/revision/t6_urbansed/tables/task6_per_class_metrics.csv and task6_macro_metrics.csv — machine-readable Task 6 tables.
- results/revision/t4_cv/principal_seed_summary.tex — five-seed robustness.
- results/revision/t3_baselines/t3_gpu_01_fix/metrics.json — control table source.

## APPENDIX FIGURE CANDIDATES

- results/revision/t5_descriptors/controlled_edits/controlled_edit_heatmap.{png,pdf} — Task 5 intervention heatmap.
- results/revision/t5_descriptors/controlled_edits/{esc50,urbansound8k,cremad}_matched_vs_null.{png,pdf} — Task 5 null comparisons.
- results/revision/t6_urbansed/figures/task6_qualitative_examples.{png,pdf} — balanced six-example panel.
- results/revision/t7_cremad/speaker_disjoint/figures/*.{png,pdf} — class-wise qualitative material, appendix only.

## STALE / REMOVE_STALE

- results/revision/t4_cv/urbansound8k_official_cv_overnight_20261015/ABORTED_NOT_USED.md and partial contents — aborted run.
- results/revision/t4_cv/urbansound8k_official_cv/urban_queue.log — zero-byte.
- results/revision/t4_cv/t4_principal_urban/per_fold.csv and t4_principal_cremad/per_fold.csv — zero-byte, non-CV principal runs.
- results/revision/t6_urbansed/preparation_ambiguous_run/* — stale ambiguous preparation.
- results/revision/esc50_compressed_gpu1* — historical partial fold artifacts; use only through consolidated ESC-50 summary.

## MISSING ASSET IF ANY

No scientifically necessary final Task 5/Task 6 PNG/PDF pair was missing. There is no single consolidated main-paper classification table spanning every canonical result; compose it from the whitelisted JSON/CSV sources.
