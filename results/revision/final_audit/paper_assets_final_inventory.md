# Final manuscript-facing figure and table inventory

Generated from frozen canonical results. No scientific result files were modified.

- Plotter: `scripts/visualization/export_final_paper_assets.py`
- Physical figure files: **43**
- Table files: **16**

## Cleaned figure replacements

- `paper_assets_final/figures/task5_controlled_edit_associations.pdf` and `.png` — main; Task 5 controlled descriptor and intervention validation; **replacement**.
- `paper_assets_final/figures/task5_controlled_edit_heatmap.pdf` and `.png` — appendix; Task 5 controlled descriptor and intervention validation; **replacement**.
- `paper_assets_final/figures/task5_cremad_matched_vs_null.pdf` and `.png` — appendix; Task 5 controlled descriptor and intervention validation; **replacement**.
- `paper_assets_final/figures/task5_esc50_matched_vs_null.pdf` and `.png` — appendix; Task 5 controlled descriptor and intervention validation; **replacement**.
- `paper_assets_final/figures/task5_urbansound8k_matched_vs_null.pdf` and `.png` — appendix; Task 5 controlled descriptor and intervention validation; **replacement**.
- `paper_assets_final/figures/task6_concept_response_heatmap.pdf` and `.png` — main; Task 6 URBAN-SED temporal localization; **replacement**.
- `paper_assets_final/figures/task6_qualitative_examples.pdf` and `.png` — appendix; Task 6 URBAN-SED temporal localization; **replacement**.
- `paper_assets_final/figures/task6_quantitative_comparison.pdf` and `.png` — main; Task 6 URBAN-SED temporal localization; **replacement**.

## Retained legacy valid figures

- `main_figures/architecture_overview.png` — main; architecture/model overview; **legacy valid**.
- `appendix_figures/older_esc50_lf_broad_temporal.png` — appendix; older valid temporal example; **legacy valid**.
- `appendix_figures/older_esc50_prediction_example.png` — appendix; older valid prediction example; **legacy valid**.
- `appendix_figures/cremad_speaker_disjoint/` — appendix; class-wise speaker-disjoint qualitative panels; **legacy valid**.

## Tables

- `tables/principal_seed_summary.csv` — appendix; five-seed robustness; **new_revision**.
- `tables/principal_seed_summary.tex` — appendix; five-seed robustness; **new_revision**.
- `tables/t3_controls_table.csv` — appendix; random-word/random-direction/CLAP controls; **new_revision**.
- `tables/t3_controls_table.tex` — appendix; random-word/random-direction/CLAP controls; **new_revision**.
- `tables/task5_controlled_edit_table.tex` — appendix; Task 5 semantic descriptor and controlled-intervention validation; **new_revision**.
- `tables/task5_descriptor_associations.tex` — appendix; Task 5 semantic descriptor and controlled-intervention validation; **new_revision**.
- `tables/task5_supplementary_table.tex` — appendix; Task 5 semantic descriptor and controlled-intervention validation; **new_revision**.
- `tables/task6_macro_metrics.csv` — appendix; Task 6 URBAN-SED temporal localization; **new_revision**.
- `tables/task6_per_class_metrics.csv` — appendix; Task 6 URBAN-SED temporal localization; **new_revision**.
- `tables/task6_supplementary_per_class.tex` — appendix; Task 6 URBAN-SED temporal localization; **new_revision**.
- `tables/cremad_speaker_disjoint_table.tex` — main; CREMA-D speaker-disjoint evaluation; **new_revision**.
- `tables/esc50_official_cv_summary.csv` — main; official ESC-50 5-fold CV; **new_revision**.
- `tables/task5_main_table.tex` — main; Task 5 semantic descriptor and controlled-intervention validation; **new_revision**.
- `tables/task6_main_quantitative.tex` — main; Task 6 URBAN-SED temporal localization; **new_revision**.
- `tables/urbansound8k_official_cv_summary.csv` — main; official UrbanSound8K 10-fold CV; **new_revision**.
- `tables/urbansound8k_official_cv_summary.tex` — main; official UrbanSound8K 10-fold CV; **new_revision**.

## Faithfulness provenance note

- No canonical final faithfulness result is present. The available evaluator references older August checkpoints, so no CPU reevaluation was run. Existing faithfulness language should be limited to decision-level controllability/faithfulness rather than strong semantic concept-faithfulness.

Machine-readable versions: `paper_assets_final_inventory.json` and `paper_assets_final_inventory.csv`.
