# Final results inventory

- Audit commit: 27da0a9bcb9516601e14aa394b719766e24586e3
- Rows: **568**
- Generated from persisted machine-readable artifacts; no inference was run.

## Canonical source files

- results/revision/t0_baseline/global/global_metrics.json
- results/revision/t1_silence/baseline_preservation.json
- results/revision/t2_leakage/retrained/comparison.json
- results/revision/t3_baselines/t3_gpu_01_fix/metrics.json
- results/revision/t4_cv/principal_seed_summary.json
- results/revision/overnight_production_20261015/esc50_official_cv/summary.json
- results/revision/overnight_production_20261015/urban_official_cv/summary.json
- results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json
- results/revision/t5_descriptors/controlled_edits/task5_metadata.json
- results/revision/t6_urbansed/analysis/task6_final_metrics.json
- results/revision/t6_urbansed/analysis/task6_qc.json
- results/revision/t7_cremad/speaker_disjoint/summary.json

## Final rows

- t4_official_ESC-50_AST_accuracy — ESC-50 / AST / accuracy: **0.9135**; uncertainty {'sd': 0.021837467801922472, 'min': 0.885, 'max': 0.935, 'folds': 5}; source results/revision/overnight_production_20261015/esc50_official_cv/summary.json
- t4_official_ESC-50_AST_macro_f1 — ESC-50 / AST / macro_f1: **0.912013**; uncertainty {'sd': 0.022415699517586454, 'min': 0.8820123363049057, 'max': 0.9328086260145083, 'folds': 5}; source results/revision/overnight_production_20261015/esc50_official_cv/summary.json
- t4_official_ESC-50_lf_broad_accuracy — ESC-50 / lf_broad / accuracy: **0.9175**; uncertainty {'sd': 0.027556759606310797, 'min': 0.8875, 'max': 0.9425, 'folds': 5}; source results/revision/overnight_production_20261015/esc50_official_cv/summary.json
- t4_official_ESC-50_lf_broad_macro_f1 — ESC-50 / lf_broad / macro_f1: **0.915574**; uncertainty {'sd': 0.029465740729485677, 'min': 0.8821277767639997, 'max': 0.9421501566888564, 'folds': 5}; source results/revision/overnight_production_20261015/esc50_official_cv/summary.json
- t4_official_UrbanSound8K_AST_accuracy — UrbanSound8K / AST / accuracy: **0.839795**; uncertainty {'sd': 0.05061830885222845, 'min': 0.7264864864864865, 'max': 0.9145299145299145, 'folds': 10}; source results/revision/overnight_production_20261015/urban_official_cv/summary.json
- t4_official_UrbanSound8K_AST_macro_f1 — UrbanSound8K / AST / macro_f1: **0.851536**; uncertainty {'sd': 0.04827862559209014, 'min': 0.7370874371586853, 'max': 0.9191607868884457, 'folds': 10}; source results/revision/overnight_production_20261015/urban_official_cv/summary.json
- t4_official_UrbanSound8K_lf_broad_accuracy — UrbanSound8K / lf_broad / accuracy: **0.848903**; uncertainty {'sd': 0.05128722167467247, 'min': 0.7394594594594595, 'max': 0.9220085470085471, 'folds': 10}; source results/revision/overnight_production_20261015/urban_official_cv/summary.json
- t4_official_UrbanSound8K_lf_broad_macro_f1 — UrbanSound8K / lf_broad / macro_f1: **0.860045**; uncertainty {'sd': 0.046482947769155715, 'min': 0.7591849798758128, 'max': 0.9253906555464685, 'folds': 10}; source results/revision/overnight_production_20261015/urban_official_cv/summary.json
- t5_cremad_expected_sign_rate — cremad / all four established variants / expected_sign_success_rate: **0.535714**; uncertainty {'success': 90, 'tests': 168}; source results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json
- t5_esc50_expected_sign_rate — esc50 / all four established variants / expected_sign_success_rate: **0.527778**; uncertainty {'success': 57, 'tests': 108}; source results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json
- t5_urbansound8k_expected_sign_rate — urbansound8k / all four established variants / expected_sign_success_rate: **0.472222**; uncertainty {'success': 34, 'tests': 72}; source results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json
- t5_null_pairings — ESC-50 + UrbanSound8K + CREMA-D / all mapped models / random_pairing_count: **12000**; uncertainty None; source results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json
- t6_0.25_ast_attention_rollout_macro_auroc — URBAN-SED / ast_attention_rollout / auroc: **0.525651**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_attention_rollout_macro_average_precision — URBAN-SED / ast_attention_rollout / average_precision: **0.135547**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_attention_rollout_macro_pointing_game — URBAN-SED / ast_attention_rollout / pointing_game: **0.27389**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_occlusion_drop_macro_auroc — URBAN-SED / ast_occlusion_drop / auroc: **0.601191**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_occlusion_drop_macro_average_precision — URBAN-SED / ast_occlusion_drop / average_precision: **0.330743**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_occlusion_drop_macro_pointing_game — URBAN-SED / ast_occlusion_drop / pointing_game: **0.785301**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_window_logit_macro_auroc — URBAN-SED / ast_window_logit / auroc: **0.742485**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_window_logit_macro_average_precision — URBAN-SED / ast_window_logit / average_precision: **0.53891**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_ast_window_logit_macro_pointing_game — URBAN-SED / ast_window_logit / pointing_game: **0.779502**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_positive_contribution_macro_auroc — URBAN-SED / cbm_positive_contribution / auroc: **0.689716**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_positive_contribution_macro_average_precision — URBAN-SED / cbm_positive_contribution / average_precision: **0.488235**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_positive_contribution_macro_pointing_game — URBAN-SED / cbm_positive_contribution / pointing_game: **0.734372**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_silence_baseline_macro_auroc — URBAN-SED / cbm_silence_baseline / auroc: **0.771919**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_silence_baseline_macro_average_precision — URBAN-SED / cbm_silence_baseline / average_precision: **0.532675**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.25_cbm_silence_baseline_macro_pointing_game — URBAN-SED / cbm_silence_baseline / pointing_game: **0.802759**; uncertainty {'soundscapes': 2000, 'windows': 80000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_attention_rollout_macro_auroc — URBAN-SED / ast_attention_rollout / auroc: **0.519675**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_attention_rollout_macro_average_precision — URBAN-SED / ast_attention_rollout / average_precision: **0.133528**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_attention_rollout_macro_pointing_game — URBAN-SED / ast_attention_rollout / pointing_game: **0.272747**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_occlusion_drop_macro_auroc — URBAN-SED / ast_occlusion_drop / auroc: **0.597825**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_occlusion_drop_macro_average_precision — URBAN-SED / ast_occlusion_drop / average_precision: **0.325976**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_occlusion_drop_macro_pointing_game — URBAN-SED / ast_occlusion_drop / pointing_game: **0.783333**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_window_logit_macro_auroc — URBAN-SED / ast_window_logit / auroc: **0.744198**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_window_logit_macro_average_precision — URBAN-SED / ast_window_logit / average_precision: **0.545042**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_ast_window_logit_macro_pointing_game — URBAN-SED / ast_window_logit / pointing_game: **0.781506**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_positive_contribution_macro_auroc — URBAN-SED / cbm_positive_contribution / auroc: **0.690931**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_positive_contribution_macro_average_precision — URBAN-SED / cbm_positive_contribution / average_precision: **0.493034**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_positive_contribution_macro_pointing_game — URBAN-SED / cbm_positive_contribution / pointing_game: **0.738441**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_silence_baseline_macro_auroc — URBAN-SED / cbm_silence_baseline / auroc: **0.772519**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_silence_baseline_macro_average_precision — URBAN-SED / cbm_silence_baseline / average_precision: **0.534363**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json
- t6_0.5_cbm_silence_baseline_macro_pointing_game — URBAN-SED / cbm_silence_baseline / pointing_game: **0.795019**; uncertainty {'soundscapes': 2000, 'windows': 40000}; source results/revision/t6_urbansed/analysis/task6_final_metrics.json

## Interpretation rule

Accuracy alone is not interpretability evidence. Random controls remain mandatory limitations. Task 5 supports partial semantic alignment only; Task 6 supports temporal information, not causal concept localization or disentanglement.

