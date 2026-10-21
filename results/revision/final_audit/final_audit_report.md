# Final deep audit report

## Scope

Read-only audit: no training, inference, GPU work, result regeneration, deletion, or upload. Audit commit: 27da0a9bcb9516601e14aa394b719766e24586e3. Machine-readable artifacts outrank prose and stale summaries.

## Canonical classification results

- T0 single-split results are HISTORICAL_ONLY; they must not replace official CV or speaker-disjoint primary estimates.
- Official ESC-50 5-fold: alpha-CBM LF-broad accuracy 0.9175 (SD 0.02756), macro-F1 0.915574 (SD 0.02947); AST accuracy 0.9135 (SD 0.02184), macro-F1 0.912013 (SD 0.02242).
- Official UrbanSound8K 10-fold: alpha-CBM LF-broad accuracy 0.848903 (SD 0.05129), macro-F1 0.860045 (SD 0.04648); AST accuracy 0.839795 (SD 0.05062), macro-F1 0.851536 (SD 0.04828).
- Five-seed fixed-split robustness: ESC-50 LF-broad accuracy 0.9425 (SD 0.00354); UrbanSound8K LF-broad 0.901075 (SD 0.00530); CREMA-D targeted-full 0.687710 (SD 0.00614).

## Robustness / leakage / controls

T1 preservation passed at absolute tolerance 1e-6. T2 leakage-cleaned comparisons are complete, with the persisted CREMA-D provenance-confounded warning. T3 includes CLAP zero-shot/linear, random-word, and random-direction controls. Random-direction accuracy remains strong: ESC-50 0.955, UrbanSound8K 0.898447, CREMA-D 0.693083. Accuracy cannot therefore establish interpretability.

## CREMA-D speaker-disjoint result

The persisted actor-disjoint split is 64 train / 9 validation / 18 test actors, 5229 / 738 / 1475 clips, seed 42, validation-only selection, and zero actor intersection. AST and targeted-full alpha-CBM both have accuracy 0.644068; macro-F1 is 0.494383 and 0.492965. The source historical manifests overlap actors by design, but the proposed split removes overlap. Demographic/gender metadata is unavailable, so no gender balance claim is valid.

## Task 5

Task 5 covers 400 ESC-50, 837 UrbanSound8K, and 1489 CREMA-D clips, four variants, six interventions, finite effects, 21 mapped descriptor tests, 2000 bootstrap resamples, and 12000 random descriptor pairings. Expected-sign success is ESC-50 57/108 (52.8%), UrbanSound8K 34/72 (47.2%), CREMA-D 90/168 (53.6%). Some correlations are clear and correctly signed (CREMA breathiness/HNR -0.547; low loudness/RMS -0.748; low pitch/F0 -0.558), but others are weak or reversed. Controlled edits are heterogeneous and do not establish causality or disentanglement. UrbanSound8K descriptor windows (1553) differ from fixed AST/CBM model windows (5859) because descriptor caching uses natural duration with end padding while model inputs are normalized to 4 s; final pairings are clip-level and ID-aligned.

## Task 6

Task 6 covers 2000 URBAN-SED test soundscapes, 9956 events, 1643 unique source clips, 1-s windows at 0.5/0.25-s hops, and five methods. The retained subset uses source UrbanSound8K folds 9-10, reference training folds 1-8, and the provenance audit reports zero prohibited source overlap; the class order is the ten UrbanSound8K classes. At 0.5 s, alpha-CBM positive contribution is AUROC/AP/pointing 0.690931/0.493034/0.738441; alpha-CBM+silence 0.772519/0.534363/0.795019; AST logits 0.744198/0.545042/0.781506; occlusion 0.597825/0.325976/0.783333; attention rollout 0.519675/0.133528/0.272747. Silence handling helps. AST logits outperform raw alpha-CBM on macro AUROC/AP; alpha-CBM exceeds occlusion and rollout on aggregate AUROC/AP, but not occlusion on pointing game. Best alpha-CBM classes include jackhammer/car horn/drilling/street music; air conditioner is below chance. No random-mapping localization null was performed. Qualitative examples are balanced: two success, two ambiguous, two failure.

## Claim matrix

| Claim | Supported? | Safe wording / limitation |
|---|---|---|
| alpha-CBM approaches AST classification | Yes, protocol-dependent | Comparable on these evaluations, not universally superior. |
| Label-free concepts are meaningful | Partial | Some mapped concepts show semantic/temporal evidence. |
| Concept leakage is not required | Supported with caveats | T2 supports tested cleaned vocabularies only. |
| Accuracy demonstrates interpretability | No | Random controls reject this inference. |
| Random directions remain predictive | Yes | Must be disclosed. |
| Task 5 supports semantic alignment | Partial | Heterogeneous/modest; no causal/disentanglement claim. |
| Task 6 supports temporal information | Yes, limited | Not causal concept localization. |
| alpha-CBM beats AST localization | No | AST wins macro AUROC/AP. |
| alpha-CBM is causal/disentangled | No | Do not use these terms. |
| Speaker-disjoint CREMA-D generalizes broadly | No | One seed, fixed actor quotas, no demographic metadata. |
| Robustness across seeds/folds | Partial | Variability and fold-7 truncation must be disclosed. |

## Mandatory disclosures

MUST_DISCLOSE: UrbanSound8K fold-7 interruption and epoch-1-24 checkpoint salvage; random-direction/random-word controls; heterogeneous/modest Task 5 effects; no Task 6 random-mapping null; CREMA speaker-disjoint limitations and absent demographic metadata; historical single-split versus official CV distinction; undefined descriptor measurements retained as missing; URBAN-SED subset/source-fold and window protocol.

SHOULD_DISCLOSE: Task 2 CREMA provenance-confounded warning; blocked generation-variance check; unresolved concept-generator request metadata; ignored/uncommitted checkpoint/data provenance; class support counts and undefined metrics.

OPTIONAL_DETAIL: bootstrap seeds, checkpoint hashes, package versions, stale-run history, zero-byte inventory.

## DO_NOT_USE_FOR_PAPER

- results/revision/t4_cv/urbansound8k_official_cv_overnight_20261015/ — explicitly aborted/not used.
- results/revision/t4_cv/urbansound8k_official_cv/urban_queue.log — zero-byte.
- results/revision/t4_cv/t4_principal_urban/per_fold.csv and t4_principal_cremad/per_fold.csv — zero-byte/non-CV.
- results/revision/t6_urbansed/preparation_ambiguous_run/ — stale ambiguous preparation.
- results/revision/esc50_compressed_gpu1* and results.stale/revision/esc50_compressed_gpu1* — partial historical fold artifacts.
- results.stale/revision/ — forensic copies, not current paper sources.
- results/revision/t0_baseline/segmented/summary.json — historical segmented context, not official primary classification.
- results/revision/t5_descriptors_corrected/analysis.json — corrected descriptor-analysis intermediate; final mapped/intervention source is consolidated Task 5 analysis.
- All zero-byte revision files — placeholders/logs/stale manifests, not results.

## Final consistency checks

Canonical numbers trace to JSON/CSV sources in final_results_inventory.json; Task 5 and Task 6 final artifacts are included; all expected datasets, six Task 5 interventions, four variants, and five Task 6 methods are represented; Task 6 score/QC manifests report finite, aligned scores and reproducible bootstrap; no new experiment, training, inference, upload, or deletion was performed.

## Overall disposition

PASS WITH WARNINGS. The experiments are scientifically frozen and the machine-readable results are ready for manuscript writing with the mandatory disclosures above. The instance is NOT safe to destroy until untracked/ignored release, data, caches, canonical results, and uncommitted final-stage scripts are independently backed up.
