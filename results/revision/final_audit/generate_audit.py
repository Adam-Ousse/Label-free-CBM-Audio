from __future__ import annotations
import csv, json, subprocess
from pathlib import Path
ROOT=Path("/workspace/Label-free-CBM-Audio")
OUT=ROOT/"results/revision/final_audit"
def load(rel): return json.loads((ROOT/rel).read_text())
rows=[]
def add(rid,exp,ds,var,split,metric,value,unc,n,src,prov,status,caveat,role):
    rows.append(dict(result_id=rid,experiment=exp,dataset=ds,variant=var,split_fold_seed=split,
        metric=metric,value=value,uncertainty=unc,n=n,canonical_source_file=src,
        checkpoint_config_provenance=prov,status=status,caveat_disclosure=caveat,
        suggested_manuscript_role=role))
def n_for(ds): return {"ESC-50":400,"esc50":400,"UrbanSound8K":837,"urbansound8k":837,"CREMA-D":1489,"cremad":1489}.get(ds)

t0=load("results/revision/t0_baseline/global/global_metrics.json")
if isinstance(t0,dict): t0=t0.get("rows",t0.get("metrics",[]))
for i,r in enumerate(t0):
    ds=r.get("dataset"); var=r.get("variant",r.get("model","unknown")); split=r.get("split",r.get("split_name","historical single split"))
    for metric in ("accuracy","macro_f1"):
        if metric in r:
            add(f"t0_{i}_{metric}","T0 original single-split classification",ds,var,split,metric,r[metric],None,r.get("num_samples"),
                "results/revision/t0_baseline/global/global_metrics.json","release/google_drive_bundle/checkpoints; T0 global metrics",
                "HISTORICAL_ONLY","Context only; official CV or speaker-disjoint results supersede it as primary where available.","context/appendix")

t0c=load("results/revision/t0_baseline/comparison/comparison.json")
add("t0_exact_reproduction_gate","T0 exact reproduction / baseline verification","ESC-50 + UrbanSound8K + CREMA-D","release checkpoints",
    "persisted T0 comparison protocol","comparison_gate_pass",bool(t0c.get("pass")),None,len(t0c.get("checks",t0c.get("rows",[]))),
    "results/revision/t0_baseline/comparison/comparison.json","release/google_drive_bundle/checkpoints","FINAL",
    "The comparison artifact warns that some segmented CREMA paths are superseded; the release-bundle global rows are canonical.","methods/QC")

t1=load("results/revision/t1_silence/baseline_preservation.json")
add("t1_baseline_preservation","T1 silence/padding audit","ESC-50 + UrbanSound8K + CREMA-D","all audited variants",
    "persisted Task-1 protocol","all_checks_pass",bool(t1.get("pass")),None,len(t1.get("checks",[])),
    "results/revision/t1_silence/baseline_preservation.json","absolute tolerance=1e-6","FINAL",
    "Audit result, not a new accuracy claim.","methods/QC")

t2=load("results/revision/t2_leakage/retrained/comparison.json")
for i,r in enumerate(t2.get("rows",[])):
    for metric in ("clean_accuracy","clean_macro_f1","accuracy_delta","macro_f1_delta"):
        if metric in r:
            add(f"t2_{i}_{metric}","T2 leakage-cleaned concept experiment",r["dataset"],r["variant"],
                "seed=42; persisted test split",metric,r[metric],None,n_for(r["dataset"]),
                "results/revision/t2_leakage/retrained/comparison.json",
                "results/revision/t2_leakage/retrained/run_metadata.json; checkpoint_validation.json","FINAL",
                "CREMA-D clean/original comparison has the persisted provenance-confounded warning.","robustness/leakage")

t3=load("results/revision/t3_baselines/t3_gpu_01_fix/metrics.json")
if isinstance(t3,dict): t3=t3.get("rows",[])
for i,r in enumerate(t3):
    if r.get("method") not in {"clap_zero_shot","clap_linear_probe","random_direction","random_word"} or r.get("capacity")!="full": continue
    for metric in ("accuracy","macro_f1"):
        add(f"t3_{i}_{metric}","T3 strong controls",r["dataset"],r.get("variant",""),
            f"persisted test split; random_seed={r.get('random_seed')}",metric,r.get(metric),None,n_for(r["dataset"]),
            "results/revision/t3_baselines/t3_gpu_01_fix/metrics.json",r.get("result_path","T3 metrics manifest"),"FINAL",
            "Random-word and random-direction controls are required limitations; accuracy is not interpretability evidence.","control/appendix")

t4=load("results/revision/t4_cv/principal_seed_summary.json")
for r in t4.get("summary",[]):
    for base,std_key in [("accuracy","accuracy_std_sample"),("macro_f1","macro_f1_std_sample")]:
        add(f"t4_seed_{r['dataset']}_{base}","T4 five-seed robustness",r["dataset"],r["variant"],
            f"seeds={r['seeds']}",base,r[f"{base}_mean"],
            {"std_sample":r[std_key],"min":r[f"{base}_min"],"max":r[f"{base}_max"]},r["n"],
            "results/revision/t4_cv/principal_seed_summary.json",f"git_commit={t4.get('git_commit')}; fixed principal split","FINAL",
            "Fixed-split seed robustness, not official CV.","robustness")

for rel,label in [("results/revision/overnight_production_20261015/esc50_official_cv/summary.json","ESC-50"),
                  ("results/revision/overnight_production_20261015/urban_official_cv/summary.json","UrbanSound8K")]:
    x=load(rel)
    for key,metric,var in [("ast_accuracy","accuracy","AST"),("ast_macro_f1","macro_f1","AST"),
                           ("lf_broad_accuracy","accuracy","lf_broad"),("lf_broad_macro_f1","macro_f1","lf_broad")]:
        a=x["aggregate"][key]
        add(f"t4_official_{label}_{var}_{metric}","T4 official cross-validation",label,var,x["protocol"],metric,a["mean"],
            {"sd":a["sd"],"min":a["min"],"max":a["max"],"folds":len(x["folds"])},
            sum(f.get("test_examples",0) for f in x["folds"]) or (8732 if label == "UrbanSound8K" else "all fold test sets"),rel,
            "official fold summaries and fold-complete artifacts","FINAL",
            "UrbanSound8K fold 7 AST training interrupted after epoch 24/30; best epochs 1-24 checkpoint retained.","main classification")

t5=load("results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json")
for s in t5["dataset_summaries"]:
    add(f"t5_{s['dataset']}_expected_sign_rate","Task 5 descriptor/intervention alignment",s["dataset"],
        "all four established variants","test; seed=42; bootstrap=2000","expected_sign_success_rate",
        s["expected_sign_success_rate"],{"success":s["expected_sign_success_count"],"tests":s["mapped_test_count"]},s["clip_count"],
        "results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json",
        "results/revision/t5_descriptors/controlled_edits/task5_metadata.json","FINAL",
        "Heterogeneous/modest semantic alignment; finite effects verified; not causal/disentanglement evidence.","Task 5 summary")
for i,r in enumerate(t5["descriptor_associations"]):
    add(f"t5_assoc_{i}","Task 5 descriptor correlations",r["dataset"],"|".join(r["models"]),
        "test; seed=42; bootstrap=2000","spearman_rho",r["spearman_rho"],
        {"ci_low":r["ci_low"],"ci_high":r["ci_high"],"expected_sign":r["expected_sign"],"expected_sign_match":r["expected_sign_match"]},
        r["n"],"results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json",
        "concept_descriptor_map.json; task5_metadata.json","FINAL",
        "Prespecified persisted mapped test; undefined descriptor values remain missing.","Task 5 detailed")
for i,r in enumerate(t5["edit_summaries"]):
    add(f"t5_edit_{i}","Task 5 controlled-edit matched-vs-null association",r["dataset"],r.get("variant",""),
        f"edit={r['edit']}; seed=42; bootstrap=2000","matched_minus_unmatched_rho",r["matched_minus_unmatched"],
        {"matched_mean_rho":r["matched_mean_rho"],"matched_ci":r["matched_ci"],
         "unmatched_mean_rho":r["unmatched_mean_rho"],"matched_minus_unmatched_ci":r["matched_minus_unmatched_ci"]},
        r["matched_pair_count"],"results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json",
        "task5_metadata.json; fixed descriptor map","FINAL",
        "Heterogeneous across edits and variants; null draws are persisted.","Task 5 detailed")
add("t5_null_pairings","Task 5 descriptor null analysis","ESC-50 + UrbanSound8K + CREMA-D","all mapped models",
    "seed=42","random_pairing_count",len(t5["descriptor_null_pairings"]),None,None,
    "results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json",
    "task5_metadata.json; null draws=1000/model","FINAL","Existing 12,000 random pairings; no new null analysis.","Task 5 summary")

# T7 CREMA-D speaker-disjoint evaluation.
t7=load("results/revision/t7_cremad/speaker_disjoint/summary.json")
for method,key,var in [("AST","ast","AST"),("alpha-CBM targeted full","cbm_targeted_full","targeted_full")]:
    z=t7[key]
    for metric in ("accuracy","macro_f1","balanced_accuracy"):
        if metric in z:
            add(f"t7_{key}_{metric}","T7 CREMA-D speaker-disjoint evaluation","CREMA-D",var,
                "actor-disjoint; seed=42; validation-only selection",metric,z[metric],
                {"train_actors":64,"val_actors":9,"test_actors":18,"actor_intersections":0},z.get("num_samples",1475),
                "results/revision/t7_cremad/speaker_disjoint/summary.json",
                "speaker_disjoint/manifest_verification.json; run_config.json","FINAL",
                "Limited one-seed actor-quota evaluation; no gender/demographic metadata; do not overgeneralize.","speaker robustness")

t6=load("results/revision/t6_urbansed/analysis/task6_final_metrics.json")
for hop,hv in t6["by_hop"].items():
    for method,mv in hv["methods"].items():
        for metric in ("auroc","average_precision","pointing_game"):
            add(f"t6_{hop}_{method}_macro_{metric}","Task 6 URBAN-SED temporal localization","URBAN-SED",method,
                f"hop={hop}s; soundscape bootstrap seed={t6['bootstrap']['seed']}",metric,mv["macro"][metric],
                {"soundscapes":hv["n_soundscapes"],"windows":hv["n_windows"]},hv["n_soundscapes"],
                "results/revision/t6_urbansed/analysis/task6_final_metrics.json",
                "inference/provenance_validation.json; analysis/task6_score_manifest.json","FINAL",
                "No random-mapping localization null; localization is not causal concept evidence.","Task 6 main")
        for cls,cv in mv["by_class"].items():
            for metric,ci in [("auroc","auroc_ci"),("average_precision","average_precision_ci"),("pointing_game","pointing_game_ci")]:
                add(f"t6_{hop}_{method}_{cls}_{metric}","Task 6 URBAN-SED temporal localization","URBAN-SED",method,
                    f"hop={hop}s; soundscape bootstrap seed={t6['bootstrap']['seed']}",metric,cv[metric],
                    {"ci_low":cv[ci][0],"ci_high":cv[ci][1]},cv["positive_soundscapes"],
                    "results/revision/t6_urbansed/analysis/task6_final_metrics.json",
                    "inference/provenance_validation.json; score manifest","FINAL",
                    "Class support is positive soundscape count; no random-mapping null.","Task 6 supplementary")

def git(*args): return subprocess.run(["git",*args],cwd=ROOT,text=True,capture_output=True,check=False).stdout.strip()
(OUT/"final_results_inventory.json").write_text(json.dumps({
    "audit_type":"read_only_final_deep_audit","repository":str(ROOT),"audit_commit":git("rev-parse","HEAD"),
    "row_count":len(rows),"primary_source_policy":"machine-readable JSON/CSV/manifests/logs outrank prose and stale summaries","rows":rows},indent=2)+"\n")
fields=list(rows[0])
with (OUT/"final_results_inventory.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        q=dict(r)
        for k in ("value","uncertainty"):
            if isinstance(q[k],(dict,list)): q[k]=json.dumps(q[k],sort_keys=True)
        w.writerow(q)
def fmt(v): return f"{v:.6g}" if isinstance(v,float) else str(v)
md=["# Final results inventory","",f"- Audit commit: {git('rev-parse','HEAD')}",f"- Rows: **{len(rows)}**","- Generated from persisted machine-readable artifacts; no inference was run.","","## Canonical source files",""]
srcs=["results/revision/t0_baseline/global/global_metrics.json","results/revision/t1_silence/baseline_preservation.json",
"results/revision/t2_leakage/retrained/comparison.json","results/revision/t3_baselines/t3_gpu_01_fix/metrics.json",
"results/revision/t4_cv/principal_seed_summary.json","results/revision/overnight_production_20261015/esc50_official_cv/summary.json",
"results/revision/overnight_production_20261015/urban_official_cv/summary.json",
"results/revision/t5_descriptors/controlled_edits/controlled_edit_analysis.json",
"results/revision/t5_descriptors/controlled_edits/task5_metadata.json",
"results/revision/t6_urbansed/analysis/task6_final_metrics.json","results/revision/t6_urbansed/analysis/task6_qc.json",
"results/revision/t7_cremad/speaker_disjoint/summary.json"]
md += [f"- {p}" for p in srcs]+["","## Final rows",""]
for r in rows:
    if r["status"]=="FINAL" and r["suggested_manuscript_role"] in {"main classification","Task 5 summary","Task 6 main"}:
        md.append(f"- {r['result_id']} — {r['dataset']} / {r['variant']} / {r['metric']}: **{fmt(r['value'])}**; uncertainty {fmt(r['uncertainty'])}; source {r['canonical_source_file']}")
md += ["","## Interpretation rule","","Accuracy alone is not interpretability evidence. Random controls remain mandatory limitations. Task 5 supports partial semantic alignment only; Task 6 supports temporal information, not causal concept localization or disentanglement.",""]
(OUT/"final_results_inventory.md").write_text("\n".join(md)+"\n")

(OUT/"paper_assets.md").write_text("""# Paper asset inventory

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
""")

(OUT/"instance_backup_manifest.md").write_text("""# Instance backup manifest

Read-only destruction-readiness inventory. Nothing was uploaded, moved, or deleted.

## MUST_BACK_UP_BEFORE_DESTROYING_INSTANCE

- release/, including ignored release/google_drive_bundle/ checkpoints and bundle manifest; required by T0, T2, T5, and Task 6 provenance.
- Canonical results/revision/ trees: t0_baseline, t1_silence, t2_leakage, t3_baselines/t3_gpu_01_fix, t4_cv, overnight_production_20261015, t5_descriptors, t6_urbansed, and t7_cremad.
- Ignored dataset manifests and prepared data under data/*/manifests/ and data/urbansed/.
- Ignored saved_activations/ and Task 5 descriptor caches, including results/revision/t5_descriptors_corrected/.
- Task 4 official CV fold checkpoints/features under results/revision/t4_cv/urbansound8k_official_cv_overnight_exactcache_20261015/.
- Task 7 speaker-disjoint manifests, features, checkpoint, and summary.
- Task 6 inference score NPZs, score manifest, provenance/QC JSON, tables, and figures.
- Audit reports in this directory.

## NICE_TO_BACK_UP

- results.stale/, historical interrupted/aborted logs, and forensic copies.
- Environment/version manifests, package files, and git history.
- Uncommitted final-stage scripts: scripts/evaluation/task5_controlled_edits.py, task6_urbansed_inference.py, and task6_urbansed_finalize.py.
- All qualitative figures and PDF/PNG pairs.

## SAFE_TO_LOSE

- __pycache__, .pytest_cache, zero-byte queue/launcher logs, and regenerable temporary logs, after the canonical artifacts above are secured.
- Historical prose-only claims not referenced by the canonical inventory.

## Uncommitted / unbacked-up warning

git status shows modified features/audio_descriptors.py, untracked release/, untracked results.stale/, and untracked final-stage Task 5/Task 6 scripts. Ignored canonical results/data are not represented by the git commit and must be copied independently.
""")

(OUT/"final_audit_report.md").write_text(f"""# Final deep audit report

## Scope

Read-only audit: no training, inference, GPU work, result regeneration, deletion, or upload. Audit commit: {git("rev-parse","HEAD")}. Machine-readable artifacts outrank prose and stale summaries.

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

Task 6 covers 2000 URBAN-SED test soundscapes, 9956 events, 1643 unique source clips, 1-s windows at 0.5/0.25-s hops, and five methods. At 0.5 s, alpha-CBM positive contribution is AUROC/AP/pointing 0.690931/0.493034/0.738441; alpha-CBM+silence 0.772519/0.534363/0.795019; AST logits 0.744198/0.545042/0.781506; occlusion 0.597825/0.325976/0.783333; attention rollout 0.519675/0.133528/0.272747. Silence handling helps. AST logits outperform raw alpha-CBM on macro AUROC/AP; alpha-CBM exceeds occlusion and rollout on aggregate AUROC/AP, but not occlusion on pointing game. Best alpha-CBM classes include jackhammer/car horn/drilling/street music; air conditioner is below chance. No random-mapping localization null was performed. Qualitative examples are balanced: two success, two ambiguous, two failure.

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
""")
print(f"wrote {len(rows)} inventory rows")

