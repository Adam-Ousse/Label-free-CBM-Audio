# Audio LF-CBM Explorer (GitHub Pages)

This folder is a fully static, backend-free demo of the **LF + broad** concept bottleneck model on the held-out ESC-50 fold-1 test split.

## Interactive views

- Filter curated test examples by true class and by correct/incorrect prediction.
- Listen to each source clip and inspect the full-clip LF + broad prediction.
- Click a displayed concept bar to set its signal from `0x` to `2x`; all 50 class logits and probabilities are recomputed in the browser from the saved final-layer effects.
- Reset interventions independently for each example.
- Switch to the segmented tab for the two views used by `plot_temporal_concept_explanations.py`:
  - aggregate top-5 concept contributions through time;
  - ranked per-window concept heatmap.

The intervention holds undisplayed concepts fixed. It is an exact last-layer counterfactual for the displayed concepts, not a new AST/CBL forward pass.

## Regenerate assets

From the repository root, using an environment with PyTorch:

```bash
python scripts/build_esc50_showcase_assets.py \
  --samples-per-class 2 \
  --max-concepts 5
```

By default, the builder reads:

- `results/audio_concept_ablation/cbm/esc50/lf_broad/run_summary.json`
- the corresponding LF + broad checkpoint;
- `results/audio_concept_ablation/segmented/esc50/lf_broad/test_temporal_concepts.pt`;
- `data/esc50/manifests/fold1_test.jsonl`.

It selects a confident correct prediction and, when available, the most confident error for each ground-truth class. It rewrites:

- `assets/audio/*.wav`
- `assets/data/esc50_showcase.json`
- `assets/data/esc50_showcase.js`

The JavaScript copy allows the page to work when opened directly with `file://`; the JSON copy is the normal HTTP fallback.

## Preview and publish

From the repository root:

```bash
python -m http.server 8000
```

Open `http://localhost:8000/docs/`. For GitHub Pages, select the `main` branch and `/docs` folder in the repository Pages settings. `.nojekyll` is already included.
