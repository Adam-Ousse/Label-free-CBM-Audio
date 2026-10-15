# Task 7 CREMA-D speaker-disjoint evaluation

Protocol: deterministic actor-disjoint split, seed 42, with 64 train, 9 validation, and 18 test actors.
No demographic metadata was available; gender balancing was not claimed or applied.

| Split | Actors | Clips | Class counts |
|---|---:|---:|---|
| train | 64 | 5229 | anger=795, disgust=480, fear=523, happy=297, neutral=2885, sad=249 |
| val | 9 | 738 | anger=122, disgust=64, fear=99, happy=42, neutral=380, sad=31 |
| test | 18 | 1475 | anger=236, disgust=140, fear=171, happy=92, neutral=746, sad=90 |

Actor intersections are all empty; all 91 actors and all 7,442 clips are assigned exactly once.

## Test results

| Model | Accuracy | Macro-F1 | Test clips | Test speakers |
|---|---:|---:|---:|---:|
| AST | 64.41% | 49.44% | 1475 | 18 |
| α-CBM targeted full | 64.41% | 49.30% | 1475 | 18 |

Checkpoint selection and CBM projection/head selection used train/validation only; the speaker-disjoint test split was evaluated after selection.

The exact commands, checkpoint/vocabulary provenance, environment, and caveats are recorded in `run_config.json`, `ast/`, `features/metadata.json`, and `cbm/`.

## Revision status

- Task 7.1 speaker-disjoint evaluation — **DONE**.
- Task 7.2 temporal figures — **DONE** after inference-only scoring with the
  speaker-disjoint targeted-full CBM. The 12-example manifest and PDF/PNG
  figures are in `figures/`.

For Task 7.2, the test set was scored with the fixed checkpoint in
`ast/test/best_model` and the fixed head in `cbm/full.pt`. The Task-1 selected
CREMA-D rule was applied: 1-second windows with 0.5-second hop were masked when
they were padding-only or had RMS below -50 dBFS, with the highest-RMS fallback
if all windows were masked; the CBM used mean pooling over the effective
retained windows. For each true emotion, the selected correct example is the
highest-confidence correct prediction, and the selected error is the
highest-confidence prediction among examples whose true class is that emotion
but whose predicted class differs. Ties are resolved by manifest order.

The figure generator is `scripts/visualization/plot_task7_cremad_temporal.py`;
`figures/manifest.csv` and `figures/manifest.json` record the audio, actor,
classes, confidence, checkpoint, masking rule, and output paths. Gray shaded
windows are Task-1 masked windows.
