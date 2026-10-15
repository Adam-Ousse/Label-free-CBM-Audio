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
