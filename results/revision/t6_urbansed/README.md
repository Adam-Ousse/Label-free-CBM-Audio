# Task 6 analysis launcher

The GPU worker must first write one NPZ per hop and scorer with `scores` shaped
`[number_of_prepared_windows, number_of_event_classes]`. The analysis below
rebuilds labels from the prepared onset/offset spans, so no threshold is fitted
on the test set.

For one scorer and each supported hop:

```bash
/venv/main/bin/python -m scripts.evaluation.task6_urbansed_analysis \
  --event-windows results/revision/t6_urbansed/preparation/event_windows_1s_hop_0.5.jsonl \
  --hop 0.5 \
  --scores results/revision/t6_urbansed/inference/scores_1s_hop_0.5_<SCORER>.npz \
  --output-dir results/revision/t6_urbansed/analysis/1s_hop_0.5/<SCORER> \
  --seed 42 --bootstrap-resamples 2000

/venv/main/bin/python -m scripts.evaluation.task6_urbansed_analysis \
  --event-windows results/revision/t6_urbansed/preparation/event_windows_1s_hop_0.25.jsonl \
  --hop 0.25 \
  --scores results/revision/t6_urbansed/inference/scores_1s_hop_0.25_<SCORER>.npz \
  --output-dir results/revision/t6_urbansed/analysis/1s_hop_0.25/<SCORER> \
  --seed 42 --bootstrap-resamples 2000
```

`<SCORER>` is one of `cbm_positive_contribution`,
`cbm_silence_baseline`, `ast_window_logit`, `ast_occlusion_drop`, or
`ast_attention_rollout`. No GPU inference is performed by this CPU analysis
command.
