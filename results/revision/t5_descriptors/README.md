# Task 5 analysis launcher

The fixed map is `concept_descriptor_map.json`. It contains 17 exact concepts
present in the retained temporal CBM models; speech rate is intentionally absent
because it is not in the prepared descriptor cache.

CPU analysis command (the temporal bundles are already cached):

```bash
/venv/main/bin/python -m scripts.evaluation.task5_descriptor_analysis \
  --descriptor-map results/revision/t5_descriptors/concept_descriptor_map.json \
  --descriptor-root results/revision/t5_descriptors \
  --temporal-concepts \
    results/revision/t0_baseline/segmented/cremad/lf/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/cremad/lf_broad/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/cremad/lf_contrastive/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/cremad/full/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/esc50/lf/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/esc50/lf_broad/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/esc50/lf_contrastive/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/esc50/full/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/urbansound8k/lf/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/urbansound8k/lf_broad/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/urbansound8k/lf_contrastive/test_temporal_concepts.pt \
    results/revision/t0_baseline/segmented/urbansound8k/full/test_temporal_concepts.pt \
  --output results/revision/t5_descriptors/analysis.json \
  --seed 42 --bootstrap-resamples 2000 --null-draws 1000
```

The controlled-edit aggregator consumes GPU-produced change matrices and the
same map. The preregistered edit names are pitch shift -4/+4 semitones,
low-pass 2 kHz, gain -20 dB, synthetic RT60 1 s reverb, and 0.7x time stretch.
