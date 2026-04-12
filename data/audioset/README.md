# AudioSet Metadata Artifacts

This folder is generated/maintained for AudioSet dataset preparation.

Expected files:
- mid_to_idx.json
- idx_to_mid.json
- idx_to_display_name.json
- manifests/balanced_train.jsonl
- manifests/eval.jsonl

Generate/update files with:

python data/prepare_audioset.py \
  --class_labels_csv /path/to/class_labels_indices.csv \
  --balanced_csv /path/to/balanced_train_segments.csv \
  --eval_csv /path/to/eval_segments.csv \
  --clips_root /path/to/local/audioset/clips
