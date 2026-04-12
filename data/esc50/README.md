# ESC-50 Metadata Artifacts

This folder is generated/maintained for ESC-50 dataset preparation.

Expected files:
- label_to_idx.json
- idx_to_label.json
- manifests/all.jsonl
- manifests/fold{1..5}_{train,val,test}.jsonl
- manifests/train.jsonl, manifests/val.jsonl, manifests/test.jsonl (default split)

Generate/update files with:

python data/prepare_esc50.py --esc50_root /path/to/ESC-50-master
