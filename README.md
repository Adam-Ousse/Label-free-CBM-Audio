# Label-free-CBM-Audio (Milestone 1)

This fork is currently focused on one concrete milestone:

**Set up ESC-50 and AudioSet so a teammate can immediately start training an audio backbone.**

This is a data-layer milestone. The full LF-CBM audio training adaptation is intentionally **not** implemented yet.

## What is implemented now

- Reproducible ESC-50 preparation script
- Reproducible AudioSet preparation script
- JSONL manifest generation
- Label mapping files for both datasets
- Unified PyTorch audio manifest dataset/dataloader API in `data_utils.py`
- Multi-label support for AudioSet targets
- Sanity-check script for one-batch loading
- Archive area for original image-paper reproduction assets

## What is intentionally not implemented yet

- Concept filtering / concept set generation for audio
- CLIP-to-audio encoder replacement (e.g., CLAP)
- CBM training adaptation for audio
- Sparse classifier/audio-method redesign
- Explanation plotting updates for audio models

## Repository layout for this milestone

- `data/prepare_esc50.py`: parse official ESC-50 metadata and build manifests/mappings
- `data/prepare_audioset.py`: parse AudioSet CSV metadata and build manifests/mappings
- `data/esc50/`: ESC-50 mappings + manifests output directory
- `data/audioset/`: AudioSet mappings + manifests output directory
- `data/esc50_classes.txt`: ESC-50 class names
- `data/audioset_classes.txt`: generated from AudioSet class labels CSV
- `data_utils.py`: unified dataset + dataloader entry point for audio datasets
- `check_audio_dataloader.py`: quick operational sanity check
- `archive/original_paper/`: archived image-paper-specific artifacts

## Install

Use Python 3.9+.

```bash
pip install -r requirements.txt
```

## ESC-50 setup

1. Download ESC-50 from the official source and extract it.
2. Ensure it has this structure:

```text
/path/to/ESC-50-master/
  meta/esc50.csv
  audio/*.wav
```

3. Generate manifests/mappings:

```bash
python data/prepare_esc50.py \
  --esc50_root /path/to/ESC-50-master \
  --out_root data/esc50 \
  --repo_root .
```

### ESC-50 split convention

- Official folds are preserved (`fold` field in each sample)
- Per-test-fold manifests are generated:
  - `fold{1..5}_train.jsonl`
  - `fold{1..5}_val.jsonl`
  - `fold{1..5}_test.jsonl`
- Default top-level manifests are also generated:
  - `train.jsonl`, `val.jsonl`, `test.jsonl`
- Default behavior uses `default_test_fold=1` and `val_fold=(test_fold+1) mod 5`

## AudioSet setup

1. Download AudioSet metadata CSVs:
- `class_labels_indices.csv`
- `balanced_train_segments.csv`
- `eval_segments.csv`

2. Prepare local clips (for example as WAV files) under one root directory.

3. Generate manifests/mappings:

```bash
python data/prepare_audioset.py \
  --class_labels_csv /path/to/class_labels_indices.csv \
  --balanced_csv /path/to/balanced_train_segments.csv \
  --eval_csv /path/to/eval_segments.csv \
  --clips_root /path/to/local/audioset/clips \
  --out_root data/audioset \
  --repo_root .
```

### AudioSet missing-clip behavior

- Missing local clips are skipped by default (counts reported in `data/audioset/summary.json`)
- To fail hard on missing clips, add `--fail_on_missing`

## Manifest schema

### ESC-50 sample

```json
{
  "id": "5-12345-A-10",
  "audio_path": "data/esc50/audio/5-12345-A-10.wav",
  "label": "dog",
  "label_idx": 16,
  "fold": 1,
  "sample_rate": 44100,
  "duration": 5.0,
  "dataset": "esc50"
}
```

### AudioSet sample

```json
{
  "id": "YOUTUBEID_30_40",
  "youtube_id": "YOUTUBEID",
  "start_sec": 30.0,
  "end_sec": 40.0,
  "audio_path": "data/audioset/clips/YOUTUBEID_30_40.wav",
  "labels_mid": ["/m/068hy", "/m/07q6cd_"],
  "label_idx": [23, 119],
  "sample_rate": 16000,
  "duration": 10.0,
  "dataset": "audioset"
}
```

## Unified data API (`data_utils.py`)

- `get_dataset_classes(dataset_name)`
- `get_audio_manifest_path(dataset_name, split)`
- `get_audio_dataset(dataset_name, split, ...)`
- `get_audio_dataloader(dataset_name, split, ...)`
- `get_audio_label_mappings(dataset_name)`

Returned sample dictionary format:

```python
{
    "id": str,
    "audio": Tensor,
    "sr": int,
    "target": Tensor | int,
    "path": str,
    "dataset": str
}
```

Defaults:
- mono waveform
- sample rate: 16000
- ESC-50 clip duration: 5.0s (pad/truncate)
- AudioSet clip duration: 10.0s (pad/truncate)

## Sanity check

After generating manifests, run:

```bash
python check_audio_dataloader.py
```

This script:
- loads ESC-50 and AudioSet datasets
- prints dataset sizes and class counts
- loads one batch from each and prints tensor shapes
- fails loudly if manifests or paths are invalid

## Notes for teammates

This repository is now prepared for **audio backbone training data plumbing**.
Method-level LF-CBM audio adaptation should build on top of these manifests and dataloaders in a later milestone.
