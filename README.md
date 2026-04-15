# Label-free-CBM-Audio (Milestone 1)

This fork is currently focused on one concrete milestone:

**Set up ESC-50 and AudioSet so a teammate can immediately start training an audio backbone.**

This is a data-layer milestone. The full LF-CBM audio training adaptation is intentionally **not** implemented yet.

Current multimodal concept scoring now uses **CLAP audio-text embeddings** (replacing CLIP image-text scoring in the active path).

## What is implemented now

- Reproducible ESC-50 preparation script
- Hugging Face AudioSet loading via `datasets.load_dataset`
- JSONL manifest generation for ESC-50
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
- `data/download_audioset.py`: inspect/cache Hugging Face AudioSet split metadata
- `clap/`: CLAP package (core loading + batched audio/text encoding + similarity helper)
- `clap_utils.py`: compatibility shim that re-exports from `clap/`
- `data/esc50/`: ESC-50 mappings + manifests output directory
- `data/audioset/`: AudioSet mappings + summary output directory
- `data/esc50_classes.txt`: ESC-50 class names
- `data/audioset_classes.txt`: generated from AudioSet class labels CSV
- `data_utils.py`: unified dataset + dataloader entry point for audio datasets
- `check_audio_dataloader.py`: quick operational sanity check
- `archive_legacy/`: legacy vision runtime code moved out of active audio path
- `archive/original_paper/`: archived image-paper-specific artifacts

## Install

Use Python 3.9+.

```bash
pip install -r requirements.txt
```

AudioSet now uses Hugging Face Datasets directly (no yt-dlp/ffmpeg dependency for AudioSet ingestion).

## Download and prepare

The pipeline is intentionally split into two stages:

1. prepare or validate ESC-50 local manifests
2. run the dataloader sanity check (ESC-50 + Hugging Face AudioSet)

## ESC-50 setup

1. Download ESC-50 with the helper script or validate a manually downloaded archive.

```bash
python data/download_esc50.py --output_dir data/esc50/raw
```

If you already downloaded the archive yourself, you can validate it instead:

```bash
python data/download_esc50.py --output_dir /path/to/esc50_tree --validate_only
```

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

## AudioSet setup (Hugging Face)

AudioSet is now loaded directly from:

- `agkphysics/AudioSet`

Supported split names in this repo are:

- `balanced` (or alias `balanced_train`)
- `unbalanced`
- `eval`
- `train` (forwarded to the HF dataset split as-is)

Optional inspection/caching helper:

```bash
python data/download_audioset.py --split balanced --max_items 64
```

Streaming mode for very large splits:

```bash
python data/download_audioset.py --split unbalanced --streaming --max_items 128
```

## Downloaded audio validation

If you want a quick smoke test of a downloaded clip directory, run:

```bash
python data/check_downloaded_audio.py --audio_dir data/audioset/clips
```

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

### AudioSet sample (from Hugging Face)

```json
{
  "audio": {"array": "waveform", "sampling_rate": 48000},
  "labels": [23, 119],
  "human_labels": ["Dog", "Bark"],
  "video_id": "YOUTUBEID"
}
```

## Unified data API (`data_utils.py`)

- `get_dataset_classes(dataset_name)`
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
- AudioSet clip duration: 10.0s (pad/truncate, loaded from HF audio arrays)

## Sanity check

After generating ESC-50 manifests, run:

```bash
python check_audio_dataloader.py --audioset_split balanced
```

This script:
- loads ESC-50 and AudioSet datasets
- prints dataset sizes and class counts
- loads one batch from each and prints tensor shapes
- fails loudly if ESC-50 manifests are invalid or HF AudioSet cannot be loaded

## Notes for teammates

This repository is now prepared for **audio backbone training data plumbing**.
Method-level LF-CBM audio adaptation should build on top of these manifests and dataloaders in a later milestone.
