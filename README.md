# Label-free-CBM-Audio

Audio LF-CBM codebase with dataset preparation utilities, AST fine-tuning scripts, and CBM training entrypoints.

Current datasets wired in the active path:

- ESC-50
- UrbanSound8K
- CREMA-D
- AudioSet (Hugging Face)

## Quick start (UrbanSound8K end-to-end)

Run from repo root.

1) create or activate environment

```bash
python -m venv ../dl_env
source ../dl_env/bin/activate
pip install -r requirements.txt
```

2) download or validate UrbanSound8K

```bash
python data/download_urbansound8k.py --output_dir data/urbansound8k/raw
```

If already downloaded:

```bash
python data/download_urbansound8k.py --output_dir data/urbansound8k/raw --validate_only
```

3) build manifests and label mappings

```bash
python data/prepare_urbansound8k.py \
  --urbansound8k_root data/urbansound8k/raw \
  --out_root data/urbansound8k \
  --repo_root .
```

4) train AST backbone on UrbanSound8K (fixed folds)

```bash
python train_urbansound8k_ast.py \
  --base_model_id MIT/ast-finetuned-audioset-10-10-0.4593 \
  --epochs 30 \
  --batch_size 32 \
  --device cuda
```

5) evaluate trained AST checkpoint

```bash
python evaluate_urbansound8k_ast.py \
  --model_id saved_models/ast_urbansound8k/fold10/best_model \
  --split fold10_test
```

6) train CBM on UrbanSound8K

```bash
python train_cbm_urbansound8k_ast.py --device cuda
```

## Quick start (CREMA-D end-to-end)

Run from repo root.

1) download or validate CREMA-D from Hugging Face datasets

```bash
python data/download_cremad.py --output_dir data/cremad/raw
```

If already downloaded:

```bash
python data/download_cremad.py --output_dir data/cremad/raw --validate_only
```

2) build manifests and label mappings

```bash
python data/prepare_cremad.py \
  --cremad_root data/cremad/raw \
  --out_root data/cremad \
  --repo_root . \
  --val_fraction 0.1 \
  --split_seed 42
```

3) train AST backbone on CREMA-D (class-weighted training loss)

```bash
python train_cremad_ast.py \
  --base_model_id MIT/ast-finetuned-audioset-10-10-0.4593 \
  --epochs 30 \
  --batch_size 32 \
  --device cuda
```

4) evaluate trained AST checkpoint

```bash
python evaluate_cremad_ast.py \
  --model_id saved_models/ast_cremad/test/best_model \
  --split test
```

5) train CBM on CREMA-D

```bash
python train_cbm_cremad_ast.py --device cuda
```

## Dataset details

### UrbanSound8K

Source:

- https://www.kaggle.com/datasets/chrisfilo/urbansound8k

Supported extracted layouts (both are accepted):

- nested layout:

```text
UrbanSound8K/
  metadata/UrbanSound8K.csv
  audio/fold1..fold10/*.wav
```

- kaggle flat layout:

```text
raw/
  UrbanSound8K.csv
  fold1..fold10/*.wav
```

Fold protocol used by UrbanSound8K scripts in this repo:

- train: folds 1-8 -> `fold10_train.jsonl`
- val: fold 9 -> `fold10_val.jsonl`
- test: fold 10 -> `fold10_test.jsonl`

Generated artifacts:

- `data/urbansound8k/manifests/all.jsonl`
- `data/urbansound8k/manifests/fold10_train.jsonl`
- `data/urbansound8k/manifests/fold10_val.jsonl`
- `data/urbansound8k/manifests/fold10_test.jsonl`
- `data/urbansound8k/manifests/train.jsonl`
- `data/urbansound8k/manifests/val.jsonl`
- `data/urbansound8k/manifests/test.jsonl`
- `data/urbansound8k/label_to_idx.json`
- `data/urbansound8k/idx_to_label.json`
- `data/urbansound8k_classes.txt`

Notes:

- urban WAV files may include ADPCM / WAVE_FORMAT_EXTENSIBLE.
- repo loading and metadata readers now use fallback decoding (`wave` -> `soundfile` -> `scipy`) to handle these files.

### ESC-50

Prepare ESC-50:

```bash
python data/download_esc50.py --output_dir data/esc50/raw
python data/prepare_esc50.py --esc50_root /path/to/ESC-50-master --out_root data/esc50 --repo_root .
```

### AudioSet (Hugging Face)

Optional metadata/cache helper:

```bash
python data/download_audioset.py --split balanced --max_items 64
```

Streaming example:

```bash
python data/download_audioset.py --split unbalanced --streaming --max_items 128
```

### CREMA-D

Source:

- https://huggingface.co/datasets/MahiA/CREMA-D

Split protocol used in this repo:

- official `train.csv` is split into `train` and `val` with stratified 10% validation
- official `test.csv` is used as `test`

Generated artifacts:

- `data/cremad/manifests/all.jsonl`
- `data/cremad/manifests/train.jsonl`
- `data/cremad/manifests/val.jsonl`
- `data/cremad/manifests/test.jsonl`
- `data/cremad/label_to_idx.json`
- `data/cremad/idx_to_label.json`
- `data/cremad/summary.json`
- `data/cremad_classes.txt`

Notes:

- CREMA-D is imbalanced across emotion classes.
- `train_cremad_ast.py` uses class-weighted cross-entropy by default.

## Training entrypoints

- `train_urbansound8k_ast.py`: fine-tune AST on UrbanSound8K (fixed fold protocol)
- `evaluate_urbansound8k_ast.py`: evaluate AST classifier checkpoint
- `train_cbm_urbansound8k_ast.py`: run CBM training with UrbanSound8K manifests
- `train_cremad_ast.py`: fine-tune AST on CREMA-D (train/val/test with class-weighted loss)
- `evaluate_cremad_ast.py`: evaluate CREMA-D AST classifier checkpoint
- `train_cbm_cremad_ast.py`: run CBM training with CREMA-D manifests
- `train_cbm_esc50.py`: run CBM training with ESC-50

## Unified audio API

Core API lives in `data_utils.py`:

- `get_dataset_classes(dataset_name)`
- `get_audio_dataset(dataset_name, split, ...)`
- `get_audio_dataloader(dataset_name, split, ...)`
- `get_audio_label_mappings(dataset_name)`

Returned sample format:

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

## Troubleshooting

- UrbanSound8K download fails with 401/403:
  - configure Kaggle credentials (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`)
- UrbanSound8K extraction validates but training fails on audio decode:
  - ensure dependencies are up to date: `pip install -r requirements.txt`
- CREMA-D download fails due auth/rate limits:
  - run `huggingface-cli login` or pass `--token` to `data/download_cremad.py`
- Quick check that manifests load:

```bash
python check_audio_dataloader.py --audioset_split balanced
```