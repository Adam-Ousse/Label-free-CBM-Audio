# Label-free-CBM-Audio

Audio adaptation of Label-Free Concept Bottleneck Models (LF-CBM), with AST backbones, dataset preparation utilities, and CBM training/evaluation scripts.

## Course Context

This project was carried out for the Multimodal Explainable AI (XAI) lecture G. FRANCHI, M. FONTAINE, M. LABEAU, M. CORD:

- https://www.master-mva.com/cours/multimodal-explainable-ai-xai/

Team:

- Adam GASSEM
- Amine MAAZIZI

## Citation

If you use this project, please cite the original Label-Free Concept Bottleneck Models paper:

```bibtex
@misc{oikarinen2023labelfreeconceptbottleneckmodels,
  title={Label-Free Concept Bottleneck Models},
  author={Tuomas Oikarinen and Subhro Das and Lam M. Nguyen and Tsui-Wei Weng},
  year={2023},
  eprint={2304.06129},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2304.06129},
}
```

## Datasets

- ESC-50
- UrbanSound8K
- CREMA-D
- AudioSet (optional helper scripts)

## Main Results (AST vs CBM)

| Dataset | AST Acc. (%) | AST Loss | CBM Acc. (%) | CBM Loss | Delta Acc. (pts) |
| --- | ---: | ---: | ---: | ---: | ---: |
| ESC-50 | 93.50 | 0.2708 | 94.00 | 0.2819 | +0.50 |
| UrbanSound8K | 89.01 | 0.5032 | 87.57 | 0.3806 | -1.43 |
| CREMA-D | 70.05 | 1.7674 | 67.36 | 0.9616 | -2.69 |

## Quick Setup

From repository root:

```bash
python -m venv ../dl_env
source ../dl_env/bin/activate
pip install -r requirements.txt
```

## ESC-50 Showcase (GitHub Pages)

Static ESC-50 demo lives in docs and includes:

- class picker (emoji/icon)
- exactly 2 examples per class
- audio playback
- top concept contribution bars per prediction

Generate assets:

```bash
python scripts/build_esc50_showcase_assets.py --samples-per-class 2 --max-concepts 10
```

Local preview:

```bash
python -m http.server 8000
```

Then open:

- http://localhost:8000/docs/

## End-to-End Training Commands

### UrbanSound8K

```bash
python data/download_urbansound8k.py --output_dir data/urbansound8k/raw
python data/prepare_urbansound8k.py --urbansound8k_root data/urbansound8k/raw --out_root data/urbansound8k --repo_root .
python train_urbansound8k_ast.py --base_model_id MIT/ast-finetuned-audioset-10-10-0.4593 --epochs 30 --batch_size 32 --device cuda
python evaluate_urbansound8k_ast.py --model_id saved_models/ast_urbansound8k/fold10/best_model --split fold10_test
python train_cbm_urbansound8k_ast.py --device cuda
```

### CREMA-D

```bash
python data/download_cremad.py --output_dir data/cremad/raw
python data/prepare_cremad.py --cremad_root data/cremad/raw --out_root data/cremad --repo_root . --val_fraction 0.1 --split_seed 42
python train_cremad_ast.py --base_model_id MIT/ast-finetuned-audioset-10-10-0.4593 --epochs 30 --batch_size 32 --device cuda
python evaluate_cremad_ast.py --model_id saved_models/ast_cremad/test/best_model --split test
python train_cbm_cremad_ast.py --device cuda
```

### ESC-50

```bash
python data/download_esc50.py --output_dir data/esc50/raw
python data/prepare_esc50.py --esc50_root /path/to/ESC-50-master --out_root data/esc50 --repo_root .
python train_cbm_esc50.py --device cuda
```

## Core Entry Points

- train_urbansound8k_ast.py
- evaluate_urbansound8k_ast.py
- train_cbm_urbansound8k_ast.py
- train_cremad_ast.py
- evaluate_cremad_ast.py
- train_cbm_cremad_ast.py
- train_cbm_esc50.py

## Unified Audio API

Implemented in data_utils.py:

- get_dataset_classes(dataset_name)
- get_audio_dataset(dataset_name, split, ...)
- get_audio_dataloader(dataset_name, split, ...)
- get_audio_label_mappings(dataset_name)

Sample format:

```python
{
    "id": str,
    "audio": Tensor,
    "sr": int,
    "target": Tensor | int,
    "path": str,
    "dataset": str,
}
```

## Troubleshooting

- Kaggle 401/403 on UrbanSound8K:
  - set ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
- CREMA-D auth/rate-limit issues:
  - run huggingface-cli login or pass --token
- Quick manifest check:

```bash
python check_audio_dataloader.py --audioset_split balanced
```