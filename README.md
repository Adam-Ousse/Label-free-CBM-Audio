# Label-Free Concept Bottleneck Models for Audio

This repository adapts Label-Free Concept Bottleneck Models (LF-CBMs) to audio
classification. Fine-tuned Audio Spectrogram Transformer (AST) features are
projected onto CLAP-grounded textual concepts and classified by a sparse linear
head. The project includes:

- DeepSeek generation of vanilla LF, broad perceptual, and group-wise contrastive concepts;
- matched CBM source ablations on ESC-50, UrbanSound8K, and CREMA-D;
- global and overlapping 1-second segmented inference;
- a static GitHub Pages explorer with exact top-five concept interventions;
- paper sources and a checksummed external checkpoint bundle.

The project was developed by [Adam Gassem and Amine Maazizi](AUTHORS.md) for the
MVA Multimodal Explainable AI course.

## Results

All numbers use one fixed held-out split and seed. Accuracy, macro-F1, and sparse
head zeros are percentages. CREMA-D uses the latest speech-targeted concept
protocol; the other two datasets use the shared audio vocabulary.

| Dataset | Model | Accuracy | Macro-F1 | Sparsity |
| --- | --- | ---: | ---: | ---: |
| ESC-50 | Fine-tuned AST | 93.50 | 93.28 | — |
| ESC-50 | LF + broad | 93.50 | 93.39 | 92.58 |
| UrbanSound8K | Fine-tuned AST | 89.01 | 90.03 | — |
| UrbanSound8K | LF + broad | 89.84 | 90.94 | 86.04 |
| CREMA-D | Fine-tuned AST | 70.05 | 52.76 | — |
| CREMA-D | Targeted full union | 70.25 | 51.58 | 92.79 |

The full LF/broad/contrastive ablation, retained concept counts, segmentation
results, and limitations are reported in [`paper/main.tex`](paper/main.tex).

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/` | Dataset download/preparation code and versioned concept sets |
| `models/` | AST adapters used by training and inference |
| `scripts/` | Release, CREMA-D follow-up, plotting, and website utilities |
| `experiments/` | Focused hyperparameter and filtering studies |
| `notebooks/` | Reproducible concept-generation walkthrough |
| `docs/` | Backend-free GitHub Pages explorer |
| `paper/` | Full paper source and figures |
| `jdse_extended_abstract/` | Three-page extended abstract source |
| `release/google_drive_bundle/` | Generated external artifact folder (ignored by Git) |
| `archive/` | Preserved upstream/original LF-CBM material |

Root-level training and evaluation scripts remain as compatibility entry points.
New project utilities belong in `scripts/`, and focused research sweeps belong in
`experiments/`.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. `DEEPSEEK_MODEL` is optional. Do not commit
`.env`; it is ignored. Run the project tests with `pytest`.

## Data and AST checkpoints

Prepare each dataset from the repository root:

```bash
python data/download_esc50.py --output_dir data/esc50/raw
python data/prepare_esc50.py \
  --esc50_root data/esc50/raw/ESC-50-master \
  --out_root data/esc50 --repo_root .

python data/download_urbansound8k.py --output_dir data/urbansound8k/raw
python data/prepare_urbansound8k.py \
  --urbansound8k_root data/urbansound8k/raw \
  --out_root data/urbansound8k --repo_root .

python data/download_cremad.py --output_dir data/cremad/raw
python data/prepare_cremad.py \
  --cremad_root data/cremad/raw \
  --out_root data/cremad --repo_root . \
  --val_fraction 0.1 --split_seed 42
```

The reported AST baselines are downloaded automatically from Hugging Face:

- `Adam-ousse/ast-esc50-finetuned-fold1`
- `Adam-ousse/ast-urbansound8k-finetuned-fold10`
- `Adam-ousse/ast-cremad-finetuned`

Dataset audio, manifests containing local paths, model caches, activations, and
experiment outputs are excluded from Git.

## Concept generation

The default CLI preserves the original LF-only behavior. Explicit modes produce
separate source files and provenance metadata:

```bash
python generate_deepseek_concept_sets.py --dataset esc50 --mode all --stage generate
python generate_deepseek_concept_sets.py --dataset urbansound8k --mode all --stage generate
python generate_deepseek_concept_sets.py \
  --datasets esc50 urbansound8k --mode all --stage process --device cuda
```

Outputs are written under `data/concept_sets/<dataset>/`. The broad vocabulary
is generated once and reused. See
[`notebooks/DeepSeek_audio_concept_generation.ipynb`](notebooks/DeepSeek_audio_concept_generation.ipynb)
for an interactive walkthrough.

The latest CREMA-D protocol is isolated so it cannot overwrite the original ablation:

```bash
python scripts/generate_cremad_targeted_concepts.py \
  --output-root results/cremad_targeted_rerun_20260828/generation
python scripts/run_cremad_targeted_source_ablation.py \
  --experiment-root results/cremad_targeted_rerun_20260828 --device cuda
```

## Training and evaluation

Train the shared source ablation:

```bash
python evaluate_ast_baselines.py --datasets esc50 urbansound8k cremad --device cuda
python run_audio_concept_ablation.py \
  --datasets esc50 urbansound8k \
  --variants lf lf_broad lf_contrastive full --device cuda
```

The paper hyperparameters are the script defaults for ESC-50/UrbanSound8K:
seed 42, 1,000 projection steps, cosine-cubed similarity, activation cutoff
0.25, projection cutoff 0.45, 1,000 SAGA iterations, `lambda=0.0007`, and
elastic-net mixing 0.99. CREMA-D uses cutoff 0.40 and `lambda=0.0015`, selected
on validation once and frozen across the four targeted variants.

After restoring the external bundle, evaluate every canonical AST/CBM with the
same 1-second window and 0.5-second hop:

```bash
python run_segmented_audio_ablation.py \
  --datasets esc50 urbansound8k cremad \
  --artifact-bundle release/google_drive_bundle --device cuda
```

## External Google Drive bundle

Canonical CBM weights are intentionally kept outside Git. Build the folder to
upload from existing results:

```bash
python scripts/build_google_drive_bundle.py
```

This creates `release/google_drive_bundle/` containing 12 canonical checkpoints,
the exact concept inputs and provenance, compact metrics, and `manifest.json`
with SHA-256 checksums. Cached AST/CLAP activations, raw data, tuning trials, and
superseded CREMA-D runs are excluded. Verify a local or downloaded copy with:

```bash
python scripts/build_google_drive_bundle.py \
  --output release/google_drive_bundle --verify-only
```

## Interactive GitHub Pages site

The static explorer in `docs/` uses the ESC-50 LF+broad model. It supports audio
playback, correct/error filtering, exact top-five concept-signal intervention,
and temporal concept plots.

```bash
python scripts/build_esc50_showcase_assets.py \
  --samples-per-class 2 --max-concepts 5
python -m http.server 8000
```

Then open `http://localhost:8000/docs/`. See [`docs/README.md`](docs/README.md)
for implementation details and GitHub Pages publishing instructions.

## Citation

This repository builds on the original LF-CBM work:

```bibtex
@misc{oikarinen2023labelfreeconceptbottleneckmodels,
  title={Label-Free Concept Bottleneck Models},
  author={Tuomas Oikarinen and Subhro Das and Lam M. Nguyen and Tsui-Wei Weng},
  year={2023},
  eprint={2304.06129},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2304.06129}
}
```
