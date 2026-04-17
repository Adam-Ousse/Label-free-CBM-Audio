# ESC-50 Showcase (GitHub Pages)

This folder contains a static ESC-50 showcase UI.

## What it does
- ESC-50 only
- Class picker with emoji/icon
- Exactly 2 examples per class
- For each example:
  - audio player
  - predicted class + confidence
  - top concept contribution bars (CBM explanation)
  - metadata (fold, duration)

## Regenerate assets
From repository root:

```bash
/home/ensta/ensta-gassem/dl_env/bin/python scripts/build_esc50_showcase_assets.py --samples-per-class 2
```

If you want GPU generation through Slurm:

```bash
srun --pty --time=02:00:00 --partition=ENSTA-l40s --gpus=1 --nodelist=ensta-l40s02.r2.enst.fr /home/ensta/ensta-gassem/dl_env/bin/python scripts/build_esc50_showcase_assets.py --samples-per-class 2 --max-concepts 10
```

This command updates:
- `assets/audio/*.wav`
- `assets/data/esc50_showcase.json`
- `assets/data/esc50_showcase.js`

## Local preview
From repository root:

```bash
python -m http.server 8000
```

Open:
- `http://localhost:8000/docs/`

Direct file open also works (`file:///.../docs/index.html`) because the page uses
`assets/data/esc50_showcase.js` as an embedded fallback when browser `fetch()` is
blocked by CORS on `file://` origins.

## GitHub Pages
- In repository settings, set Pages source to `main` branch and `/docs` folder.
- Site entrypoint is `docs/index.html`.
