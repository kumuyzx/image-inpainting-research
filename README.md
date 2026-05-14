# LaMa Inpainting Thesis

Fork-style LaMa image inpainting project for thesis work.

This project is based on the open-source LaMa implementation:
https://github.com/advimman/lama

The main work is intended to optimize LaMa itself. The `backend/` and
`frontend/` directories provide a simple local web demo for easier testing.
The original upstream README is kept in `UPSTREAM_LAMA_README.md`.

## Structure

```text
image-inpainting-thesis/
├── bin/              # Upstream LaMa scripts
├── configs/          # Upstream LaMa configs
├── saicinpainting/   # Core LaMa source code
├── backend/          # Local HTTP API, calls LaMa inference
├── frontend/         # Browser UI
├── models/           # Model weights and upstream auxiliary models
├── outputs/          # Runtime uploads, generated results, previous runs
└── samples/          # Example images and masks
```

## Run

```bash
cd /home/kumu/dev/image-inpainting-thesis
conda activate paper_env

export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)
export MPLCONFIGDIR=/tmp/matplotlib

python backend/server.py --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

## API

```text
POST /api/inpaint
```

Multipart fields:

- `image`: source image
- `mask`: mask image; non-black areas are inpainted
