# Cloud Fine-Tuning Guide

This guide prepares a cloud GPU machine for Places365 building-scene fine-tuning.

## 1. Directory Layout

Use this layout on the cloud machine:

```text
/workspace/image-inpainting-research/
├── data/
│   ├── places365_raw/              # extracted Places365 images
│   └── places365_building/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── big-lama/
│       ├── config.yaml
│       └── models/best.ckpt
└── outputs/
    ├── train_runs/
    └── tb_logs/
```

The training configs assume this root by default. Override `location.*` values if your cloud path is different.

## 2. Environment

```bash
cd /workspace/image-inpainting-research
conda create -n lama-ft python=3.10 -y
conda activate lama-ft
pip install -r requirements.txt
```

Install a PyTorch build matching the cloud CUDA version if the image does not already provide one.

Set runtime variables:

```bash
export PROJECT_ROOT=/workspace/image-inpainting-research
export TORCH_HOME=$PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT
export MPLCONFIGDIR=/tmp/matplotlib
```

## 3. Prepare Places365 Building Subset

After downloading and extracting Places365 images into `data/places365_raw`, build a manageable subset:

```bash
python bin/prepare_places365_building_subset.py \
  --source-root data/places365_raw \
  --output-root data/places365_building \
  --train-count 3000 \
  --val-count 300 \
  --test-count 300 \
  --mode symlink
```

Use `--mode copy` if the cloud storage does not support symlinks.

Building categories are listed in `configs/places365_building_categories.txt`.

## 4. Smoke Test

Run a short job before the full fine-tune:

```bash
python bin/train.py -cn big-lama-building-default-ft \
  trainer.kwargs.max_epochs=1 \
  trainer.kwargs.limit_train_batches=20 \
  trainer.kwargs.val_check_interval=20 \
  data.batch_size=1 \
  data.num_workers=2
```

This checks data loading, checkpoint initialization, validation, and checkpoint writing.

## 5. Ordinary Building Fine-Tune

This is the first formal model: same Big-LaMa structure, building subset, default random mask distribution.

```bash
python bin/train.py -cn big-lama-building-default-ft
```

Useful overrides for a 24GB GPU:

```bash
python bin/train.py -cn big-lama-building-default-ft \
  data.batch_size=2 \
  trainer.kwargs.max_epochs=10 \
  trainer.kwargs.limit_train_batches=1000
```

## 6. Mask-Optimized Fine-Tune

Train this from the same original Big-LaMa checkpoint after the ordinary fine-tune is complete:

```bash
python bin/train.py -cn big-lama-building-maskopt-ft
```

The mask-optimized config uses a weighted mask generator:

```text
thin scratch-like masks: 15%
medium irregular masks: 30%
large irregular masks: 45%
large box masks: 10%
```

This creates a controlled comparison:

```text
Original Big-LaMa
Building ordinary fine-tune
Building mask-optimized fine-tune
```

## 7. Output

Hydra writes each run to:

```text
outputs/train_runs/<user>_<date>_<job>_<config>_<run_title>/
```

Each run contains:

```text
config.yaml
models/last.ckpt
models/*.ckpt
samples/
```

Download only the selected checkpoint, config, TensorBoard logs, and visual samples back to the local machine.
