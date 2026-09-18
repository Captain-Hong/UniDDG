# UniDDG

Training and evaluation code for the paper **"Rethinking Domain Generalization in Medical Image Segmentation: One Image as One Domain"** ([arXiv:2501.04741](https://arxiv.org/abs/2501.04741)).

The implementation supports multi-center optic cup/disc segmentation and prostate segmentation. During training, UniDDG separates anatomy and image style, recombines them to synthesize style variations, and enforces segmentation and reconstruction consistency. Evaluation uses only the anatomy encoder and segmentation head.

## Installation

Python 3.10 or newer is required.

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m pip install -r requirements.txt
```

## Dataset

The Fundus and Prostate datasets are available from [Google Drive](https://drive.google.com/file/d/1xb00zAzWP1tnWybVBUBn2Wmkbv6yYMqY/view?usp=sharing).

Each domain split must contain an `image` directory, a `mask` directory, and a `filenames.txt` file. Images and masks are stored as NumPy arrays with matching filenames.

```text
npy_data/
├── D1/test/
│   ├── image/
│   ├── mask/
│   └── filenames.txt
├── D2/train/
├── D3/train/
├── D4/train/
├── Da/train/
├── Db/train/
├── Dc/train/
├── Dd/train/
└── De/test/
```

The Fundus configuration trains on D2, D3, and D4 and evaluates on D1. The Prostate configuration trains on Da, Db, Dc, and Dd and evaluates on De. Other leave-one-domain-out settings can be defined by editing a YAML configuration.

## Training

Fundus:

```bash
python train.py \
  --config configs/fundus_234_to_1.yaml \
  --data optic \
  --root /path/to/npy_data
```

Prostate:

```bash
python train.py \
  --config configs/prostate_abcd_to_e.yaml \
  --data prostate \
  --root /path/to/npy_data
```

Training writes TensorBoard logs, `best_model.pth`, `last_model.pth`, and `result.json` under `logs/`.

## Evaluation

Fundus:

```bash
python test.py \
  --config configs/fundus_234_to_1.yaml \
  --data optic \
  --root /path/to/npy_data \
  --weight_path logs/<run>/best_model.pth
```

Prostate:

```bash
python test.py \
  --config configs/prostate_abcd_to_e.yaml \
  --data prostate \
  --root /path/to/npy_data \
  --weight_path logs/<run>/best_model.pth
```

Fundus evaluation reports Dice and average surface distance (ASD) for the optic cup and disc. Prostate evaluation reports segmentation Dice and ASD. Per-case Dice scores can be written to CSV, and aggregate results can be written to JSON with `--save_json`.

## Repository Layout

```text
train.py             Training entry point
test.py              Checkpoint evaluation entry point
oiod_core.py         Model construction, losses, training, and evaluation loops
dataset.py           NumPy dataset loader
config/              Default configuration schema
configs/             Fundus and Prostate experiment configurations
models/              Anatomy/style encoders, decoder, and segmentation head
utils/               Augmentation, losses, metrics, and post-processing
```

Datasets, checkpoints, logs, and generated results are intentionally excluded from version control.
