# ece-5831-002-traffic-sign-transfer

Traffic Sign Recognition (GTSRB) – Transfer Learning Demo

This project trains a traffic-sign classifier on the **German Traffic Sign Recognition Benchmark (GTSRB)** using **transfer learning** in PyTorch. It supports:

- Pretrained backbones (MobileNetV3-Small by default; optional ResNet50)
- Two-stage training (classifier head warmup, then fine-tuning)
- Data augmentation + optional MixUp/CutMix
- Evaluation (Accuracy, Macro-F1) + calibration (Expected Calibration Error / Reliability diagram)
- Explainability via Grad-CAM overlays

## Project links (placeholders)
- Pre-recorded presentation video: https://drive.google.com/drive/folders/1xX9xt8DHLD0jopRsqk1K-PxYZTpx20hL
- Presentation slides: https://drive.google.com/drive/folders/1i6l5jL5bQ46xAsmetbvVSEBkz-WLZ1Ez
- Report: https://drive.google.com/drive/folders/1rF0IqoAqhDJlCV5A78DZwtcG5TPjx9ty
- Dataset: https://drive.google.com/drive/folders/1fHJuCiBvjKcZ2v0lQarXsgM4GfaqT_qS
- Demo video: https://www.youtube.com/watch?v=IPZow5NnAMQ

## Recommended way to run (Notebook)
Open and run `final-project.ipynb`. It contains cells to:

1) Install dependencies
2) Download/verify the dataset
3) Train + evaluate
4) Visualize metrics/plots and Grad-CAM outputs
5) Run a demo prediction on one sample image

## Environment setup
### Python version
Use **Python 3.11 or 3.12**. If you use Python 3.13, `pip install torch` may fail because PyTorch wheels are typically not available yet.

### Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### Jupyter kernel (for the notebook)
```bash
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name gtsrb-transfer --display-name "Python (gtsrb-transfer)"
```
Then choose **Python (gtsrb-transfer)** as the notebook kernel.

## CLI Quickstart (optional)
If you prefer running via CLI, set `PYTHONPATH` so the project can import modules from `src/`:

```bash
export PYTHONPATH="$PWD/src"
```

1) Download data (uses torchvision's built-in GTSRB loader)
```bash
python -m scripts.download_gtsrb --data-root data
```

2) Train + evaluate (head warmup + fine-tune)
```bash
python -m src.train --data-root data --output-dir outputs --run-gradcam
```

Artifacts land in `outputs/` (or your chosen `--output-dir`):
- `best_model.pt`: best checkpoint (dict containing `model` state and `epoch`)
- `metrics.json`: summary metrics + config + per-epoch history
- `metrics_summary.png`: bar chart of val/test metrics
- `confusion_matrix.png`: confusion matrix heatmap
- `reliability.png`: reliability diagram (calibration)
- `gradcam/*.jpg`: Grad-CAM overlays (when enabled)

## Dataset
- **Dataset**: GTSRB (43 classes)
- **Loader**: `torchvision.datasets.GTSRB` (auto-download supported)
- **Storage location**: if `--data-root data`, torchvision will place files under `data/gtsrb/`
- **Offline use**: if you already have the data present, add `--no-download` to training

## Method (what the code does)
### Transfer learning backbones
- `mobilenet_v3_small` (default): efficient backbone for quick iteration
- `resnet50`: larger backbone for potentially higher accuracy (more compute)

### Two-stage training
Training is split into two phases:
1) **Head warmup** (`--head-epochs`): freeze the backbone and train only the classifier head
2) **Fine-tuning** (`--epochs`): unfreeze the last N backbone blocks (`--unfreeze-blocks`) and fine-tune end-to-end

### Training/evaluation loop (high level)
`src/train.py` implements the following workflow:
1) Build dataloaders with an internal train/val split (`--val-split`, seeded)
2) Build a pretrained model, replace its classifier for 43 classes
3) Warm up the classifier head (backbone frozen)
4) Unfreeze the last backbone blocks and fine-tune
5) Track the best checkpoint by **validation accuracy**
6) At the end: generate plots (confusion matrix, reliability diagram, metrics summary) and save `metrics.json`
7) Optionally: export a few Grad-CAM overlays from the validation set (`--run-gradcam`)

### Optimization + regularization
- Optimizer: **AdamW** (`--head-lr`, `--lr`, `--weight-decay`)
- LR scheduler: `onecycle` (default), `cosine`, or `none`
- Label smoothing: `--label-smoothing`
- Optional MixUp/CutMix: `--mixup-alpha`, `--cutmix-alpha`
- Optional class balancing: `--balanced-sampler`

### Augmentations
Training transforms include (see `src/data.py`):
- Random resized crop, rotation, color jitter, horizontal flip, and occasional blur
Evaluation transforms use deterministic resize + normalization.

### Metrics and calibration
Evaluation computes:
- **Accuracy**
- **Macro-F1** (robust to class imbalance)
- **Expected Calibration Error (ECE)** + reliability diagram

## Key training flags
See `python -m src.train -h` for the full list. Commonly used flags:
- `--model {mobilenet_v3_small,resnet50}`: backbone architecture
- `--img-size`: input resize (square)
- `--batch-size`, `--num-workers`: performance/throughput knobs
- `--head-epochs`, `--epochs`: warmup + fine-tune epoch counts
- `--unfreeze-blocks`: number of backbone blocks to unfreeze for fine-tuning
- `--scheduler {onecycle,cosine,none}`: learning rate schedule
- `--mixup-alpha`, `--cutmix-alpha`, `--label-smoothing`: regularization
- `--balanced-sampler`: class-balanced sampling for training
- `--run-gradcam`: export Grad-CAM overlays into `OUTPUT_DIR/gradcam/`
- `--no-download`: do not attempt to download the dataset
- `--no-pretrained`: train from scratch (ablation; slower and usually worse)

## Suggested runs
All commands assume:
```bash
export PYTHONPATH="$PWD/src"
```

### Smoke test (fast)
```bash
python -m src.train --data-root data --output-dir outputs/smoke --head-epochs 1 --epochs 1 --num-workers 0 --skip-test --no-download
```

### Default run (better metrics)
```bash
python -m src.train --data-root data --output-dir outputs/run_default --run-gradcam
```

## Demo inference + Grad-CAM
Use any trained checkpoint to classify an image and optionally save a Grad-CAM overlay:
```bash
export PYTHONPATH="$PWD/src"
python -m scripts.predict --image path/to/sign.jpg --checkpoint outputs/best_model.pt --gradcam
```
Output: top prediction with confidence and an overlay image (default: `outputs/demo_gradcam.jpg`).

## Repository structure
- `final-project.ipynb`: end-to-end notebook runner
- `src/data.py`: dataset config, transforms/augmentations, balanced sampling
- `src/models.py`: model builders (MobileNetV3-Small, ResNet50) and freeze/unfreeze helpers
- `src/train.py`: training loop (warmup + fine-tune) + evaluation + plotting + Grad-CAM export
- `src/gradcam.py`: lightweight Grad-CAM implementation and overlay rendering
- `src/utils.py`: metrics helpers and plotting utilities
- `scripts/download_gtsrb.py`: dataset download helper
- `scripts/predict.py`: single-image inference + optional Grad-CAM

## Troubleshooting
- **`pip` can’t find `torch`**: check `python -V` (use 3.11/3.12), upgrade pip (`python -m pip install -U pip`), then reinstall.
- **Notebook DataLoader hangs**: set `--num-workers 0` (the notebook defaults to 0).
- **No GPU**: training will run on CPU if CUDA is unavailable; reduce epochs/batch size for a quick smoke test.
