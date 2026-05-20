# CPR-MIAD

CPR-MIAD is a multi-class anomaly detection project built around prototype-guided feature correction and reconstruction-based anomaly localization.

## Overview

This repository contains the training, testing, and visualization code used for CPR-MIAD experiments.

Main components:

- `model/cpr.py`: CPR-MIAD model implementation.
- `configs/uniad/uniad_mvtec_cpr.py`: CPR-MIAD experiment configuration.
- `trainer/uniad_trainer.py`: training, testing, metrics, and visualization workflow.
- `data/`: dataset loading and metadata generation utilities.
- `util/`: metrics, visualization, registry, and common utilities.

## Installation

Create a Python environment and install the required deep learning dependencies.

```shell
pip install torch torchvision torchaudio
pip install timm pandas transformers openpyxl imgaug numba numpy tensorboard fvcore scikit-image fastprogress geomloss
```

Install any CUDA-specific packages according to your local PyTorch and CUDA versions.

## Dataset

Prepare your anomaly detection dataset under `data/`, then generate or provide the corresponding `meta.json` file.

For supported dataset formats and metadata examples, see `data/README.md`.

## Training

Edit the dataset path and experiment settings in:

```text
configs/uniad/uniad_mvtec_cpr.py
```

Run training:

```shell
python run.py -c configs/uniad/uniad_mvtec_cpr.py -m train
```

## Testing

Set `trainer.resume_dir` or `model.kwargs["checkpoint_path"]` in the config file, then run:

```shell
python run.py -c configs/uniad/uniad_mvtec_cpr.py -m test
```

## Visualization

```shell
python run.py -c configs/uniad/uniad_mvtec_cpr.py -m test vis=True vis_dir=visualization
```

## Project Status

This repository is maintained for CPR-MIAD experiments and related multi-class anomaly detection research.
