# Major Project

This repository contains a deep learning project for image classification and segmentation.

## Structure

- `train/` - training scripts for classifier and UNet models
- `models/` - model definitions for ResNet50 and UNet
- `inference/` - inference and visualization utilities
- `data/` - dataset folders for classification and localization

## Train classifier

```bash
python train/train_classifier.py
```

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

- Do not commit model weights (`*.pth`) or large dataset files.
- The repository already contains a `.gitignore` configured to ignore weights and cached Python files.
