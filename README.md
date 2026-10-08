# CALIPOSE: Benchmarking Architectural Paradigms for Event-Based Spacecraft Pose Estimation

Official PyTorch implementation and pre-trained weights for our WACV submission evaluating **Direct Regression** versus **Modular Keypoint-Tracking** on the SPADES dataset.

---

## 🛠️ Repository Contents
* **`train.py`**: Complete training pipeline, data loaders, 3-channel event tensor builder, spatial-temporal augmentations, and the `SparkV30Net` architecture.
* **`weights/spark_v30_best.pth`**: Pre-trained model weights achieving state-of-the-art orientation robustness on the SPADES real test split.

---

## 🚀 Quick Start Guide

### 1. Environment Requirements
Ensure your environment has PyTorch, Torchvision, H5Py, and SciPy installed:
```bash
pip install torch torchvision h5py numpy scipy pillow
