# CALIPOSE: Benchmarking Architectural Paradigms for Event-Based Spacecraft Pose Estimation

Official PyTorch implementation and pre-trained weights for our WACV submission evaluating **Direct Regression** versus **Modular Keypoint-Tracking** on the SPADES dataset.

---

## 🛠️ Repository Contents
* **`train.py`**: Complete training pipeline, data loaders, 3-channel event tensor builder, spatial-temporal augmentations, and the `SparkV30Net` architecture.


---

## 🚀 Quick Start & Pre-Trained Weights

### 1. Environment Requirements
Ensure your environment has PyTorch, Torchvision, H5Py, and SciPy installed:
```bash
pip install torch torchvision h5py numpy scipy pillow
* **Pre-Trained Weights (Hugging Face):** Download `spark_v30_best.pth` directly from [Hugging Face Model Hub](https://huggingface.co/your-username/calipose-sparkv30).
