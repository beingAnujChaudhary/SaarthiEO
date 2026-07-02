---
title: SaarthiEO Disaster Detection
emoji: 🌍
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "4.44.1"
app_file: app_hf.py
pinned: true
license: mit
short_description: Disaster detection from satellite & drone imagery
---

# 🌍 SaarthiEO — Disaster Detection from Satellite & Drone Imagery

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-FFD21E)](https://huggingface.co/spaces/beingAnujChaudhary/SaarthiEO-Disaster-Detection)
[![License](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-97.38%25-FE320A)](#-results)

**An end-to-end AI pipeline for classifying natural and man-made disasters from satellite and drone imagery — featuring Grad-CAM explainability, real-time inference, and a production-grade web interface.**

[🚀 Live Demo](https://huggingface.co/spaces/beingAnujChaudhary/SaarthiEO-Disaster-Detection) &nbsp;·&nbsp; [📓 Notebooks](notebooks/) &nbsp;·&nbsp; [🤗 Model](https://huggingface.co/beingAnujChaudhary/models) &nbsp;·&nbsp; [👤 Author](#-author)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [App Features](#-app-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Training Process](#-training-process)
- [Results](#-results)
- [Grad-CAM Explainability](#-grad-cam-explainability)
- [Run on Google Colab](#-run-on-google-colab)
- [Setup & Run Locally](#-setup--run-locally)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🌐 Overview

**SaarthiEO** *(Saarthi = "companion" in Hindi · EO = Earth Observation)* is a deep learning system that classifies disaster events captured in satellite and drone imagery into four critical categories:

| Category | Description |
|---|---|
| 🔥 **Fire** | Active wildfires or structural fires visible from aerial view |
| 🌊 **Flood** | Surface water inundation of normally dry land |
| 🏚️ **Collapsed Building** | Structural collapse from earthquake, explosion, or other events |
| 🚗 **Traffic Incident** | Major road accidents, pile-ups, or blocked arterial roads |

Built for **rapid first-responder triage** — upload an image, get a prediction and confidence score within milliseconds, and view a Grad-CAM heatmap showing *exactly which regions drove the prediction*.

---

## ✨ App Features

| Feature | Details |
|---|---|
| 🔍 **Predict Tab** | Upload any satellite/drone image → top class + all 4 probability bars + inference time |
| 🧠 **Grad-CAM Tab** | Side-by-side original vs. heatmap overlay — see what the model focused on |
| 📦 **Batch Predict Tab** | Upload multiple images at once → results table → download full CSV report |
| ⚡ **Fast Inference** | ~60 ms on CPU · ~10 ms on GPU |
| 🎯 **Example Images** | One-click examples for all 4 disaster classes |
| 📱 **Clean UI** | Custom portfolio-themed design — warm beige, orange-red accents |

---

## 🎬 Demo

### Tab 1 — Predict

Upload any satellite or drone image and instantly receive:
- **Top predicted class** with coloured badge
- **Confidence score** (large gradient number)
- **All 4 class probability bars** sorted by confidence
- **Model metadata** — inference time, device, image size, model name

### Tab 2 — Grad-CAM Explainability

See a side-by-side view of the original image and its Grad-CAM heatmap:

> 🔴 **Warm (red/orange) regions** = areas the model focused on most  
> 🔵 **Cool (blue) regions** = low attention areas

The model correctly focuses on the fire plumes, flood water, rubble patterns, and vehicle density — not the background.

### Tab 3 — Batch Predict

Upload 10, 50, or 100 images at once. Get a full results table with:
- Predicted class per image
- Confidence score
- All 4 class probabilities
- Inference time per image
- One-click CSV download

---

## 🏗️ Architecture

```
Input Image (any resolution)
        │
        ▼
  Resize → 300 × 300
        │
        ▼
  ImageNet Normalisation
  (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        │
        ▼
┌──────────────────────────────────────────────────┐
│              EfficientNet-B3 Backbone            │
│  Pretrained on ImageNet · Fine-tuned on dataset  │
│                                                  │
│  features[0]  Conv2d stem                        │
│  features[1]  MBConv ×2  (stride 1)              │
│  features[2]  MBConv ×3  (stride 2)              │
│  features[3]  MBConv ×3  (stride 2)              │
│  features[4]  MBConv ×5  (stride 2)              │
│  features[5]  MBConv ×5  (stride 1)              │
│  features[6]  MBConv ×6  (stride 2)              │
│  features[7]  MBConv ×2  (stride 1)              │
│  features[8]  Conv2d head → 1536 channels        │
│                                                  │
│  AdaptiveAvgPool2d → 1536-dim vector             │
└──────────────────────────────────────────────────┘
        │
        ▼
  Custom Classifier Head
  ┌─────────────────────┐
  │  Dropout (p=0.4)    │
  │  Linear 1536 → 512  │
  │  ReLU               │
  │  BatchNorm1D (512)  │
  │  Dropout (p=0.3)    │
  │  Linear 512 → 4     │
  └─────────────────────┘
        │
        ▼
  Softmax → 4 Class Probabilities
```

**Why EfficientNet-B3?**
- 12M parameters — lightweight but powerful
- Compound scaling balances depth, width, and resolution uniformly
- Strong ImageNet pretraining → excellent low-level spatial feature extraction
- Native 300×300 input resolution is ideal for aerial imagery detail

---

## 📊 Dataset

| Class | Train | Validation | Test |
|---|---|---|---|
| 🔥 Fire | ~700 | ~150 | ~150 |
| 🌊 Flood | ~700 | ~150 | ~150 |
| 🏚️ Collapsed Building | ~700 | ~150 | ~150 |
| 🚗 Traffic Incident | ~700 | ~150 | ~150 |
| **Total** | **~2,800** | **~600** | **~600** |

**Data augmentation applied during training:**

| Augmentation | Parameters |
|---|---|
| Random Horizontal Flip | p = 0.5 |
| Random Vertical Flip | p = 0.3 |
| Random Rotation | ±15° |
| Color Jitter | brightness=0.2, contrast=0.2 |
| Normalisation | ImageNet mean & std |

---

## 🏋️ Training Process

Training was done in two phases on Google Colab (T4 GPU).

### Phase 1 — Feature Extraction (Frozen Backbone)

The EfficientNet-B3 backbone was frozen. Only the custom classifier head was trained.

| Setting | Value |
|---|---|
| Epochs | 10 |
| Optimiser | Adam |
| Learning Rate | 1e-3 |
| Scheduler | ReduceLROnPlateau (patience=3) |
| Loss | CrossEntropyLoss |

### Phase 2 — Full Fine-Tuning (All Layers Unfrozen)

All layers were unfrozen and the entire network was fine-tuned end-to-end with a lower learning rate.

| Setting | Value |
|---|---|
| Epochs | 20 |
| Optimiser | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Early Stopping | Patience = 7 |
| Loss | CrossEntropyLoss |

Best checkpoint saved by validation accuracy → `models/best_efficientnet_b3.pth`

---

## 📈 Results

### Test Set Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **97.38%** |
| Top-1 Error Rate | 2.62% |

### Per-Class Metrics (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 🔥 Fire | ~97% | ~97% | ~97% | ~150 |
| 🌊 Flood | ~98% | ~98% | ~98% | ~150 |
| 🏚️ Collapsed Building | ~96% | ~96% | ~96% | ~150 |
| 🚗 Traffic Incident | ~98% | ~98% | ~98% | ~150 |
| **Weighted Avg** | **~97%** | **~97%** | **~97%** | **~600** |

> 📊 Full confusion matrix and training curves available in `notebooks/`.

---

## 🔥 Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is built directly into the app's **Explain** tab.

**How it works:**

1. A forward pass produces class logits → the target class score is selected
2. Gradients flow backward to `model.features[-1]` (last MBConv block — richest spatial activations)
3. Gradient magnitudes are averaged spatially → per-channel importance weights
4. Weights × activations are summed → a single spatial attention map
5. ReLU applied → only positive influences kept
6. Map is upsampled to 400×400 and blended with the original image using a jet colormap (55% original + 45% heatmap)

**Why this matters:**  
The model is not a black box. Grad-CAM shows that it correctly focuses on fire plumes, flood water, building rubble, and vehicle density — making it suitable for high-stakes disaster response applications where **explainability is essential**.

---

## ☁️ Run on Google Colab

Open `app_colab.ipynb` in your Drive and run all 4 cells:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Set project path
import os
PROJECT_PATH = "/content/drive/MyDrive/SaarthiEO/disasterDetection"
os.chdir(PROJECT_PATH)
print(os.getcwd())

# Cell 3 — Install missing packages
# (torch + torchvision already pre-installed on Colab)
!pip install -q gradio opencv-python-headless pandas

# Cell 4 — Launch the full 3-tab app
!python app/app.py
```

A public Gradio URL appears within ~30 seconds:
```
* Running on public URL: https://xxxxxxxx.gradio.live
```

> ⚡ Switch to **T4 GPU** runtime for ~6× faster inference.

---

## 💻 Setup & Run Locally

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (optional)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/beingAnujChaudhary/SaarthiEO.git
cd SaarthiEO

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
python app/app.py
```

Opens at `http://localhost:7860` with a public share link printed in the terminal.

### GPU support

Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) before step 3. The app detects CUDA automatically and shows **GPU 🚀** in the metadata panel.

---

## 📁 Project Structure

```
SaarthiEO/
│
├── app/
│   └── app.py                    # Full 3-tab Gradio application
│
├── app_hf.py                     # Hugging Face Spaces entry point
├── app_colab.ipynb               # Google Colab launcher notebook
│
├── src/
│   ├── config.py                 # Class names, colours, emoji, model path
│   ├── model.py                  # EfficientNet-B3 architecture + weight loader
│   ├── inference.py              # Single-image predict() → all probs + timing
│   ├── gradcam.py                # Grad-CAM heatmap generator
│   ├── batch_predict.py          # Batch inference → pandas DataFrame
│   └── utils.py                  # Image preprocessing transform pipeline
│
├── models/
│   └── best_efficientnet_b3.pth  # Trained model weights (~46 MB)
│
├── notebooks/                    # Training, evaluation, GradCAM notebooks
│
├── examples/                     # Sample images for all 4 classes
│   ├── fire.jpg
│   ├── flood.jpg
│   ├── collapsed_building.jpg
│   └── traffic_incident.jpg
│
├── assets/                       # Banner, screenshots, demo GIFs
├── docs/                         # Additional documentation
│
├── requirements.txt              # Python dependencies
├── DEPLOY.md                     # Step-by-step HF Spaces deployment guide
├── .gitignore
└── README.md
```

---

## 🔭 Future Work

- [ ] **Multi-label classification** — one image may show fire + flood simultaneously
- [ ] **Severity scoring** — mild / moderate / severe damage scale per class
- [ ] **Geolocation tagging** — parse GPS EXIF metadata from drone images
- [ ] **Video inference** — frame-by-frame analysis of drone footage streams
- [ ] **Alert webhook** — real-time notification to first-responder systems
- [ ] **Edge deployment** — quantised INT8 ONNX model for on-device drone inference
- [ ] **Larger dataset** — AIDER, EMSR, and DisasterGAN synthetic augmentation
- [ ] **Transformer backbone** — experiment with ViT-B/16 and Swin-T

---

## 👤 Author

**Anuj Chaudhary**

Transitioning into AI/ML engineering. SaarthiEO demonstrates a complete end-to-end ML workflow: data preparation, model fine-tuning, explainability, deployment, and software engineering.

<div align="center">

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-beingAnujChaudhary-FFD21E)](https://huggingface.co/beingAnujChaudhary)
[![GitHub](https://img.shields.io/badge/GitHub-beingAnujChaudhary-181717?logo=github)](https://github.com/beingAnujChaudhary)

</div>

---

> **📄 Resume Bullet Point**
>
> **SaarthiEO — Disaster Detection from Satellite & Drone Imagery** *(PyTorch · Gradio · Hugging Face)*
> - Fine-tuned EfficientNet-B3 on a 4-class aerial disaster dataset achieving **97.38% test accuracy** (Fire, Flood, Collapsed Building, Traffic Incident).
> - Built an end-to-end Grad-CAM explainability pipeline hooking the last convolutional block to produce interpretable attention heatmaps.
> - Deployed a production-grade 3-tab Gradio web app (single predict, Grad-CAM, batch CSV export) to Hugging Face Spaces with Git LFS model storage.

---

<div align="center">
Made with ❤️ &nbsp;·&nbsp; SaarthiEO &nbsp;·&nbsp; EfficientNet-B3 + PyTorch + Gradio
</div>
