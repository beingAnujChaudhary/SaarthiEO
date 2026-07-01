# 🛰️ SaarthiEO

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-ffea00.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-1da1f2.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)

**Semantic AI Assistant for Earth Observation Retrieval**  
*A cross-modal satellite image retrieval system bridging the gap between Optical and SAR modalities.*

---

## 📖 The Problem: The Modality Gap

Earth observation relies heavily on multi-sensor data. Optical imagery (Sentinel-2) is easy for humans to interpret but is blocked by cloud cover and night. SAR imagery (Sentinel-1) penetrates clouds and works at night but is highly complex and non-intuitive to analyze. 

**SaarthiEO** solves this by enabling **cross-modal semantic retrieval**. It allows users to query an Optical image and instantly retrieve the geographically and semantically matching SAR image (and vice versa) using advanced Vision Foundation Models.

---

## 🚀 Key Features

* **🔍 Cross-Modal Retrieval:** Query with Sentinel-2 (Optical) to find matching Sentinel-1 (SAR) data, or vice versa.
* **🧠 Vision Foundation Models:** Utilizes pre-trained Contrastive Language-Image Pretraining (CLIP) and RemoteCLIP models from Hugging Face for powerful zero-shot feature extraction.
* **⚡ Blazing Fast Vector Search:** Employs Facebook AI Similarity Search (FAISS) for `<100ms` retrieval times over thousands of image embeddings.
* **🌍 Geospatial Preprocessing:** Custom 3-channel composite generation using `rasterio` to handle complex multi-band `.tif` files.
* **☁️ Colab-Native Architecture:** Entirely optimized to run in Google Colab with Google Drive integration, bypassing heavy local hardware requirements.

---

## 🏗️ Architecture & Workflow

1. **Data Ingestion:** Downloads a subset of the **SEN12MS** dataset (aligned Sentinel-1 and Sentinel-2 patches) via Kaggle API.
2. **Preprocessing:** 
   * *Optical:* Extracts Bands 4, 3, 2 (RGB) and normalizes.
   * *SAR:* Extracts VV, VH, creates a VV/VH ratio, applies log-transform, and scales to a 3-channel pseudo-RGB format.
3. **Embedding Generation:** Passes preprocessed images through `openai/clip-vit-base-patch32` to generate 512-dimensional vector representations.
4. **Indexing:** Stores embeddings in a `faiss.IndexFlatIP` (Cosine Similarity) vector database.
5. **Dashboard:** A Streamlit web application running via `localtunnel` for real-time visual querying.

---

## 📁 Repository Structure

```text
SaarthiEO/
├── app/
│   └── app.py                     # Streamlit dashboard
├── data/
│   ├── raw/                       # Raw SEN12MS .tif files
│   ├── processed/                 # 3-channel normalized arrays
│   └── gallery/                   # Sample images for UI display
├── embeddings/
│   ├── optical_embeddings.npy     # Cached CLIP vectors
│   ├── sar_embeddings.npy
│   └── faiss_index.index          # Saved FAISS index
├── notebooks/
│   ├── 01_explore_data.ipynb      # EDA and dataset fetching
│   ├── 02_preprocess.ipynb        # Rasterio pipeline
│   ├── 03_embeddings.ipynb        # Embedding generation & FAISS build
│   └── 04_retrieval.ipynb         # Testing retrieval logic
├── src/
│   ├── data/preprocess.py         # Data handling classes
│   ├── models/encoder.py          # Hugging Face model wrapper
│   ├── retrieval/index.py         # FAISS index builder
│   ├── retrieval/search.py        # Query execution
│   └── utils/helpers.py           # Configs and utilities
└── requirements.txt               # Dependencies

```

---

## 🛠️ Getting Started (Google Colab)

This project is built to run on Google Colab using Google Drive for storage.

### 1. Mount Google Drive

In your Colab notebook, mount your drive and append the project source to your path:

```python
from google.colab import drive
import sys
drive.mount('/content/drive')
sys.path.append('/content/drive/MyDrive/SaarthiEO/')

```

### 2. Install Dependencies

```bash
!pip install -r /content/drive/MyDrive/SaarthiEO/requirements.txt

```

*(Key dependencies include: `torch`, `transformers`, `faiss-cpu`, `rasterio`, `streamlit`, `localtunnel`)*

### 3. Fetch Dataset (Kaggle API)

Upload your `kaggle.json` token to Colab and run:

```bash
!kaggle datasets download -d sanketdhamdare/sen12ms-dataset -p /content/drive/MyDrive/SaarthiEO/data/raw --unzip

```

---

## 🖥️ Running the Streamlit Dashboard

Because Colab blocks standard web ports, we use `localtunnel` to expose the Streamlit UI. Run this in a Colab cell:

```bash
# 1. Start the Streamlit app in the background
!streamlit run /content/drive/MyDrive/SaarthiEO/dashboard/app.py &>/content/logs.txt &

# 2. Expose the port via localtunnel
!npx localtunnel --port 8501

```

Click the generated `localtunnel` URL to access the SaarthiEO interactive dashboard!

---

## 👥 Team

* **Anuj Chaudhary** - [GitHub Profile](https://www.google.com/search?q=https://github.com/%5BYOUR-GITHUB-USERNAME%5D)
* **Ankur Garg** - [GitHub Profile](https://www.google.com/search?q=https://github.com/%5BTEAMMATE-2-USERNAME%5D)
* **Likith Gopal Ganiga** - [GitHub Profile](https://www.google.com/search?q=https://github.com/%5BTEAMMATE-3-USERNAME%5D)

---

## 📊 Dataset Attribution

This project utilizes a subset of the **SEN12MS** dataset for demonstration purposes.

> Schmitt, M., Hughes, L. H., Qiu, C., & Zhu, X. X. (2019). *SEN12MS – A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion.* ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.

---

## 📜 License

This project is licensed under the MIT License.
