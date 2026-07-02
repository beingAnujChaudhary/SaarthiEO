# 🚀 Deploy SaarthiEO to Hugging Face Spaces

This guide takes you from zero to a live public URL in ~10 minutes.

---

## Prerequisites

- [x] Hugging Face account: https://huggingface.co/beingAnujChaudhary
- [x] Git installed
- [x] Git LFS installed (for the `.pth` model file)

### Install Git LFS (if you haven't already)

```bash
# Windows — download installer from:
# https://git-lfs.github.com/

# Then initialise:
git lfs install
```

---

## Step 1 — Create the Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Owner:** `beingAnujChaudhary`
   - **Space name:** `SaarthiEO-Disaster-Detection`
   - **SDK:** `Gradio`
   - **Hardware:** `CPU basic` (free tier — sufficient for demos)
3. Click **Create Space**

You will see a placeholder page at:
```
https://huggingface.co/spaces/beingAnujChaudhary/SaarthiEO-Disaster-Detection
```

---

## Step 2 — Clone the Space repository

```bash
git clone https://huggingface.co/spaces/beingAnujChaudhary/SaarthiEO
cd SaarthiEO
```

---

## Step 3 — Copy your project files into the Space

Copy everything from your local `disasterDetection/` folder into the cloned Space folder. The structure must look like this at the Space root:

```
SaarthiEO/              ← Space repo root
├── app/
│   └── app.py
├── app_hf.py           ← HF will run THIS as the entry point
├── src/
│   ├── config.py
│   ├── model.py
│   ├── inference.py
│   ├── gradcam.py
│   ├── batch_predict.py
│   └── utils.py
├── models/
│   └── best_efficientnet_b3.pth   ← 46 MB — tracked by LFS
├── examples/
│   ├── fire.jpg
│   ├── flood.jpg
│   ├── collapsed_building.jpg
│   └── traffic_incident.jpg
├── requirements.txt
└── README.md
```

> [!IMPORTANT]
> HF Spaces looks for the Gradio `demo` object. Our `app_hf.py` at the root imports `demo` from `app/app.py` — this is the correct entry point. Make sure `app_hf.py` exists at the root.

---

## Step 4 — Track the model with Git LFS

The `.pth` file is 46 MB — too large for regular Git. Use LFS:

```bash
# Inside the Space repo:
git lfs track "*.pth"
git add .gitattributes
```

---

## Step 5 — Add a Space configuration card

HF Spaces requires a YAML card at the top of `README.md`. Add these lines at the very beginning of your `README.md`:

```yaml
---
title: SaarthiEO Disaster Detection
emoji: 🌍
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: app_hf.py
pinned: false
license: mit
---
```

---

## Step 6 — Commit and push

```bash
git add .
git commit -m "feat: initial SaarthiEO deployment"
git push
```

> [!NOTE]
> The first push may take a few minutes because Git LFS uploads the 46 MB model file. Subsequent pushes will be fast.

---

## Step 7 — Watch it build

Go to your Space URL:
```
https://huggingface.co/spaces/beingAnujChaudhary/SaarthiEO-Disaster-Detection
```

Click the **Logs** tab. You will see:
```
Installing requirements...
Loading model...
Running on: http://0.0.0.0:7860
```

Once the build is complete the Space goes live automatically. 🎉

---

## Updating the Space

Whenever you make changes locally:

```bash
# Copy updated files to the Space repo, then:
git add .
git commit -m "fix: ..."
git push
```

HF Spaces auto-rebuilds on every push.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: cv2` | Make sure `opencv-python-headless` is in `requirements.txt` (not `opencv-python`) |
| Model not found | Check that `models/best_efficientnet_b3.pth` is tracked by LFS |
| Space crashes immediately | Check the **Logs** tab on HF — usually a missing import or wrong `app_file` in README card |
| LFS quota exceeded | Free tier has 1 GB LFS storage. If exceeded, host model on HF Hub model repo and load with `hf_hub_download()` |
