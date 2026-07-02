"""
==========================================================
Inference Pipeline — SaarthiEO
==========================================================

Returns:
    {
        "class":        str,         # top-1 class key
        "label":        str,         # human-readable label
        "confidence":   float,       # top-1 probability (0–1)
        "probabilities": {str: float},  # all class probabilities
        "inference_ms": float,       # wall-clock time in ms
    }
"""

import time

import torch

from .model import load_model
from .utils import preprocess_image
from .config import CLASSES, CLASS_LABELS

# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Load model once at import time
# ----------------------------------------------------------

model = load_model()
model.to(device)
model.eval()


# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------

@torch.no_grad()
def predict(image):
    """
    Run inference on a single PIL image.

    Parameters
    ----------
    image : PIL.Image or str
        Input image (or file path).

    Returns
    -------
    dict
    """

    x = preprocess_image(image).to(device)

    # ---- Timed forward pass ----
    t0 = time.perf_counter()
    outputs = model(x)
    t1 = time.perf_counter()

    inference_ms = (t1 - t0) * 1000.0

    # ---- Probabilities ----
    probs = torch.softmax(outputs, dim=1).squeeze(0)  # shape [NUM_CLASSES]

    # Top-1
    confidence, pred_idx = probs.max(dim=0)
    top_class = CLASSES[pred_idx.item()]

    # All classes
    probabilities = {
        cls: probs[i].item()
        for i, cls in enumerate(CLASSES)
    }

    return {
        "class":         top_class,
        "label":         CLASS_LABELS[top_class],
        "confidence":    confidence.item(),
        "probabilities": probabilities,
        "inference_ms":  inference_ms,
    }