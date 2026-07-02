"""
==========================================================
Grad-CAM — SaarthiEO
==========================================================

Generates a Grad-CAM heatmap for EfficientNet-B3.

Target layer: model.features[-1]
    (last MBConv block — richest spatial activations)

Public API
----------
    generate_gradcam(image, target_class=None)
        -> (original: PIL.Image, overlay: PIL.Image)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CLASSES, CLASS_LABELS, CLASS_EMOJI
from .model import load_model
from .utils import preprocess_image

# ----------------------------------------------------------
# Device
# ----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Load model (singleton, shared with inference.py via import)
# ----------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_model()
        _model.to(device)
        _model.eval()
    return _model


# ----------------------------------------------------------
# Grad-CAM core
# ----------------------------------------------------------

class _GradCam:
    """
    Registers forward/backward hooks on a target layer and
    computes the class-weighted activation map.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        """
        Parameters
        ----------
        x : torch.Tensor  [1, C, H, W]
        class_idx : int

        Returns
        -------
        np.ndarray  [H_orig, W_orig] float32 in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(x)                        # forward
        score  = logits[0, class_idx]
        score.backward()                              # backward

        # Global average pool over spatial dims
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam     = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ----------------------------------------------------------
# Colour map helper
# ----------------------------------------------------------

def _apply_colormap(cam_np, size):
    """
    Resize cam_np to (size, size) and apply jet colormap.
    Returns uint8 RGB numpy array.
    """
    import cv2  # only needed here

    cam_resized = cv2.resize(cam_np, (size, size))
    heatmap     = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


# ----------------------------------------------------------
# Public API
# ----------------------------------------------------------

def generate_gradcam(image, target_class=None):
    """
    Generate a Grad-CAM explanation for the given image.

    Parameters
    ----------
    image : PIL.Image or str
        Input image.
    target_class : str or None
        Class key (e.g. "fire").  If None, uses the predicted class.

    Returns
    -------
    original_pil : PIL.Image   — resized to display size
    overlay_pil  : PIL.Image   — heatmap overlay
    pred_label   : str         — "🔥 Fire"
    confidence   : float       — 0–1
    """
    model = _get_model()

    # ---- Preprocess ----
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    orig_rgb = np.array(image.convert("RGB"))
    h, w = orig_rgb.shape[:2]

    x = preprocess_image(image).to(device)

    # ---- Resolve target class ----
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

    if target_class is None:
        pred_idx = probs.argmax().item()
    else:
        pred_idx = CLASSES.index(target_class)

    confidence  = probs[pred_idx].item()
    class_key   = CLASSES[pred_idx]
    pred_label  = f"{CLASS_EMOJI[class_key]} {CLASS_LABELS[class_key]}"

    # ---- Grad-CAM ----
    target_layer = model.features[-1]
    gcam = _GradCam(model, target_layer)

    x_grad = preprocess_image(image).to(device)
    x_grad.requires_grad_(False)

    cam = gcam(x_grad, pred_idx)
    gcam.remove()

    # ---- Build overlay ----
    import cv2

    # Resize original for display
    display_size = 400
    orig_display = image.convert("RGB").resize(
        (display_size, display_size), Image.LANCZOS
    )
    orig_np = np.array(orig_display)

    heatmap_rgb = _apply_colormap(cam, display_size)

    # Blend: 55% original + 45% heatmap
    overlay_np = (0.55 * orig_np + 0.45 * heatmap_rgb).astype(np.uint8)

    original_pil = orig_display
    overlay_pil  = Image.fromarray(overlay_np)

    return original_pil, overlay_pil, pred_label, confidence
