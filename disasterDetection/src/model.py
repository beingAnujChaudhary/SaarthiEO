"""
==========================================================
Model Definition
EfficientNet-B3 for Disaster Detection
==========================================================
"""

import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b3,
    EfficientNet_B3_Weights,
)

from src.config import (
    MODEL_PATH,
    NUM_CLASSES,
)


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Build Model
# ==========================================================

def build_model():

    """
    Builds the EfficientNet-B3 architecture.
    """

    model = efficientnet_b3(weights=None)

    model.classifier = nn.Sequential(

        nn.Dropout(0.4),

        nn.Linear(1536, 512),

        nn.ReLU(),

        nn.BatchNorm1d(512),

        nn.Dropout(0.3),

        nn.Linear(512, NUM_CLASSES)

    )

    return model


# ==========================================================
# Load Trained Model
# ==========================================================

def load_model(model_path=MODEL_PATH):

    """
    Loads trained weights into EfficientNet-B3.

    Parameters
    ----------
    model_path : str

    Returns
    -------
    model
    """

    model = build_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
    )

    # ------------------------------------------------------
    # Handle different checkpoint formats
    # ------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif "state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint["state_dict"]
            )

        else:

            model.load_state_dict(checkpoint)

    else:

        model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# ==========================================================
# Model Summary
# ==========================================================

if __name__ == "__main__":

    model = load_model()

    print(model)

    print("\nModel loaded successfully!")

    print(f"Running on: {DEVICE}")