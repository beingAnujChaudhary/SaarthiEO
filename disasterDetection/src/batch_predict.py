"""
==========================================================
Batch Prediction — SaarthiEO
==========================================================

Accepts a list of file paths, returns a pandas DataFrame
with one row per image.
"""

import os
import time

import pandas as pd
from PIL import Image

from .inference import predict
from .config import CLASS_LABELS, CLASS_EMOJI


def batch_predict(file_paths):
    """
    Run inference on a list of image file paths.

    Parameters
    ----------
    file_paths : list[str]

    Returns
    -------
    pd.DataFrame with columns:
        filename, prediction, emoji, confidence_pct,
        collapsed_building_pct, fire_pct, flood_pct,
        traffic_incident_pct, inference_ms
    """

    rows = []

    for path in file_paths:
        filename = os.path.basename(path)

        try:
            image = Image.open(path).convert("RGB")
            result = predict(image)

            row = {
                "filename":              filename,
                "prediction":            CLASS_LABELS[result["class"]],
                "emoji":                 CLASS_EMOJI[result["class"]],
                "confidence_%":          f"{result['confidence'] * 100:.2f}",
                "collapsed_building_%":  f"{result['probabilities']['collapsed_building'] * 100:.2f}",
                "fire_%":                f"{result['probabilities']['fire'] * 100:.2f}",
                "flood_%":               f"{result['probabilities']['flood'] * 100:.2f}",
                "traffic_incident_%":    f"{result['probabilities']['traffic_incident'] * 100:.2f}",
                "inference_ms":          f"{result['inference_ms']:.1f}",
                "error":                 "",
            }

        except Exception as e:
            row = {
                "filename":              filename,
                "prediction":            "ERROR",
                "emoji":                 "❌",
                "confidence_%":          "",
                "collapsed_building_%":  "",
                "fire_%":                "",
                "flood_%":               "",
                "traffic_incident_%":    "",
                "inference_ms":          "",
                "error":                 str(e),
            }

        rows.append(row)

    return pd.DataFrame(rows)
