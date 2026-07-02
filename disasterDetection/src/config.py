# ==========================================================
# Configuration — SaarthiEO Disaster Detection
# ==========================================================

IMAGE_SIZE = 300

NUM_CLASSES = 4

# Internal class keys (must match training order)
CLASSES = [
    "collapsed_building",
    "fire",
    "flood",
    "traffic_incident",
]

# Human-readable display labels (used in UI)
CLASS_LABELS = {
    "collapsed_building": "Collapsed Building",
    "fire": "Fire",
    "flood": "Flood",
    "traffic_incident": "Traffic Incident",
}

# Emoji per class (used in UI badges)
CLASS_EMOJI = {
    "collapsed_building": "🏚️",
    "fire": "🔥",
    "flood": "🌊",
    "traffic_incident": "🚗",
}

# Hex accent colours per class (used in UI)
CLASS_COLORS = {
    "collapsed_building": "#e67e22",   # orange
    "fire":               "#e74c3c",   # red
    "flood":              "#3498db",   # blue
    "traffic_incident":   "#f1c40f",   # yellow
}

# Model path (relative to project root)
MODEL_PATH = "models/best_efficientnet_b3.pth"

# ImageNet normalisation constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]