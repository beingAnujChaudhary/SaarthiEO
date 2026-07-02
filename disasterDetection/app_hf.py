"""
==========================================================
SaarthiEO — Hugging Face Spaces Entry Point
==========================================================

This file is the entry point for the Hugging Face Space.
It is identical to app/app.py but adapted for the Spaces
environment:
  - share=False  (HF handles the public URL)
  - Paths resolved relative to this file (repo root)
==========================================================
"""

# Re-export the demo object from app/app.py so HF Spaces
# picks it up. HF Spaces expects a `demo` variable that
# is a gr.Blocks / gr.Interface object.

import sys
import os

# Make sure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Import and re-export the app
from app.app import demo  # noqa: F401

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
