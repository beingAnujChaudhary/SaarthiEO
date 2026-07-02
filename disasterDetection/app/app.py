"""
==========================================================
SaarthiEO — Disaster Detection App
==========================================================

Run:
    python app/app.py          (local)
    !python app/app.py         (Colab — via app_colab.ipynb)

Tabs:
    1. Predict       — single image, all class probabilities, inference time
    2. Explain       — Grad-CAM heatmap overlay
    3. Batch         — multi-image upload → results table → CSV download
==========================================================
"""

import sys
import os
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import gradio as gr
import pandas as pd
from PIL import Image

from src.inference import predict
from src.gradcam import generate_gradcam
from src.batch_predict import batch_predict
from src.config import CLASS_LABELS, CLASS_COLORS, CLASS_EMOJI, CLASSES

# ----------------------------------------------------------
# Custom CSS — Portfolio theme
# Background : #EFEAE3  (warm off-white)
# Accent     : #FE320A  (orange-red)
# Text       : #1a1a1a
# Font       : DM Sans (loaded from Google Fonts) ≈ Neue Haas
# ----------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Reset / Base ── */
:root {
    --bg:       #EFEAE3;
    --card:     #E6E0D8;
    --border:   rgba(26,26,26,0.12);
    --accent:   #FE320A;
    --text:     #1a1a1a;
    --muted:    rgba(26,26,26,0.45);
    --radius:   14px;
    --shadow:   0 4px 24px rgba(26,26,26,0.08);
}

body,
.gradio-container,
.gradio-container > .main,
.gradio-container > .main > .wrap,
#root,
.app {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                 Helvetica, Arial, sans-serif !important;
    color: var(--text) !important;
}

/* Focus ring */
*:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
}

/* ── Header ── */
.saarthi-header {
    padding: 36px 32px 28px;
    text-align: center;
    margin-bottom: 8px;
}
.saarthi-header h1 {
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 700;
    letter-spacing: -1px;
    background: linear-gradient(to right, orange, #FE320A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.1;
}
.saarthi-header p {
    color: #1a1a1a;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Pill tags (sub-header chips) ── */
.pill-row {
    display: flex;
    justify-content: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.pill {
    display: inline-block;
    padding: 4px 14px;
    border: 1px solid rgba(26,26,26,0.3);
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #1a1a1a;
    background: transparent;
    transition: color 0.3s ease, background-color 0.3s ease, border-color 0.3s ease;
}
.pill:hover {
    color: var(--accent);
    border-color: var(--accent);
}

/* ── Divider ── */
.saarthi-divider {
    height: 1px;
    background: var(--border);
    margin: 0 0 24px 0;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
    box-shadow: var(--shadow);
    margin-bottom: 14px;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

/* ── Prediction badge ── */
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 22px;
    border-radius: 50px;
    font-size: 1.25rem;
    font-weight: 600;
    margin-top: 4px;
    border: 1px solid rgba(26,26,26,0.15);
    color: var(--text);
    background: rgba(254,50,10,0.07);
    border-color: rgba(254,50,10,0.25);
}
.badge-fire            { background: rgba(254,50,10,0.09);  border-color: rgba(254,50,10,0.3); }
.badge-flood           { background: rgba(52,152,219,0.09); border-color: rgba(52,152,219,0.3); }
.badge-collapsed_building { background: rgba(180,100,20,0.09); border-color: rgba(180,100,20,0.3); }
.badge-traffic_incident   { background: rgba(200,160,0,0.09);  border-color: rgba(200,160,0,0.3); }

/* ── Confidence number ── */
.confidence-value {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(to right, orange, #FE320A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    letter-spacing: -2px;
    margin-top: 4px;
}

/* ── Probability bars ── */
.prob-row { margin-bottom: 12px; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 5px;
    color: var(--text);
}
.prob-pct { color: var(--muted); font-weight: 400; }
.prob-bar-bg {
    background: rgba(26,26,26,0.08);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.7s cubic-bezier(.23,1,.32,1);
}

/* ── Meta panel ── */
.meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.meta-item {
    background: rgba(255,255,255,0.45);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
.meta-key {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--muted);
    margin-bottom: 4px;
    font-weight: 500;
}
.meta-val {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
}

/* ── Tabs ── */
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
.tab-nav button {
    color: var(--muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 10px 22px !important;
    border-radius: 50px !important;
    border: 1px solid transparent !important;
    transition: all 0.3s ease !important;
    margin-bottom: 4px !important;
}
.tab-nav button:hover {
    color: var(--accent) !important;
    border-color: rgba(254,50,10,0.2) !important;
    background: rgba(254,50,10,0.04) !important;
}
.tab-nav button.selected {
    color: var(--accent) !important;
    border-color: rgba(254,50,10,0.3) !important;
    background: rgba(254,50,10,0.07) !important;
}

/* ── Primary button — pill style ── */
button.primary, .gr-button-primary, button[variant="primary"] {
    background-color: #1a1a1a !important;
    color: #EFEAE3 !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    cursor: pointer !important;
    transition: background-color 0.3s ease, transform 0.15s ease !important;
    will-change: transform !important;
}
button.primary:hover, .gr-button-primary:hover, button[variant="primary"]:hover {
    background-color: #FE320A !important;
    transform: translateY(-1px) !important;
}

/* ── Inputs / uploads ── */
.gradio-container input, .gradio-container textarea,
.gradio-container select {
    background: rgba(255,255,255,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.gradio-container label > span {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
.gr-image, .gr-file {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: rgba(255,255,255,0.45) !important;
}

/* ── GradCAM info card ── */
.gcam-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 22px;
    text-align: center;
    box-shadow: var(--shadow);
}
.gcam-label {
    font-size: 1.1rem;
    font-weight: 600;
    background: linear-gradient(to right, orange, #FE320A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
}
.gcam-legend {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 6px;
}

/* ── Dataframe ── */
.gradio-container table {
    background: rgba(255,255,255,0.5) !important;
    border-radius: 10px !important;
}

/* ── Footer ── */
.saarthi-footer {
    text-align: center;
    padding: 24px 0 12px;
    border-top: 1px solid var(--border);
    margin-top: 16px;
    font-size: 0.82rem;
    color: var(--muted);
}
.saarthi-footer a {
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
    transition: opacity 0.2s ease;
}
.saarthi-footer a:hover { opacity: 0.7; }

/* Reduced motion */
@media (prefers-reduced-motion) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        will-change: auto !important;
    }
}
"""

# ----------------------------------------------------------
# Bar colours (accent-driven for light theme)
# ----------------------------------------------------------

BAR_COLORS = {
    "fire":               "linear-gradient(90deg, orange, #FE320A)",
    "flood":              "linear-gradient(90deg, #3498db, #74b9ff)",
    "collapsed_building": "linear-gradient(90deg, #b46414, #e67e22)",
    "traffic_incident":   "linear-gradient(90deg, #c8a000, #f1c40f)",
}


# ----------------------------------------------------------
# Helper — HTML fragments
# ----------------------------------------------------------

def _prob_bar(cls_key, prob_pct):
    color = BAR_COLORS[cls_key]
    label = f"{CLASS_EMOJI[cls_key]} {CLASS_LABELS[cls_key]}"
    return f"""
    <div class="prob-row">
        <div class="prob-label">
            <span>{label}</span>
            <span class="prob-pct">{prob_pct:.1f}%</span>
        </div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill"
                 style="width:{prob_pct:.1f}%;background:{color}"></div>
        </div>
    </div>"""


def _result_html(result):
    import torch
    cls        = result["class"]
    label      = CLASS_LABELS[cls]
    emoji      = CLASS_EMOJI[cls]
    conf_pct   = result["confidence"] * 100
    probs      = result["probabilities"]
    inf_ms     = result["inference_ms"]
    device_lbl = "GPU 🚀" if torch.cuda.is_available() else "CPU"

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    bars = "".join(_prob_bar(k, v * 100) for k, v in sorted_probs)

    return f"""
    <div class="card">
        <div class="card-title">Prediction</div>
        <div class="pred-badge badge-{cls}">{emoji} {label}</div>
    </div>

    <div class="card">
        <div class="card-title">Confidence</div>
        <div class="confidence-value">{conf_pct:.2f}%</div>
    </div>

    <div class="card">
        <div class="card-title">Class Probabilities</div>
        {bars}
    </div>

    <div class="card">
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-key">Model</div>
                <div class="meta-val">EfficientNet-B3</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Image Size</div>
                <div class="meta-val">300 × 300</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Inference Time</div>
                <div class="meta-val">{inf_ms:.1f} ms</div>
            </div>
            <div class="meta-item">
                <div class="meta-key">Device</div>
                <div class="meta-val">{device_lbl}</div>
            </div>
        </div>
    </div>
    """


def _empty_hint(text):
    return f"<p style='color:rgba(26,26,26,0.35);text-align:center;margin-top:60px;font-size:0.9rem'>{text}</p>"


# ----------------------------------------------------------
# Tab 1 — Predict
# ----------------------------------------------------------

def classify(image):
    if image is None:
        return _empty_hint("Upload an image to see results.")
    return _result_html(predict(image))


# ----------------------------------------------------------
# Tab 2 — GradCAM
# ----------------------------------------------------------

def explain(image, target_display):
    if image is None:
        return None, None, _empty_hint("Upload an image to generate explanation.")

    display_to_key = {v: k for k, v in CLASS_LABELS.items()}
    target_key = display_to_key.get(target_display)

    original_pil, overlay_pil, pred_label, confidence = generate_gradcam(
        image, target_class=target_key
    )

    info_html = f"""
    <div class="gcam-card">
        <div class="gcam-label">{pred_label} &nbsp; {confidence*100:.2f}%</div>
        <div class="gcam-legend">
            🔴 Warm regions = high model attention
            &nbsp;·&nbsp;
            🔵 Cool regions = low attention
        </div>
    </div>
    """
    return original_pil, overlay_pil, info_html


# ----------------------------------------------------------
# Tab 3 — Batch
# ----------------------------------------------------------

def batch_run(files):
    if not files:
        return None, None

    paths = [f.name for f in files]
    df = batch_predict(paths)

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", prefix="saarthi_batch_"
    )
    df.to_csv(tmp.name, index=False)
    tmp.close()

    return df, tmp.name


# ----------------------------------------------------------
# Build app
# ----------------------------------------------------------

HEADER_HTML = """
<div class="saarthi-header">
    <h1>🌍 SaarthiEO Disaster Detection</h1>
    <p style="color:#1a1a1a">AI-powered classification of disasters from satellite &amp; drone imagery</p>
    <div class="pill-row">
        <span class="pill">EfficientNet-B3</span>
        <span class="pill">97.38% Accuracy</span>
        <span class="pill">Grad-CAM XAI</span>
        <span class="pill">PyTorch</span>
    </div>
</div>
<div class="saarthi-divider"></div>
"""

FOOTER_HTML = """
<div class="saarthi-footer">
    SaarthiEO Disaster Detection &nbsp;·&nbsp; EfficientNet-B3 &nbsp;·&nbsp;
    <a href="https://huggingface.co/beingAnujChaudhary" target="_blank">🤗 Hugging Face</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/beingAnujChaudhary" target="_blank">GitHub</a>
</div>
"""

EXAMPLE_IMAGES = [
    [os.path.join(PROJECT_ROOT, "examples", "fire.jpg")],
    [os.path.join(PROJECT_ROOT, "examples", "flood.jpg")],
    [os.path.join(PROJECT_ROOT, "examples", "collapsed_building.jpg")],
    [os.path.join(PROJECT_ROOT, "examples", "traffic_incident.jpg")],
]
EXAMPLE_IMAGES = [e for e in EXAMPLE_IMAGES if os.path.exists(e[0])]

with gr.Blocks(
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
    ),
    title="SaarthiEO Disaster Detection",
) as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ── Tab 1: Predict ───────────────────────────────
        with gr.Tab("🔍  Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        type="pil",
                        label="Upload Satellite / Drone Image",
                        height=340,
                        sources=["upload", "clipboard"],
                    )
                    predict_btn = gr.Button(
                        "Analyse Image", variant="primary", size="lg"
                    )
                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=EXAMPLE_IMAGES,
                            inputs=[img_input],
                            label="Quick Examples",
                            examples_per_page=4,
                        )

                with gr.Column(scale=1):
                    result_html = gr.HTML(
                        _empty_hint("Upload an image and click <strong>Analyse Image</strong>.")
                    )

            predict_btn.click(fn=classify, inputs=[img_input], outputs=[result_html])
            img_input.change(fn=classify, inputs=[img_input], outputs=[result_html])

        # ── Tab 2: Grad-CAM ──────────────────────────────
        with gr.Tab("🧠  Explain (Grad-CAM)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gcam_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        height=300,
                        sources=["upload", "clipboard"],
                    )
                    target_class_dd = gr.Dropdown(
                        choices=["Auto (predicted class)"] + list(CLASS_LABELS.values()),
                        value="Auto (predicted class)",
                        label="Target Class (optional)",
                    )
                    gcam_btn = gr.Button(
                        "Generate Grad-CAM", variant="primary", size="lg"
                    )
                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=EXAMPLE_IMAGES,
                            inputs=[gcam_input],
                            label="Quick Examples",
                            examples_per_page=4,
                        )

                with gr.Column(scale=2):
                    with gr.Row():
                        orig_out    = gr.Image(label="Original", height=300)
                        overlay_out = gr.Image(label="Grad-CAM Overlay", height=300)
                    gcam_info = gr.HTML()

            def _explain_wrapper(image, target_display):
                key = None if target_display == "Auto (predicted class)" else target_display
                return explain(image, key)

            gcam_btn.click(
                fn=_explain_wrapper,
                inputs=[gcam_input, target_class_dd],
                outputs=[orig_out, overlay_out, gcam_info],
            )

        # ── Tab 3: Batch Predict ─────────────────────────
        with gr.Tab("📦  Batch Predict"):
            gr.Markdown(
                "Upload multiple images at once. Results are shown as a table "
                "and you can download a full CSV report."
            )
            with gr.Row():
                batch_input = gr.File(
                    file_count="multiple",
                    label="Upload Images (JPG / PNG)",
                    file_types=["image"],
                )
                batch_btn = gr.Button(
                    "Run Batch Analysis", variant="primary", size="lg"
                )
            batch_table = gr.Dataframe(label="Results", interactive=False, wrap=True)
            batch_csv   = gr.File(label="⬇ Download CSV")

            batch_btn.click(
                fn=batch_run,
                inputs=[batch_input],
                outputs=[batch_table, batch_csv],
            )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True,
        server_name="0.0.0.0",
    )