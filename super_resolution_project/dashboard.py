"""
Streamlit Dashboard for Image Super-Resolution Benchmarking.

Launch with:
    streamlit run dashboard.py
"""

import io
import numpy as np
from PIL import Image
import streamlit as st

from utils import degrade_image, numpy_to_pil, pil_to_numpy
from metrics import calculate_psnr, calculate_ssim
from models import (
    NearestModel,
    BilinearModel,
    BicubicModel,
    StackedInterpolationModel,
    SRCNNModel,
    VDSRModel,
)

# ------------------------------------------------------------------ #
#  Page configuration
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="SR Benchmark Dashboard",
    page_icon="🔬",
    layout="wide",
)

# ------------------------------------------------------------------ #
#  Custom CSS for premium look
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSlider > div > div {
        background: #6c63ff;
    }

    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.6rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 8px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(108, 99, 255, 0.1);
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #6c63ff;
    }

    /* Image containers */
    .img-container {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.25);
    }

    /* Success execute button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ #
#  Header
# ------------------------------------------------------------------ #
st.markdown('<div class="main-title">🔬 Image Super-Resolution Benchmark</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Compare interpolation &amp; deep-learning upscaling methods side-by-side</div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
#  Sidebar — configuration
# ------------------------------------------------------------------ #
with st.sidebar:
    st.header("⚙️ Configuration")

    # ---------- Image upload ----------
    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an HR ground truth image",
        type=["jpg", "jpeg", "png"],
    )

    # ---------- Degradation ----------
    st.subheader("🔽 Degradation")
    scale_factor = st.slider(
        "Downscale Factor",
        min_value=2,
        max_value=16,
        value=4,
        step=1,
        help="The LR image is created by shrinking the HR image by this factor.",
    )

    # ---------- Model selection ----------
    st.subheader("🧠 Models")
    use_nearest = st.checkbox("Nearest Neighbour", value=True)
    use_bilinear = st.checkbox("Bilinear", value=True)
    use_bicubic = st.checkbox("Bicubic", value=True)
    use_srcnn = st.checkbox("SRCNN (Lornatang pretrained)", value=False)
    use_vdsr = st.checkbox("VDSR (model_epoch_50 pretrained)", value=True)

    # ---------- Custom 2-stage stack ----------
    st.subheader("🔗 Custom 2-Stage Stack")
    use_stack = st.toggle("Enable Stacked Model", value=False)

    MODEL_OPTIONS = {"Nearest": NearestModel, "Bilinear": BilinearModel, "Bicubic": BicubicModel}

    if use_stack:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            stage1_name = st.selectbox("Stage 1 Model", list(MODEL_OPTIONS.keys()), index=2, key="s1_model")
            stage1_scale = st.number_input("Stage 1 Scale", min_value=2, max_value=8, value=2, step=1, key="s1_scale")
        with col_s2:
            stage2_name = st.selectbox("Stage 2 Model", list(MODEL_OPTIONS.keys()), index=1, key="s2_model")
            stage2_scale = st.number_input("Stage 2 Scale", min_value=2, max_value=8, value=2, step=1, key="s2_scale")

# ------------------------------------------------------------------ #
#  Helper: render a metric card (HTML)
# ------------------------------------------------------------------ #
def metric_html(label: str, value: float, unit: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value:.2f}{unit}</div>'
        f"</div>"
    )

# ------------------------------------------------------------------ #
#  Main area — execute pipeline
# ------------------------------------------------------------------ #
if uploaded_file is not None:
    # Load HR image
    hr_pil = Image.open(uploaded_file).convert("RGB")
    hr_image = np.array(hr_pil, dtype=np.uint8)

    # Degrade
    lr_image = degrade_image(hr_image, scale_factor)

    # ---------- Show HR / LR side-by-side ----------
    st.markdown("---")
    st.subheader("📸 Input Images")
    col_hr, col_lr = st.columns(2)
    with col_hr:
        st.image(hr_image, caption=f"HR Ground Truth — {hr_image.shape[1]}×{hr_image.shape[0]}", use_container_width=True)
    with col_lr:
        st.image(lr_image, caption=f"LR Degraded ({scale_factor}×) — {lr_image.shape[1]}×{lr_image.shape[0]}", use_container_width=True)

    # ---------- Execute button ----------
    st.markdown("---")
    execute = st.button("🚀 Execute Pipeline", use_container_width=True)

    if execute:
        # Collect (name, model_instance) pairs
        models_to_run: list[tuple[str, object]] = []
        has_error = False

        if use_nearest:
            models_to_run.append(("Nearest", NearestModel(scale_factor)))
        if use_bilinear:
            models_to_run.append(("Bilinear", BilinearModel(scale_factor)))
        if use_bicubic:
            models_to_run.append(("Bicubic", BicubicModel(scale_factor)))
        if use_srcnn:
            srcnn = SRCNNModel(scale_factor)
            weights_path = "weights/srcnn_x3-T91-919a959c.pth.tar"
            try:
                srcnn.load_weights(weights_path)
                models_to_run.append(("SRCNN", srcnn))
            except FileNotFoundError:
                st.error(f"⚠️ Error: Pretrained weights not found at {weights_path}")
                has_error = True

        if use_vdsr:
            vdsr = VDSRModel(scale_factor)
            vdsr_weights = "weights/model_epoch_50.pth"
            try:
                vdsr.load_weights(vdsr_weights)
                models_to_run.append(("VDSR", vdsr))
            except FileNotFoundError:
                st.error(f"⚠️ Error: Pretrained weights not found at {vdsr_weights}")
                has_error = True

        if use_stack:
            m1 = MODEL_OPTIONS[stage1_name](stage1_scale)
            m2 = MODEL_OPTIONS[stage2_name](stage2_scale)
            stacked = StackedInterpolationModel(m1, stage1_scale, m2, stage2_scale)
            models_to_run.append((stacked.name, stacked))

        if has_error:
            st.stop()
            
        if not models_to_run:
            st.warning("⚠️ Select at least one model from the sidebar.")
        else:
            st.subheader("📊 Results")

            # --- Run all models -------------------------------------------
            results: list[tuple[str, np.ndarray, float, float]] = []
            progress = st.progress(0, text="Processing…")

            for idx, (name, model) in enumerate(models_to_run):
                sr_image = model.predict(lr_image)

                # Resize SR to match HR dimensions for metric calculation
                if sr_image.shape[:2] != hr_image.shape[:2]:
                    sr_pil = Image.fromarray(sr_image, mode="RGB")
                    sr_pil = sr_pil.resize(
                        (hr_image.shape[1], hr_image.shape[0]), Image.BICUBIC
                    )
                    sr_image = np.array(sr_pil, dtype=np.uint8)

                psnr = calculate_psnr(hr_image, sr_image)
                ssim = calculate_ssim(hr_image, sr_image)
                results.append((name, sr_image, psnr, ssim))
                progress.progress(
                    (idx + 1) / len(models_to_run),
                    text=f"Processed {name}",
                )

            progress.empty()

            # --- Display results in columns --------------------------------
            cols = st.columns(len(results))
            for col, (name, sr_img, psnr, ssim) in zip(cols, results):
                with col:
                    st.image(
                        sr_img,
                        caption=f"{name} — {sr_img.shape[1]}×{sr_img.shape[0]}",
                        use_container_width=True,
                    )
                    st.markdown(metric_html("PSNR", psnr, " dB"), unsafe_allow_html=True)
                    st.markdown(metric_html("SSIM", ssim), unsafe_allow_html=True)

            # --- Summary table --------------------------------------------
            st.markdown("---")
            st.subheader("📋 Summary Table")
            table_data = {
                "Model": [r[0] for r in results],
                "PSNR (dB)": [f"{r[2]:.2f}" for r in results],
                "SSIM": [f"{r[3]:.4f}" for r in results],
            }
            st.table(table_data)

else:
    st.info("👈 Upload a high-resolution image from the sidebar to begin.")
