from huggingface_hub import hf_hub_download
import os
import gc

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from model import HybridSegModel

MODEL_CONFIGS = {
    "🩻 Breast Ultrasound (BUSI)": {
        "weights": "busi_best_weights.pth",
        "in_channels": 1,
        "img_size": 384,
        "desc": "Breast lesion segmentation from ultrasound images"
    },
}

torch.set_num_threads(1)
DEVICE = torch.device("cpu")

def ensure_weights(fname):
    if not os.path.exists(fname):
        with st.spinner(f"Downloading model weights... (~102MB)"):
            hf_hub_download(
                repo_id="hybridseg-demo/hybridseg-weights",
                filename=fname,
                local_dir="."
            )

@st.cache_resource
def load_model(dataset_name):
    cfg = MODEL_CONFIGS[dataset_name]
    ensure_weights(cfg["weights"])
    model = HybridSegModel(num_classes=1, d_state=8, ssm_warmup=5, in_channels=cfg["in_channels"])
    state = torch.load(cfg["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    gc.collect()
    return model


# ── Cleaning (Notebook 1 logic, mask-free version for inference) ──────────────
def clean_image(gray: np.ndarray) -> np.ndarray:
    """
    Replicates Notebook-1 cleaning steps that are mask-independent:
      - Grayscale is assumed (caller passes uint8 H×W array)
      - Corner text/marker masking  (Task 2, Notebook 1 & 2)
      - Duplicate/hash checks are dataset-level only → skipped at inference
    """
    img = gray.copy()

    # Corner masking — exact same fractions as notebook (h//8, w//8)
    h, w = img.shape
    ch, cw = h // 8, w // 8
    img[:ch,   :cw]  = 0
    img[:ch,  w-cw:] = 0
    img[h-ch:, :cw]  = 0
    img[h-ch:,w-cw:] = 0

    return img


# ── Preprocessing (Notebook 2, Tasks 1-6, mask-free) ─────────────────────────
def preprocess_image(gray: np.ndarray, img_size: int) -> np.ndarray:
    """
    Replicates Notebook-2 preprocessing steps exactly:
      Task 1  – already grayscale (caller guaranteed)
      Task 2  – corner masking (done in clean_image)
      Task 3  – ROI crop & resize
                  • With mask  → lesion-centered crop (training)
                  • Without mask (inference) → largest bright-region crop
                    using Otsu threshold so the field-of-view is still
                    lesion-biased rather than a plain center-crop.
      Task 4  – CLAHE with adaptive clip_limit
      Task 5  – NLM denoising
      Task 6  – Instance z-score normalization (±3σ clip → [0,255])
    Returns uint8 H×W array ready for /255 → tensor.
    """
    img = gray.copy()

    # ── Task 3: ROI crop & resize (mask-free fallback) ──────────────────────
    # Otsu threshold to find the brightest (tissue) region, then take its
    # bounding-box as the crop window — closest proxy to lesion-centered crop
    # without an actual mask.
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape
    if contours:
        all_pts  = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_pts)

        # Fixed 20% padding — midpoint of training's random 10-35% range
        pad_x = int(w * 0.20)
        pad_y = int(h * 0.20)
        x1 = max(0, x - pad_x);      y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x);  y2 = min(H, y + h + pad_y)
        img = img[y1:y2, x1:x2]
    else:
        # Absolute fallback: center-square crop (same as plain resize for square inputs)
        short = min(H, W)
        y1 = (H - short) // 2
        x1 = (W - short) // 2
        img = img[y1:y1+short, x1:x1+short]

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # ── Task 4: CLAHE with adaptive clip_limit ───────────────────────────────
    dr   = int(img.max()) - int(img.min())
    clip = 3.0 if dr < 150 else 2.0
    img  = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)

    # ── Task 5: NLM denoising ────────────────────────────────────────────────
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # ── Task 6: Instance z-score normalization (±3σ clip → [0, 255]) ────────
    g     = img.astype(np.float32)
    mu    = g.mean()
    sigma = max(g.std(), 1e-6)
    g     = (g - mu) / sigma
    g     = np.clip(g, -3.0, 3.0)
    g     = ((g + 3.0) / 6.0 * 255.0).astype(np.uint8)

    return g


def prepare_tensor(pil_img: Image.Image, img_size: int) -> torch.Tensor:
    """Full pipeline: PIL → cleaned → preprocessed → tensor (1,1,H,W)"""
    # Task 1: Grayscale
    gray = np.array(pil_img.convert("L"))

    # Notebook-1 cleaning (corner mask)
    gray = clean_image(gray)

    # Notebook-2 preprocessing (ROI crop, CLAHE, NLM, z-score)
    gray = preprocess_image(gray, img_size)

    # Task 7: /255 → float tensor
    arr = gray.astype(np.float32) / 255.0
    return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("⚕️ HybridSegModel")
st.markdown("**ResNet34 + VMamba SSM Bridge + UNet3+** | 25.53M params | Trained from scratch")
dataset_name = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()))
uploaded = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded and st.button("▶ Run Segmentation"):
    pil_img = Image.open(uploaded)
    cfg = MODEL_CONFIGS[dataset_name]
    with st.spinner("Running segmentation..."):
        model  = load_model(dataset_name)
        tensor = prepare_tensor(pil_img, cfg["img_size"]).to(DEVICE)
        with torch.no_grad():
            seg, _ = model(tensor, epoch=999)
            prob   = torch.sigmoid(seg)[0, 0].cpu().numpy()
        del tensor, seg
        gc.collect()

        orig    = np.array(pil_img.convert("RGB").resize((cfg["img_size"], cfg["img_size"])))
        overlay = orig.copy()
        overlay[prob > 0.5] = [255, 50, 50]
        blended = cv2.addWeighted(orig, 0.55, overlay, 0.45, 0)
        contours, _ = cv2.findContours(
            (prob > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, contours, -1, (0, 255, 150), 2)
        mask = Image.fromarray((prob * 255).astype(np.uint8))

    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.fromarray(blended), caption="Segmentation Overlay")
    with col2:
        st.image(mask, caption="Binary Mask")
    st.success(f"Done! Confidence: {float(prob.mean()):.3f}")