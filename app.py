
import os
import gc
import numpy as np
import cv2
import torch
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
from model import HybridSegModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "🩻 Breast Ultrasound (BUSI)": {
        "weights": "busi_best_weights.pth",
        "in_channels": 1,
        "img_size": 384,
    }
}

DEVICE = torch.device("cpu")
torch.set_num_threads(1)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(dataset_name):

    cfg = MODEL_CONFIGS[dataset_name]

    if not os.path.exists(cfg["weights"]):
        hf_hub_download(
            repo_id="hybridseg-demo/hybridseg-weights",
            filename=cfg["weights"],
            local_dir="."
        )

    model = HybridSegModel(
        num_classes=1,
        d_state=8,
        ssm_warmup=5,
        in_channels=cfg["in_channels"]
    )

    state = torch.load(cfg["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    gc.collect()
    return model


# ─────────────────────────────────────────────
# CONTRAST BOOST
# ─────────────────────────────────────────────
def enhance(img):

    img = img.astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)

    gamma = 1.3
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)]).astype(np.uint8)

    return cv2.LUT(img, lut)


# ─────────────────────────────────────────────
# ROI DETECTION (TEXTURE BASED)
# ─────────────────────────────────────────────
def get_roi(img):

    H, W = img.shape

    lap = cv2.Laplacian(img, cv2.CV_64F)
    score = np.abs(lap).astype(np.uint8)

    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    pad_x = int(w * 0.25)
    pad_y = int(h * 0.25)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return img[y1:y2, x1:x2]


# ─────────────────────────────────────────────
# PREPROCESS PIPELINE
# ─────────────────────────────────────────────
def preprocess(img, size):

    img = enhance(img)
    img = get_roi(img)

    img = cv2.resize(img, (size, size))

    # Speckle reduction
    img = cv2.bilateralFilter(img, 7, 50, 50)

    # CLAHE
    dr = img.max() - img.min()
    clip = 3.0 if dr < 150 else 2.0

    img = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)

    return img


# ─────────────────────────────────────────────
# MULTI-SCALE INFERENCE
# ─────────────────────────────────────────────
def infer_multiscale(model, img, device):

    scales = [256, 384, 512]

    probs = []

    for s in scales:

        im = cv2.resize(img, (s, s))

        t = torch.FloatTensor(im / 255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            seg, _ = model(t, epoch=999)
            p = torch.sigmoid(seg)[0, 0].cpu().numpy()

        p = cv2.resize(p, (img.shape[1], img.shape[0]))

        probs.append(p)

    return np.mean(probs, axis=0)


# ─────────────────────────────────────────────
# POST PROCESS
# ─────────────────────────────────────────────
def postprocess(prob):

    kernel = np.ones((5, 5), np.uint8)

    mask = (prob > 0.5).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("⚕️ HybridSeg Full Fixed Pipeline")

dataset_name = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()))
uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if uploaded and st.button("▶ Run"):

    pil = Image.open(uploaded)

    cfg = MODEL_CONFIGS[dataset_name]

    model = load_model(dataset_name)

    # grayscale
    img = np.array(pil.convert("L"))

    # preprocess
    img = preprocess(img, cfg["img_size"])

    # inference
    prob = infer_multiscale(model, img, DEVICE)

    mask = postprocess(prob)

    # confidence FIXED
    confidence = prob[mask == 1].mean() if mask.sum() > 0 else prob.max()

    # visualization
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    overlay = rgb.copy()
    overlay[mask == 1] = [255, 50, 50]

    blended = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)

    # OUTPUT
    col1, col2 = st.columns(2)

    with col1:
        st.image(blended, caption="Prediction Overlay")

    with col2:
        st.image(mask * 255, caption="Mask")

    st.image(img, caption="Preprocessed Input")

    st.image((prob * 255).astype(np.uint8), caption="Probability Map")

    st.success(f"Confidence: {confidence:.3f}")
