from huggingface_hub import hf_hub_download
import os
import gc

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from model import HybridSegModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# DOWNLOAD WEIGHTS
# ─────────────────────────────────────────────

def ensure_weights(fname):
    if not os.path.exists(fname):
        with st.spinner("Downloading model weights..."):
            hf_hub_download(
                repo_id="hybridseg-demo/hybridseg-weights",
                filename=fname,
                local_dir="."
            )

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model(dataset_name):
    cfg = MODEL_CONFIGS[dataset_name]
    ensure_weights(cfg["weights"])
    model = HybridSegModel(
        num_classes=1,
        d_state=8,
        ssm_warmup=5,
        in_channels=cfg["in_channels"]
    )
    state = torch.load(
        cfg["weights"],
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    gc.collect()
    return model

# ─────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────

def clean_image(gray):
    img = gray.copy()
    h, w = img.shape
    ch = h // 8
    cw = w // 8
    img[:ch,  :cw]  = 0
    img[:ch,  w-cw:] = 0
    img[h-ch:, :cw]  = 0
    img[h-ch:, w-cw:] = 0
    return img

# ─────────────────────────────────────────────
# [NEW] GAMMA CORRECTION
# উদ্দেশ্য: dark image এ lesion আরও visible করা।
# training এ ছিল না, কিন্তু inference এ এটা
# CLAHE এর আগে image brightness normalize করে
# যাতে CLAHE আরও ভালো কাজ করতে পারে।
# ─────────────────────────────────────────────

def apply_gamma(gray, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype(np.uint8)
    return cv2.LUT(gray, table)

# ─────────────────────────────────────────────
# [IMPROVED] MULTI-CROP PREPROCESS
# পুরনো: 3 fixed padding + 1 center crop
# নতুন:  top-3 contour merge → 3 ROI crops
#         + 1 center crop + 1 full image crop
#         = মোট 5টা crop, আরও বেশি coverage
# ─────────────────────────────────────────────

def preprocess_image(gray, img_size):
    img = gray.copy()
    H, W = img.shape

    # ── [NEW] Gamma correction ──────────────────
    # Dark region এ lesion থাকলে CLAHE এর আগে
    # gamma দিয়ে brighten করলে threshold আরও stable হয়
    img = apply_gamma(img, gamma=1.2)

    # ── Blur for stable threshold ───────────────
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # ── OTSU threshold ──────────────────────────
    _, otsu = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crops = []

    # ── [IMPROVED] Top-3 contour merge ROI ─────
    # পুরনো কোড: শুধু largest contour নিত।
    # সমস্যা: ছোট lesion (secondary nodule) miss হতো।
    # নতুন: top-3 largest contour একসাথে merge করে
    # একটা combined bounding box বানানো হয়।
    if len(contours) > 0:
        sorted_contours = sorted(
            contours, key=cv2.contourArea, reverse=True
        )
        top_contours = sorted_contours[:3]
        all_points = np.vstack(top_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        for pad_ratio in [0.10, 0.20, 0.30]:
            pad_x = int(w * pad_ratio)
            pad_y = int(h * pad_ratio)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    # ── Center crop (fallback) ──────────────────
    short = min(H, W)
    cy = (H - short) // 2
    cx = (W - short) // 2
    crops.append(img[cy:cy+short, cx:cx+short])

    # ── [NEW] Full image crop ───────────────────
    # কখনো কখনো lesion image এর corner বা boundary তে
    # থাকলে ROI crop মিস করে। Full image দিলে সেটা cover হয়।
    crops.append(img.copy())

    # ── Process each crop ───────────────────────
    processed = []
    for crop in crops:

        crop = cv2.resize(
            crop, (img_size, img_size),
            interpolation=cv2.INTER_AREA
        )

        # ── [NEW] Bilateral Filter ──────────────
        # NLM এর আগে bilateral filter দিলে edge
        # preserve করে speckle noise কমে।
        # Ultrasound image এর জন্য NLM এর চেয়ে
        # edge-aware, তাই lesion boundary sharper থাকে।
        crop = cv2.bilateralFilter(
            crop,
            d=9,
            sigmaColor=50,
            sigmaSpace=50
        )

        # ── CLAHE ──────────────────────────────
        dr = int(crop.max()) - int(crop.min())
        clip = 3.0 if dr < 150 else 2.0
        crop = cv2.createCLAHE(
            clipLimit=clip, tileGridSize=(8, 8)
        ).apply(crop)

        # ── NLM Denoising ───────────────────────
        crop = cv2.fastNlMeansDenoising(
            crop, None,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # ── Z-score normalization ───────────────
        g = crop.astype(np.float32)
        mu = g.mean()
        sigma = max(g.std(), 1e-6)
        g = (g - mu) / sigma
        g = np.clip(g, -3.0, 3.0)
        g = ((g + 3.0) / 6.0 * 255.0).astype(np.uint8)

        processed.append(g)

    return processed

# ─────────────────────────────────────────────
# PREPARE MULTIPLE TENSORS
# ─────────────────────────────────────────────

def prepare_tensors(pil_img, img_size):
    gray = np.array(pil_img.convert("L"))
    gray = clean_image(gray)
    processed_imgs = preprocess_image(gray, img_size)

    tensors = []
    for img in processed_imgs:
        arr = img.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)
        tensors.append((tensor, img))

    return tensors

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("⚕️ HybridSegModel")
st.markdown("**ResNet34 + VMamba SSM Bridge + UNet3+**")

dataset_name = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()))
uploaded = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

if uploaded and st.button("▶ Run Segmentation"):
    pil_img = Image.open(uploaded)
    cfg = MODEL_CONFIGS[dataset_name]

    with st.spinner("Running segmentation..."):
        model = load_model(dataset_name)
        tensor_list = prepare_tensors(pil_img, cfg["img_size"])

        # ── [IMPROVED] Weighted ensemble ────────
        # পুরনো: শুধু best score এর mask নিত।
        # সমস্যা: একটা crop এ noise থাকলে ভুল mask আসতো।
        # নতুন: সব crop এর prob map গুলো score² দিয়ে
        # weighted average করা হয়।
        # উপকার: ভালো crop এর weight বেশি পায়,
        # খারাপ crop এর effect কমে যায়।
        weighted_prob = None
        weight_sum = 0.0
        best_input = None
        best_score = -1

        for tensor, prep_img in tensor_list:
            tensor = tensor.to(DEVICE)

            with torch.no_grad():
                seg, _ = model(tensor, epoch=999)
                prob = torch.sigmoid(seg)[0, 0].cpu().numpy()

            score = float(prob.max())
            weight = score ** 2  # score² দিয়ে weight, ভালো crop এর importance বেশি

            if weighted_prob is None:
                weighted_prob = prob * weight
            else:
                weighted_prob += prob * weight

            weight_sum += weight

            if score > best_score:
                best_score = score
                best_input = prep_img

            del tensor, seg

        gc.collect()

        # Normalize weighted sum
        if weight_sum > 0:
            final_prob = weighted_prob / weight_sum
        else:
            final_prob = weighted_prob

        # ── Visualization ────────────────────────
        prep_rgb = cv2.cvtColor(best_input, cv2.COLOR_GRAY2RGB)
        overlay = prep_rgb.copy()
        overlay[final_prob > 0.5] = [255, 50, 50]
        blended = cv2.addWeighted(prep_rgb, 0.55, overlay, 0.45, 0)

        contours, _ = cv2.findContours(
            (final_prob > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(blended, contours, -1, (0, 255, 150), 2)

        mask = Image.fromarray((final_prob * 255).astype(np.uint8))

    # ── Show results ─────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.fromarray(blended), caption="Segmentation Overlay")
    with col2:
        st.image(mask, caption="Predicted Mask (Ensemble)")

    st.image(best_input, caption="Best Preprocessed Input")
    st.success(f"Done! Confidence: {best_score:.3f}")