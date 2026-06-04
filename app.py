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

def preprocess(pil_img, in_channels, img_size):
    # Step 1: Grayscale
    img = np.array(pil_img.convert("L"))

    # Step 2: Corner masking
    h, w = img.shape
    ch, cw = h // 8, w // 8
    img[:ch, :cw] = 0
    img[:ch, w-cw:] = 0
    img[h-ch:, :cw] = 0
    img[h-ch:, w-cw:] = 0

    # Step 3: ROI crop & resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Step 4: CLAHE
    dr = int(img.max()) - int(img.min())
    clip = 3.0 if dr < 150 else 2.0
    img = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)

    # Step 5: NLM denoising
    img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Step 6: Instance z-score normalization
    g = img.astype(np.float32)
    mu = g.mean()
    sigma = max(g.std(), 1e-6)
    g = (g - mu) / sigma
    g = np.clip(g, -3.0, 3.0)
    g = ((g + 3.0) / 6.0 * 255.0).astype(np.uint8)

    # Step 7: To tensor
    arr = g.astype(np.float32) / 255.0
    return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)

st.title("⚕️ HybridSegModel")
st.markdown("**ResNet34 + VMamba SSM Bridge + UNet3+** | 25.53M params | Trained from scratch")
dataset_name = st.selectbox("Select Model", list(MODEL_CONFIGS.keys()))
uploaded = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg"])

if uploaded and st.button("▶ Run Segmentation"):
    pil_img = Image.open(uploaded)
    cfg = MODEL_CONFIGS[dataset_name]
    with st.spinner("Running segmentation..."):
        model = load_model(dataset_name)
        tensor = preprocess(pil_img, cfg["in_channels"], cfg["img_size"]).to(DEVICE)
        with torch.no_grad():
            seg, _ = model(tensor, epoch=999)
            prob = torch.sigmoid(seg)[0, 0].cpu().numpy()
        del tensor, seg
        gc.collect()
        orig = np.array(pil_img.convert("RGB").resize((cfg["img_size"], cfg["img_size"])))
        overlay = orig.copy()
        overlay[prob > 0.5] = [255, 50, 50]
        blended = cv2.addWeighted(orig, 0.55, overlay, 0.45, 0)
        contours, _ = cv2.findContours((prob > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (0, 255, 150), 2)
        mask = Image.fromarray((prob * 255).astype(np.uint8))
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.fromarray(blended), caption="Segmentation Overlay")
    with col2:
        st.image(mask, caption="Binary Mask")
    st.success(f"Done! Confidence: {float(prob.mean()):.3f}")