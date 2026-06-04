from huggingface_hub import hf_hub_download
import os
import gc

def download_weights():
    repo = "hybridseg-demo/hybridseg-weights"
    for fname in ["chest_best_weights.pth", "polyp_best_weights.pth", "busi_best_weights.pth"]:
        if not os.path.exists(fname):
            hf_hub_download(repo_id=repo, filename=fname, local_dir=".")

download_weights()

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from model import HybridSegModel, MODEL_CONFIGS

torch.set_num_threads(1)
DEVICE = torch.device("cpu")

@st.cache_resource
def load_model(dataset_name):
    cfg = MODEL_CONFIGS[dataset_name]
    model = HybridSegModel(num_classes=1, d_state=8, ssm_warmup=5, in_channels=cfg["in_channels"])
    state = torch.load(cfg["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    gc.collect()
    return model

def preprocess(pil_img, in_channels, img_size):
    if in_channels == 1:
        img = pil_img.convert("L").resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)
    img = pil_img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.FloatTensor(arr).permute(2, 0, 1).unsqueeze(0)

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