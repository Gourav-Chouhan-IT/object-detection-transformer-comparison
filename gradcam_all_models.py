"""
gradcam_all_models.py
=====================
Generates Grad-CAM heatmaps for every leaf image in your Sample leaves folder,
for all 3 models (VGG16, ViT-Base, Swin-Base).

Output structure:
    gradcam_output/
        Apple_leaf/
            vgg16_gradcam.jpg
            vit_gradcam.jpg
            swin_gradcam.jpg
        Apple_rust_leaf/
            ...
        ...
    gradcam_output/
        combined_gradcam_poster.png   ← one big summary poster

Usage (run from your Project Exhibition 2 folder, or adjust paths):
    python gradcam_all_models.py

Requirements:
    pip install torch torchvision transformers opencv-python matplotlib pillow numpy
    pip install grad-cam          <-- pytorch-grad-cam library
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import OrderedDict

# pytorch-grad-cam
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# torchvision (VGG16)
from torchvision import models, transforms

# HuggingFace (ViT, Swin)
from transformers import (
    ViTForImageClassification, ViTImageProcessor,
    AutoModelForImageClassification, AutoImageProcessor
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these paths if needed
# ══════════════════════════════════════════════════════════════════════════════
VGG16_WEIGHTS  = r"D:\Project Exhibition 2\VGG16\vgg16_best.pth"
VIT_WEIGHTS    = r"D:\Project Exhibition 2\vision transformer\2 data set\vit_best.pth"
SWIN_WEIGHTS   = r"D:\Project Exhibition 2\Swin-Base Transformer\swin_best.pth"

VGG16_CLASSES  = r"D:\Project Exhibition 2\VGG16\class_names.json"
VIT_CLASSES    = r"D:\Project Exhibition 2\vision transformer\2 data set\class_names.json"
SWIN_CLASSES   = r"D:\Project Exhibition 2\Swin-Base Transformer\class_names.json"

IMAGE_FOLDER   = r"D:\Project Exhibition 2\Sample leaves"
OUTPUT_FOLDER  = r"D:\Project Exhibition 2\gradcam_output"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_weights(model, path):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(new_sd)
    model.to(device).eval()
    return model

def load_classes(path):
    with open(path) as f:
        return json.load(f)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(folder):
    imgs = {}
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS or ext == "":   # Bell_pepper_leaf has no ext
            label = os.path.splitext(fname)[0].replace("_", " ")
            imgs[label] = os.path.join(folder, fname)
    return imgs

def rgb_float(path, size=224):
    """Returns (H,W,3) float32 in [0,1] for overlay, and PIL image."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr, pil

def save_cam_figure(pil_img, cam, label, pred, conf, correct, out_path):
    """Saves a 2-panel figure: original | Grad-CAM overlay."""
    rgb = np.array(pil_img.resize((224, 224)), dtype=np.float32) / 255.0
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    fig.patch.set_facecolor("#0D1117")

    axes[0].imshow(pil_img.resize((224, 224)))
    axes[0].axis("off")
    axes[0].set_title("Input", color="white", fontsize=8)

    axes[1].imshow(overlay)
    axes[1].axis("off")
    color = "#2EA043" if correct else "#F85149"
    mark  = "✓" if correct else "✗"
    axes[1].set_title(f"{mark} {pred}\n{conf:.1f}%", color=color,
                      fontsize=8, fontweight="bold")

    plt.suptitle(label, color="#58A6FF", fontsize=9,
                 fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor="#0D1117", edgecolor="none")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── VGG16 ─────────────────────────────────────────────────────────────────────
class VGG16Wrapper:
    def __init__(self):
        self.classes = load_classes(VGG16_CLASSES)
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, len(self.classes))
        self.model = load_weights(m, VGG16_WEIGHTS)
        self.preprocess = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # Last conv layer for Grad-CAM
        self.target_layers = [self.model.features[-1]]
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)

    def predict(self, pil_img):
        t = self.preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.model(t)
            probs = torch.softmax(out, 1)[0]
            conf, idx = torch.max(probs, 0)
        return self.classes[idx.item()], conf.item() * 100, idx.item()

    def get_cam(self, pil_img, class_idx):
        t = self.preprocess(pil_img).unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(class_idx)]
        cam_map = self.cam(input_tensor=t, targets=targets)
        return cam_map[0]   # (H, W) in [0,1]


# ── ViT Wrapper ───────────────────────────────────────────────────────────────
# ViT needs special handling: no conv layers, use attention rollout or
# hook into the last LayerNorm as a proxy target for GradCAM
class ViTWrapper:
    def __init__(self):
        self.classes = load_classes(VIT_CLASSES)
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        m = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(self.classes),
            id2label={str(i): c for i, c in enumerate(self.classes)},
            label2id={c: str(i) for i, c in enumerate(self.classes)},
            ignore_mismatched_sizes=True
        )
        self.model = load_weights(m, VIT_WEIGHTS)
        # Target the last encoder block's LayerNorm for Grad-CAM
        self.target_layers = [self.model.vit.encoder.layer[-1].layernorm_before]
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers,
                                   reshape_transform=self._reshape)

    def _reshape(self, tensor):
        # ViT outputs (batch, num_patches+1, hidden) — drop CLS, reshape to spatial
        result = tensor[:, 1:, :]   # remove CLS token
        num_patches = int(result.shape[1] ** 0.5)
        result = result.reshape(result.size(0), num_patches, num_patches, result.size(2))
        result = result.permute(0, 3, 1, 2)
        return result

    def predict(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.logits, 1)[0]
            conf, idx = torch.max(probs, 0)
        return self.classes[idx.item()], conf.item() * 100, idx.item()

    def get_cam(self, pil_img, class_idx):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(device)

        class HFInputWrapper(torch.nn.Module):
            def __init__(self, hf_model):
                super().__init__()
                self.hf_model = hf_model
            def forward(self, pixel_values):
                return self.hf_model(pixel_values=pixel_values).logits

        wrapper = HFInputWrapper(self.model)
        wrapper_cam = GradCAMPlusPlus(model=wrapper,
                                      target_layers=[wrapper.hf_model.vit.encoder.layer[-1].layernorm_before],
                                      reshape_transform=self._reshape)
        targets = [ClassifierOutputTarget(class_idx)]
        cam_map = wrapper_cam(input_tensor=inputs["pixel_values"], targets=targets)
        return cam_map[0]


# ── Swin Wrapper ──────────────────────────────────────────────────────────────
class SwinWrapper:
    def __init__(self):
        self.classes = load_classes(SWIN_CLASSES)
        self.processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
        m = AutoModelForImageClassification.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            num_labels=len(self.classes),
            id2label={str(i): c for i, c in enumerate(self.classes)},
            label2id={c: str(i) for i, c in enumerate(self.classes)},
            ignore_mismatched_sizes=True
        )
        self.model = load_weights(m, SWIN_WEIGHTS)
        # Swin: hook into last stage's norm layer
        self.target_layers = [self.model.swin.layernorm]

    def _reshape(self, tensor):
        # Swin outputs (batch, H*W, C) — reshape to spatial
        b, n, c = tensor.shape
        h = w = int(n ** 0.5)
        return tensor.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def predict(self, pil_img):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out.logits, 1)[0]
            conf, idx = torch.max(probs, 0)
        return self.classes[idx.item()], conf.item() * 100, idx.item()

    def get_cam(self, pil_img, class_idx):
        inputs = self.processor(images=pil_img, return_tensors="pt").to(device)

        class HFSwinWrapper(torch.nn.Module):
            def __init__(self, hf_model):
                super().__init__()
                self.hf_model = hf_model
            def forward(self, pixel_values):
                return self.hf_model(pixel_values=pixel_values).logits

        wrapper = HFSwinWrapper(self.model)
        wrapper_cam = GradCAMPlusPlus(
            model=wrapper,
            target_layers=[wrapper.hf_model.swin.layernorm],
            reshape_transform=self._reshape
        )
        targets = [ClassifierOutputTarget(class_idx)]
        cam_map = wrapper_cam(input_tensor=inputs["pixel_values"], targets=targets)
        return cam_map[0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nLoading models...")
    vgg16 = VGG16Wrapper();  print("✅ VGG16 ready")
    vit   = ViTWrapper();    print("✅ ViT ready")
    swin  = SwinWrapper();   print("✅ Swin ready")

    images = list_images(IMAGE_FOLDER)
    print(f"\nFound {len(images)} leaf images\n")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    all_results = []   # collect for poster

    for label, img_path in images.items():
        print(f"Processing: {label}")
        pil_img, rgb_arr = None, None
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠️  Could not open {img_path}: {e}")
            continue

        out_dir = os.path.join(OUTPUT_FOLDER, label.replace(" ", "_"))
        os.makedirs(out_dir, exist_ok=True)

        row = {"label": label, "img_path": img_path, "models": {}}

        for model_name, model_obj in [("VGG16", vgg16), ("ViT-Base", vit), ("Swin-Base", swin)]:
            try:
                pred, conf, class_idx = model_obj.predict(pil_img)
                correct = pred.lower().replace("_"," ") == label.lower()
                cam_map = model_obj.get_cam(pil_img, class_idx)

                out_path = os.path.join(out_dir, f"{model_name.lower().replace('-','_')}_gradcam.jpg")
                save_cam_figure(pil_img, cam_map, label, pred, conf, correct, out_path)

                row["models"][model_name] = {
                    "pred": pred, "conf": conf,
                    "correct": correct, "cam": cam_map,
                    "cam_path": out_path
                }
                mark = "✓" if correct else "✗"
                print(f"  [{model_name}] {mark} {pred} ({conf:.1f}%)")
            except Exception as e:
                print(f"  ⚠️  {model_name} failed: {e}")

        all_results.append(row)

    # ── Combined poster ────────────────────────────────────────────────────────
    print("\nGenerating combined poster...")
    N = len(all_results)
    COLS = 3   # one column per model
    ROWS = N

    BG = "#0D1117"
    PANEL = "#161B22"
    MODEL_COLORS = {"VGG16": "#2EA043", "ViT-Base": "#1F6FEB", "Swin-Base": "#DB6D28"}

    fig_w = 18
    fig_h = N * 2.8 + 1.5
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    fig.text(0.5, 1 - 0.3/fig_h,
             "CROP LEAF DISEASE DETECTION — GRAD-CAM COMPARISON",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="white", fontfamily="monospace")
    fig.text(0.5, 1 - 0.65/fig_h,
             "VGG16  |  ViT-Base-Patch16  |  Swin-Base-Patch4     Input  +  Grad-CAM Overlay",
             ha="center", va="top", fontsize=9, color="#8B949E", fontfamily="monospace")

    # column headers
    for ci, (mname, mc) in enumerate(MODEL_COLORS.items()):
        fig.text(0.18 + ci * 0.27, 1 - 1.1/fig_h, mname,
                 ha="center", va="top", fontsize=11, color=mc,
                 fontweight="bold", fontfamily="monospace")

    TOP = 1 - 1.5/fig_h
    for ri, row in enumerate(all_results):
        y0 = TOP - ri * (2.8 / fig_h)
        row_h = 2.5 / fig_h

        # row label
        fig.text(0.01, y0 - row_h * 0.5, row["label"],
                 ha="left", va="center", fontsize=7, color="#58A6FF",
                 fontfamily="monospace", rotation=0)

        for ci, (mname, mc) in enumerate(MODEL_COLORS.items()):
            mdata = row["models"].get(mname)
            x0 = 0.07 + ci * 0.31
            col_w = 0.29

            if mdata and os.path.exists(mdata["cam_path"]):
                # load saved 2-panel image
                img = plt.imread(mdata["cam_path"])
                ax = fig.add_axes([x0, y0 - row_h, col_w, row_h])
                ax.imshow(img)
                ax.axis("off")

                correct = mdata["correct"]
                color = "#2EA043" if correct else "#F85149"
                mark = "✓" if correct else "✗"
                ax.set_title(f"{mark} {mdata['pred']}  {mdata['conf']:.1f}%",
                             fontsize=7, color=color, fontweight="bold",
                             pad=2)
            else:
                ax = fig.add_axes([x0, y0 - row_h, col_w, row_h])
                ax.set_facecolor(PANEL)
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="#8B949E", fontsize=10, transform=ax.transAxes)
                ax.axis("off")

    poster_path = os.path.join(OUTPUT_FOLDER, "combined_gradcam_poster.png")
    plt.savefig(poster_path, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"\n✅ Combined poster saved → {poster_path}")
    print(f"✅ Individual Grad-CAMs saved in subfolders under → {OUTPUT_FOLDER}")
    print("\nDone!")