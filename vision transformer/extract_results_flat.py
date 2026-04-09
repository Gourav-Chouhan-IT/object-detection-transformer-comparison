"""
extract_results_flat.py  —  Works for VGG16, Swin-Base, and ViT-Base
Uses HuggingFace transformers for Swin and ViT (matching your demo scripts exactly)

Usage:
    # From VGG16 folder:
    python extract_results_flat.py --model vgg16 --weights vgg16_best.pth --images "D:\Project Exhibition 2\Sample leaves"

    # From Swin-Base Transformer folder:
    python extract_results_flat.py --model swin --weights swin_best.pth --images "D:\Project Exhibition 2\Sample leaves"

    # From vision transformer folder:
    python extract_results_flat.py --model vit --weights vit_best.pth --images "D:\Project Exhibition 2\Sample leaves"
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model",   required=True, choices=["vgg16", "vit", "swin"])
parser.add_argument("--weights", required=True)
parser.add_argument("--images",  required=True)
parser.add_argument("--classes", default="class_names.json")
args = parser.parse_args()

# -- Class names
with open(args.classes, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)
print(f"Loaded {num_classes} classes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -- Build model matching your demo scripts exactly
if args.model == "vgg16":
    from torchvision import models, transforms

    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, num_classes)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def predict(img_path):
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        return class_names[idx.item()], round(conf.item() * 100, 2)

elif args.model == "swin":
    from transformers import AutoModelForImageClassification, AutoImageProcessor

    model_name = "microsoft/swin-base-patch4-window7-224"
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
        ignore_mismatched_sizes=True
    )
    processor = AutoImageProcessor.from_pretrained(model_name)

    def predict(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=0)
        return class_names[idx.item()], round(conf.item() * 100, 2)

elif args.model == "vit":
    from transformers import ViTForImageClassification, ViTImageProcessor

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
        ignore_mismatched_sizes=True
    )
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    def predict(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            conf, idx = torch.max(probs, dim=0)
        return class_names[idx.item()], round(conf.item() * 100, 2)

# -- Load weights
checkpoint = torch.load(args.weights, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()
print(f"[OK] {args.model.upper()} loaded from {args.weights}")

# -- Walk flat image folder (filename = class name)
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
results = {}

for fname in sorted(os.listdir(args.images)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in image_exts:
        continue

    true_label = os.path.splitext(fname)[0].replace("_", " ")
    img_path = os.path.join(args.images, fname)

    try:
        pred, conf = predict(img_path)
        results[true_label] = {"predicted": pred, "confidence": conf}
        status = "OK" if pred.lower() == true_label.lower() else "XX"
        print(f"  [{status}] [{true_label}] -> {pred} ({conf:.1f}%)")
    except Exception as e:
        print(f"  [!!] Skipped {fname}: {e}")

# -- Save
out_file = f"{args.model}_per_class_results.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Saved -> {out_file}  ({len(results)} classes)")
