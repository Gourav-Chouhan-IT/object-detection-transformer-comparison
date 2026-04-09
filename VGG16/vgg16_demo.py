import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import requests
import numpy as np
import tkinter as tk
from tkinter import filedialog
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

import json

results = {}
for image_path, true_label in test_samples:  # your test images
    probs = model(image)  # softmax output
    pred_class = classes[probs.argmax()]
    confidence = probs.max().item() * 100
    results[true_label] = {
        "predicted": pred_class,
        "confidence": round(confidence, 2)
    }

with open("vgg16_per_class_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
print("Loading Class Names...")
try:
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
except FileNotFoundError:
    print("❌ ERROR: class_names.json not found in this folder!")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading VGG16 model to {device}...")

# 1A. Initialize VGG16 Architecture
# We don't need pretrained weights here because we are loading our own!
model = models.vgg16(weights=None)
# Modify the final layer to match our 25 classes
model.classifier[6] = nn.Linear(4096, len(class_names))

# 1B. Load your custom trained VGG16 weights
try:
    checkpoint = torch.load("vgg16_best.pth", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Strip 'module.' if it was trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', 'unknown')
    if isinstance(val_acc, float):
        print(f"✅ VGG16 Model loaded successfully! (Trained to {val_acc:.2f}% accuracy)")
    else:
        print(f"✅ VGG16 Model loaded successfully!")
        
except FileNotFoundError:
    print("❌ ERROR: vgg16_best.pth not found in this folder!")
    exit()

# 1C. Define VGG16 Image Transforms
# This replicates the preprocessing used during your Kaggle training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. INFERENCE FUNCTIONS
# ==========================================
def predict_image(pil_img):
    """Runs the image through VGG16 and returns the label and confidence"""
    # Apply transforms and add batch dimension [1, 3, 224, 224]
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top prediction
        confidence, class_idx = torch.max(probabilities, dim=0)
        predicted_class = class_names[class_idx.item()]
        
    return predicted_class, confidence.item() * 100

def show_static_result(pil_img, label, confidence):
    """Pops up a Matplotlib window showing the leaf and the prediction"""
    plt.figure(figsize=(8, 6))
    plt.imshow(pil_img)
    plt.axis('off')
    
    # Color text red for disease, green for healthy
    text_color = 'green' if 'healthy' in label.lower() else 'red'
    
    plt.title(f"VGG16 Prediction: {label}\nConfidence: {confidence:.1f}%", 
              fontsize=14, color=text_color, fontweight='bold')
    
    # Brings window to front
    mng = plt.get_current_fig_manager()
    try:
        mng.window.attributes('-topmost', 1) # Works on Windows
    except:
        pass
        
    plt.show()

# ==========================================
# 3. MENU ACTIONS
# ==========================================
def choose_local_file():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    
    print("\nOpening file explorer...")
    file_path = filedialog.askopenfilename(
        title="Select a Leaf Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        print(f"Analyzing {os.path.basename(file_path)}...")
        pil_img = Image.open(file_path).convert("RGB")
        label, conf = predict_image(pil_img)
        show_static_result(pil_img, label, conf)
    else:
        print("No file selected.")

def use_url():
    url = input("\nPaste the leaf image URL here and press Enter: ")
    try:
        print("Downloading image...")
        response = requests.get(url, timeout=5)
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        
        print("Analyzing...")
        label, conf = predict_image(pil_img)
        show_static_result(pil_img, label, conf)
    except Exception as e:
        print(f"Error loading image from URL: {e}")

def use_webcam():
    print("\n" + "="*40)
    print("🚀 STARTING LIVE VGG16 WEBCAM...")
    print("👉 IMPORTANT: Click on the video window and press 'q' to stop!")
    print("="*40)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert OpenCV's BGR format to RGB for PIL
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_converted)
        
        label, conf = predict_image(pil_img)
        
        # Draw background rectangle for text readability
        cv2.rectangle(frame, (10, 10), (600, 80), (0, 0, 0), -1)
        
        color = (0, 255, 0) if 'healthy' in label.lower() else (0, 0, 255)
        
        cv2.putText(frame, f"VGG16: {label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {conf:.1f}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("VGG16 Live Diagnosis", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 4. INTERACTIVE TERMINAL MENU
# ==========================================
if __name__ == "__main__":
    while True:
        print("\n" + "="*35)
        print(" 🍃 VGG16 DISEASE DEMO MENU")
        print("="*35)
        print("1. Select image from PC (Pop-up)")
        print("2. Paste an image URL")
        print("3. Live Webcam Diagnosis")
        print("4. Exit")
        print("="*35)
        
        choice = input("Type 1, 2, 3, or 4 and press Enter: ")
        
        if choice == '1':
            choose_local_file()
        elif choice == '2':
            use_url()
        elif choice == '3':
            use_webcam()
        elif choice == '4':
            print("Closing demo...")
            break
        else:
            print("Invalid choice. Please type 1, 2, 3, or 4.")