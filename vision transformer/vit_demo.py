import os
import json
import torch
import cv2
import requests
import numpy as np
import tkinter as tk
from tkinter import filedialog
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
from collections import OrderedDict

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
print(f"Loading ViT-Base model to {device}...")

# Initialize the architecture (matches your training setup)
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=len(class_names),
    id2label={str(i): c for i, c in enumerate(class_names)},
    label2id={c: str(i) for i, c in enumerate(class_names)},
    ignore_mismatched_sizes=True
)

# Load your custom trained weights
try:
    checkpoint = torch.load("vit_best.pth", map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Strip 'module.' if it was trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully! (Trained to {checkpoint.get('val_acc', 'unknown')}% accuracy)")
except FileNotFoundError:
    print("❌ ERROR: vit_best.pth not found in this folder!")
    exit()

# The processor handles resizing and normalizing the images for the ViT
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# ==========================================
# 2. INFERENCE FUNCTIONS
# ==========================================
def predict_image(pil_img):
    """Runs the image through the ViT and returns the label and confidence"""
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get top prediction
        confidence, class_idx = torch.max(probabilities, dim=0)
        predicted_class = class_names[class_idx.item()]
        
    return predicted_class, confidence.item() * 100

def show_static_result(pil_img, label, confidence):
    """Pops up a Matplotlib window showing the leaf and the prediction"""
    plt.figure(figsize=(8, 6))
    plt.imshow(pil_img)
    plt.axis('off')
    
    # Color text red for disease, green for healthy (optional touch)
    text_color = 'green' if 'healthy' in label.lower() else 'red'
    
    plt.title(f"Prediction: {label}\nConfidence: {confidence:.1f}%", 
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
    print("🚀 STARTING LIVE WEBCAM...")
    print("👉 IMPORTANT: Click on the video window and press 'q' to stop!")
    print("👉 Hold a leaf (or phone screen showing a leaf) up to the camera.")
    print("="*40)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert OpenCV BGR frame to PIL RGB image for the model
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(color_converted)
        
        # Run inference
        label, conf = predict_image(pil_img)
        
        # Draw the background box for text readability
        cv2.rectangle(frame, (10, 10), (550, 80), (0, 0, 0), -1)
        
        # Determine color based on health
        color = (0, 255, 0) if 'healthy' in label.lower() else (0, 0, 255)
        
        # Write the prediction on the video frame
        cv2.putText(frame, f"{label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Conf: {conf:.1f}%", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("ViT Live Leaf Diagnosis", frame)
        
        # Press 'q' to quit
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
        print(" 🌱 ViT LEAF DISEASE DEMO MENU")
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