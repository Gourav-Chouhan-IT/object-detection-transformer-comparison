import tkinter as tk
from tkinter import filedialog
import torch
import torchvision
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import torchvision.transforms.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIDENCE_THRESHOLD = 0.05

# VOC 2012 Classes (Index 0 is background)
VOC_CLASSES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ==========================================
# 1. LOAD THE PYTORCH MODEL
# ==========================================
print("Initializing PyTorch Model...")

model = torchvision.models.detection.ssd300_vgg16(num_classes=21)
model.load_state_dict(torch.load("best.pth", map_location=DEVICE))

model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# ==========================================
# 2. INFERENCE & DRAWING ENGINE
# ==========================================
def draw_boxes(image_bgr, predictions):
    """Draws boxes and labels on an OpenCV BGR image"""
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = map(int, box)
            class_name = VOC_CLASSES[label]
            
            # Draw Rectangle
            cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw Label Background and Text
            text = f"{class_name}: {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_bgr, (xmin, ymin - text_h - 4), (xmin + text_w, ymin), (0, 255, 0), -1)
            cv2.putText(image_bgr, text, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
    return image_bgr

def process_and_show(img_pil):
    """Converts PIL to Tensor, runs inference, and displays via CV2"""
    img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions = model(img_tensor)
        
    # Convert PIL to BGR for OpenCV Drawing
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    result_img = draw_boxes(img_bgr, predictions)
    
    cv2.imshow("PyTorch Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==========================================
# 3. MENU FUNCTIONS
# ==========================================
def choose_local_file():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        print(f"Detecting objects in {file_path}...")
        img_pil = Image.open(file_path).convert("RGB")
        process_and_show(img_pil)

def use_url():
    url = input("Paste the image URL here: ")
    try:
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content)).convert("RGB")
        print("Detecting objects...")
        process_and_show(img_pil)
    except Exception as e:
        print(f"Error loading image from URL: {e}")

def use_webcam():
    print("\n" + "="*40)
    print("🚀 STARTING LIVE WEBCAM...")
    print("👉 Press 'q' to stop!")
    print("="*40)
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Convert BGR frame to Tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(img_tensor)
            
        annotated_frame = draw_boxes(frame, predictions)
        cv2.imshow("PyTorch Live Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# INTERACTIVE TERMINAL MENU
# ==========================================
while True:
    print("\n" + "="*30)
    print("   PYTORCH LIVE DEMO MENU")
    print("="*30)
    print("1. Select image from PC")
    print("2. Paste an image URL")
    print("3. Live Webcam Detection")
    print("4. Exit")
    print("="*30)
    
    choice = input("Type 1, 2, 3, or 4 and press Enter: ")
    
    if choice == '1': choose_local_file()
    elif choice == '2': use_url()
    elif choice == '3': use_webcam()
    elif choice == '4': 
        print("Exiting...")
        break
    else: print("Invalid choice. Try again.")