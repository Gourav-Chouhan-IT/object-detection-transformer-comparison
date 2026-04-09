import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO

# 1. Load the model once when the script starts
print("Loading YOLOv8s model...")
model = YOLO("best.pt")

def display_results(results):
    """Helper function to draw boxes and open the image"""
    for r in results:
        im_array = r.plot()  # Draws the boxes
        im = Image.fromarray(im_array[..., ::-1])  # Converts BGR colors to RGB
        im.show()  # Opens your computer's default photo viewer

def choose_local_file():
    """Summons the Windows pop-up window to select a file"""
    root = tk.Tk()
    root.attributes('-topmost', True) # Brings the window to the front
    root.withdraw() # Hides the blank background window
    
    print("\nOpening file explorer...")
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        print(f"Detecting objects in {file_path}...")
        results = model.predict(source=file_path, conf=0.50)
        display_results(results)
    else:
        print("No file selected.")

def use_url():
    """Allows you to paste a web URL"""
    url = input("\nPaste the image URL here and press Enter: ")
    try:
        print("Downloading image...")
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        print("Detecting objects...")
        results = model.predict(source=img, conf=0.50)
        display_results(results)
    except Exception as e:
        print(f"Error loading image from URL: {e}. Make sure the link goes directly to an image (.jpg/.png).")

def use_webcam():
    """Opens the laptop webcam for live real-time detection"""
    print("\n" + "="*40)
    print("🚀 STARTING LIVE WEBCAM...")
    print("👉 IMPORTANT: Click on the video window and press 'q' to stop!")
    print("="*40)
    
    # source=0 tells YOLO to use the default laptop camera
    # show=True tells it to instantly pop up the live video window
    model.predict(source=0, show=True, conf=0.50)

# ==========================================
# INTERACTIVE TERMINAL MENU
# ==========================================
while True:
    print("\n" + "="*30)
    print("   YOLOv8 LIVE DEMO MENU")
    print("="*30)
    print("1. Select image from PC (Pop-up)")
    print("2. Paste an image URL")
    print("3. Live Webcam Detection")
    print("4. Exit")
    print("="*30)
    
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