# 🌿 Crop Leaf Disease Detection — Model Comparison

Benchmarking **CNN vs Transformer** architectures for plant disease classification across 30 disease classes on a fused dataset of 56,857 images.

> **Models compared:** VGG16 · ViT-Base-Patch16-224 · Swin-Base-Patch4-Window7-224  
> **Dataset:** PlantDoc + PlantVillage (Combined) · 30 Disease Classes · 56,857 Images  
> **Explainability:** Grad-CAM visualizations for all three models

---

## 📊 Results Summary

| Model | Architecture | Parameters | Test Accuracy | FPS | Latency |
|-------|-------------|------------|---------------|-----|---------|
| **VGG16** | CNN (13 Conv + 3 FC) | 138M | **97.72%** | 267.5 | 9.49 ms |
| **ViT-Base** | Vision Transformer (Patch16) | 86M | 74.22% | 70.1 | 14.26 ms |
| **Swin-Base** | Swin Transformer (Patch4, W7) | 88M | 73.83% | 20.5 | 48.81 ms |

### Per-Image Prediction Accuracy (26 sample classes)
| Model | Correct | Avg Confidence |
|-------|---------|----------------|
| VGG16 | 6/26 | 39.1% |
| ViT-Base | **19/26** | **74.3%** |
| Swin-Base | 17/26 | 71.6% |

> **Key insight:** VGG16 achieved highest test accuracy (97.72%) but suffered from a "PlantDoc-Dataset" catch-all class prediction on out-of-distribution samples. ViT-Base demonstrated significantly stronger generalisation on unseen images despite lower overall accuracy.

---

## 🗂️ Repository Structure

```
├── Faster RCNN/              # Faster R-CNN experiments
├── SSD300/                   # SSD300 experiments
├── Swin-Base Transformer/    # Swin-Base training & evaluation
├── VGG16/                    # VGG16 training & evaluation
├── vision transformer/       # ViT-Base training & evaluation
├── yolov5/                   # YOLOv5s object detection
├── yolov8s_voc/              # YOLOv8s on PASCAL VOC 2012
├── Sample leaves/            # Sample leaf images used for evaluation
├── gradcam_output/           # Grad-CAM visualizations
├── gradcam_all_models.py     # Script to generate Grad-CAM for all models
└── model_results.xlsx        # Compiled results across all models
```

---

## 📁 Dataset

**Fused Dataset: PlantDoc + PlantVillage**
- Total images: **56,857**
- Classes: **30 plant disease categories**
- Fusion rationale: PlantDoc-only training yielded ~55% accuracy; dataset fusion was critical to achieving meaningful performance.

Disease classes include: Apple Scab, Apple Rust, Bell Pepper Leaf Spot, Blueberry Leaf, Cherry Leaf, Corn Gray Leaf Spot, Corn Rust, Grape Leaf Black Rot, Potato Early/Late Blight, Squash Powdery Mildew, Strawberry Leaf, Tomato Septoria Leaf Spot, Tomato Mosaic Virus, and more.

---

## 🔑 Key Observations

1. **VGG16** is fastest (267.5 FPS, 9.49ms latency) and best suited for real-time deployment, but shows class imbalance sensitivity with out-of-distribution inputs.
2. **ViT-Base** shows the strongest generalisation — 19/26 correct on unseen samples vs VGG16's 6/26 — making it preferable for robust field deployment.
3. **Swin-Base** is architecturally sophisticated but ~13x slower than VGG16 (48.81ms vs 9.49ms), limiting edge deployment suitability.
4. Transformers (ViT, Swin) are more parameter-efficient (86M/88M vs VGG16's 138M).
5. All models struggled with visually similar tomato disease classes (Tomato leaf yellow virus, Tomato mold leaf, Tomato bacterial spot).
6. Dataset fusion (PlantDoc + PlantVillage) was essential — ablation on PlantDoc-only showed ~55% accuracy, confirming that data diversity matters more than architecture choice alone.

---

## 🧠 Grad-CAM Explainability

Grad-CAM visualizations are generated for all three models to highlight which regions of the leaf each model attends to during classification. Output maps are stored in `gradcam_output/`.

To regenerate Grad-CAM visualizations:
```bash
python gradcam_all_models.py
```

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install torch torchvision timm opencv-python matplotlib numpy pandas
```

### Training
Each model folder contains its own training script. Navigate to the respective folder and run:
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py --model [vgg16 | vit | swin]
```

---

## 📈 Future Work

- Ensemble approaches combining CNN speed with Transformer generalisation
- Data augmentation strategies to improve performance on visually similar disease classes
- Edge deployment optimization for Swin-Base via pruning/quantization
- Extension to additional crop types beyond the current 30 classes

---

## 👤 Author

**Gourav Chouhan**  
B.Tech Information Technology, VIT Bhopal  
[GitHub](https://github.com/Gourav-Chouhan-IT) · [LinkedIn](https://www.linkedin.com/in/gourav-chouhan-071036374)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
