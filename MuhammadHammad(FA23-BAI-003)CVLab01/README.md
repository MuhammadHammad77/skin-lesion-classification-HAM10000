# Skin Lesion Classification Using Transfer Learning

A comparative study of **8 deep learning models** and **7 classical classifiers** for automated skin lesion classification on the HAM10000 dataset.

---

## Overview

Skin cancer is one of the most common cancers worldwide. Early and accurate detection through dermoscopic image analysis can significantly improve patient outcomes. This project evaluates multiple transfer learning architectures and classical machine learning classifiers for classifying **7 types of skin lesions** from dermoscopic images.

### Key Findings

| Approach | Best Model | Accuracy |
|----------|-----------|----------|
| Transfer Learning | DenseNet121 | 87.82% |
| Deep Features + Classifier | XGBoost | 89.22% |
| Most Efficient | EfficientNet-B0 | 83.93% (only 5.29M params) |

---

## Dataset

**HAM10000** (Human Against Machine with 10000 training images) — a large collection of multi-source dermoscopic images of common pigmented skin lesions.

| Class | Description | Images |
|-------|------------|--------|
| nv | Melanocytic Nevi | 6,705 |
| mel | Melanoma | 1,113 |
| bkl | Benign Keratosis-like Lesions | 1,099 |
| bcc | Basal Cell Carcinoma | 514 |
| akiec | Actinic Keratoses | 327 |
| vasc | Vascular Lesions | 142 |
| df | Dermatofibroma | 115 |
| **Total** | | **10,015** |

**Split:** 80% train / 10% validation / 10% test (stratified to preserve class ratios)

**Source:** [Kaggle — Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

---

## Methodology

### 1. Data Preprocessing
- Images resized to **224 × 224** pixels
- Normalized using ImageNet mean and standard deviation
- Data augmentation: random horizontal/vertical flips, affine transforms

### 2. Class Imbalance Handling
- Weighted cross-entropy loss — rare classes receive higher weight during training

### 3. Transfer Learning (Table 1)
Eight pretrained models (ImageNet weights) with the final classification layer replaced for 7-class output:

- AlexNet
- VGG16, VGG19
- ResNet18, ResNet50, ResNet101
- DenseNet121
- EfficientNet-B0

**Training:** Adam optimizer, learning rate 1e-4, best model saved based on validation accuracy.

### 4. Deep Feature Extraction + Classical Classifiers (Table 2)
The best-performing model (DenseNet121) was used as a feature extractor, producing a **1024-dimensional** feature vector per image. Seven classifiers were trained on these features:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Linear SVM
- RBF-SVM
- XGBoost

### 5. Computational Efficiency (Table 3)
Each model was profiled for parameter count, model size, FLOPs, and inference time.

---

## Results

### Table 1 — Transfer Learning Models

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| AlexNet | 75.45% | 80.54% | 75.45% | 77.06% | 95.10% |
| VGG16 | 73.95% | 80.27% | 73.95% | 75.62% | 93.89% |
| VGG19 | 54.59% | 71.26% | 54.59% | 58.44% | 82.10% |
| ResNet18 | 86.53% | 86.76% | 86.53% | 86.59% | 96.94% |
| ResNet50 | 85.63% | 86.19% | 85.63% | 85.67% | 97.74% |
| ResNet101 | 84.13% | 86.33% | 84.13% | 84.85% | 97.81% |
| **DenseNet121** | **87.82%** | **87.74%** | **87.82%** | **87.73%** | **97.78%** |
| EfficientNet-B0 | 83.93% | 85.71% | 83.93% | 84.55% | 97.33% |

### Table 2 — Deep Features + Classical Classifiers

| Classifier | Accuracy | Precision | Recall | F1-Score | AUC |
|-----------|----------|-----------|--------|----------|-----|
| Logistic Regression | 88.92% | 88.92% | 88.92% | 88.88% | 97.98% |
| Decision Tree | 80.94% | 80.72% | 80.94% | 80.59% | 80.00% |
| Random Forest | 86.23% | 85.71% | 86.23% | 85.20% | 98.11% |
| KNN | 87.43% | 86.76% | 87.43% | 86.78% | 94.59% |
| Linear SVM | 88.32% | 87.74% | 88.32% | 87.79% | 98.09% |
| RBF-SVM | 88.92% | 88.57% | 88.92% | 88.62% | 98.22% |
| **XGBoost** | **89.22%** | **88.85%** | **89.22%** | **88.83%** | **98.31%** |

### Table 3 — Computational Efficiency

| Model | Params (M) | Size (MB) | FLOPs (G) | Inference (ms) | Accuracy |
|-------|-----------|-----------|-----------|----------------|----------|
| AlexNet | 61.10 | 244.4 | 0.71 | 2.0 | 75.45% |
| VGG16 | 138.36 | 553.4 | 15.47 | 9.3 | 73.95% |
| VGG19 | 143.67 | 574.7 | 19.63 | 11.3 | 54.59% |
| ResNet18 | 11.69 | 46.8 | 1.82 | 2.9 | 86.53% |
| ResNet50 | 25.56 | 102.6 | 4.13 | 8.9 | 85.63% |
| ResNet101 | 44.55 | 178.8 | 7.87 | 11.3 | 84.13% |
| DenseNet121 | 7.98 | 32.6 | 2.90 | 14.9 | 87.82% |
| **EfficientNet-B0** | **5.29** | **21.6** | **0.42** | **8.3** | **83.93%** |

---

## How to Run

### Requirements
```
Python 3.8+
PyTorch
torchvision
scikit-learn
xgboost
thop
pandas
pillow
kagglehub
```

### Quick Start (Google Colab / Kaggle)

```bash
pip install -q kagglehub thop xgboost
```

```python
import kagglehub

path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

Then run the notebook cells in order: data loading → model training → evaluation → efficiency profiling.

---

## Project Structure

```
├── README.md
├── Task_01CV_Filled.docx      # Filled results tables
├── ham10000_assignment.py      # Complete training + evaluation script
└── notebooks/
    └── skin_lesion_classification.ipynb
```

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| PyTorch + torchvision | Model training and transfer learning |
| scikit-learn | Classical classifiers and metrics |
| XGBoost | Gradient boosting classifier |
| thop | FLOPs and parameter counting |
| kagglehub | Dataset download |

---

## Reference Paper

Ratul, M. A. et al. — *Skin Lesions Classification Using Deep Learning Based on Dilated Convolution* (bioRxiv, 2020). Uses dilated convolution with VGG16, VGG19, MobileNet, and InceptionV3 on HAM10000, achieving up to 89.81% accuracy with Dilated InceptionV3.

---

## License

This project is for academic/educational purposes as part of a university assignment.
