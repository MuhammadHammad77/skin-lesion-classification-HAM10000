# Lab Assignment — HOG-Based Industrial Defect Detection and Classification

## Overview

This lab applies **HOG (Histogram of Oriented Gradients)** feature extraction combined with classical machine learning classifiers to detect and classify steel surface defects from the **NEU Steel Surface Defect Database**. The pipeline covers preprocessing, feature extraction, multi-classifier comparison, parameter optimization, robustness testing, and an industrial quality-control decision module.

## Dataset

- **NEU Steel Surface Defect Database** (via Kaggle: `sovitrath/neu-steel-surface-defect-detect-trainvalid-split`)
- **6 Defect Classes:**
  - **Crazing** — fine crack patterns on the surface
  - **Inclusion** — foreign material embedded in the steel
  - **Patches** — irregular surface discoloration
  - **Pitted Surface** — small pits/holes on the surface
  - **Rolled-in Scale** — oxide scale pressed into the surface during rolling
  - **Scratches** — linear surface damage

## Pipeline

```
Original Image → Resize (128×128) → Grayscale → HOG Feature Extraction → StandardScaler → ML Classifier → Defect Type + ACCEPT/REJECT
```

## Notebook Structure

| Section | Task | Description |
|---------|------|-------------|
| §1 | Setup & Dataset Loading | Install dependencies, download dataset via kagglehub, load images with class labels |
| §2 | Data Inspection | Visualize sample images from each defect class, image size statistics |
| §3 | Preprocessing (Tasks 2–3) | Resize to 128×128, convert to grayscale |
| §4 | HOG Feature Extraction (Task 4) | Extract HOG features (orient=9, cell=8×8, block=2×2) |
| §5 | HOG Visualization (Task 5) | Visualize HOG gradient maps per defect class |
| §6 | SVM Classifier (Task 6) | Train SVM with RBF kernel (C=10) |
| §7 | Random Forest (Task 7) | Train Random Forest (300 trees) |
| §8 | Classifier Comparison (Task 8) | Compare SVM vs Random Forest vs KNN with bar charts |
| §9 | Confusion Matrix & Metrics (Tasks 9–10) | Confusion matrices, accuracy, precision, recall, F1-score |
| §10 | HOG Parameter Investigation (Task 11) | Grid search: 3 cell sizes × 3 orientations = 9 configs |
| §11 | Robustness Testing (Task 12) | Test under brightness ±50, noise σ=25, rotation 15°, blur 7×7 |
| §12 | Quality-Control Decision Module (Task 13) | ACCEPT/REJECT system with confidence scoring |
| §13 | Summary & Discussion | Key findings, limitations, and conclusion |

## Key Results

### Classifier Comparison

| Classifier | Description |
|------------|-------------|
| SVM (RBF, C=10) | Best overall — handles high-dimensional HOG features well |
| Random Forest (300 trees) | Competitive alternative, faster training |
| KNN (k=5) | Baseline, typically lower accuracy on HOG vectors |

### HOG Parameter Investigation

| Parameter | Values Tested | Finding |
|-----------|--------------|---------|
| Cell Size | 4×4, 8×8, 16×16 | 8×8 gives best balance of detail vs feature length |
| Orientations | 6, 9, 12 | 9 is standard; 12 gives diminishing returns |

### Robustness Testing

| Condition | Expected Impact |
|-----------|----------------|
| Brightness ±50 | Moderate — HOG is partially illumination-invariant |
| Gaussian Noise (σ=25) | Strongest negative impact — noise corrupts gradient histograms |
| Rotation (15°) | Significant — HOG is not rotationally invariant |
| Blur (7×7) | Reduces accuracy — smooths out edges HOG relies on |

### Quality-Control Decision Module

- Classifies each input image as one of 6 defect types
- Reports confidence score and probability distribution
- Outputs **ACCEPT** (non-defective / low confidence) or **REJECT** (defective / high confidence)

## Key Findings

1. **HOG captures defect patterns well** — each defect class produces a distinctive gradient signature (directional gradients for scratches, scattered patterns for pitting, fine structures for crazing)
2. **SVM with RBF kernel** generally performs best on HOG features due to effective handling of high-dimensional feature spaces
3. **Cell size has the strongest impact** — smaller cells (4×4) capture finer detail but create longer feature vectors; larger cells (16×16) lose fine defect information
4. **Gaussian noise is the biggest robustness challenge** — random gradients corrupt HOG histograms more than brightness or blur
5. **HOG is not rotationally invariant** — rotated defects produce different gradient histograms, reducing accuracy

## Limitations

- No "normal/non-defective" class in the NEU dataset — the model classifies defect type, not defect vs. normal
- HOG is not rotationally invariant — defect orientation matters
- Fixed image resolution (128×128) may lose detail from high-resolution industrial cameras
- Single-scale HOG may miss multi-scale defect patterns

## Dependencies

```
pip install kagglehub scikit-image scikit-learn opencv-python-headless numpy pandas matplotlib pillow
```

## How to Run

1. Open `ComputerVisionLab5_HOG_DefectDetection.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Total runtime: ~3–5 minutes (CPU only, no GPU needed)
4. Outputs are saved to `lab05_outputs/` folder

## Output Files

| File | Description |
|------|-------------|
| `01_sample_images.png` | Sample images from each defect class |
| `02_preprocessing.png` | Preprocessing pipeline (original → resize → grayscale) |
| `03_hog_visualization.png` | HOG feature visualization per class |
| `04_hog_distributions.png` | HOG feature value distributions |
| `05_classifier_comparison.png` | Accuracy and F1 bar charts for SVM vs RF vs KNN |
| `06_confusion_matrices.png` | Confusion matrices for all 3 classifiers |
| `07_hog_parameters.png` | Effect of cell size and orientations on accuracy |
| `08_hog_cell_sizes.png` | HOG visualization at different cell sizes |
| `09_robustness_examples.png` | Visual examples of each augmentation |
| `10_robustness_chart.png` | Accuracy change under each condition |
| `11_quality_control.png` | QC decision module results |
| `12_clean_surface.png` | Simulated non-defective surface test |
| `hog_parameter_comparison.csv` | Full parameter grid search results |
| `robustness_results.csv` | Robustness testing accuracy table |

## Techniques Used

- **HOG (Histogram of Oriented Gradients):** Captures edge and gradient structure by computing gradient magnitudes and orientations in local cells, then normalizing across blocks
- **SVM (Support Vector Machine):** RBF kernel classifier that finds optimal decision boundaries in high-dimensional HOG feature space
- **Random Forest:** Ensemble of 300 decision trees for robust classification
- **KNN (K-Nearest Neighbors):** Distance-based baseline classifier
- **StandardScaler:** Z-score normalization of HOG features before classification
- **scikit-image HOG:** `skimage.feature.hog` with L2-Hys block normalization
- **Confusion Matrix & Classification Report:** Per-class precision, recall, F1-score evaluation

## Relationship to Other Labs

| Lab | Focus | Connection |
|-----|-------|------------|
| Lab 1 | CNN classification (DenseNet121, ResNet18, ResNet50) | Deep learning approach to image classification |
| Lab 2 | Effect of image filtering on classification | Same preprocessing concepts (Gaussian, Average, Median) |
| Lab 3 | Edge detection + classification comparison | Edge-based features (Sobel, Canny) vs HOG gradients |
| Lab 4 | Boundary detection & measurement | Classical CV segmentation on medical images |
| **Lab 5** | **HOG + ML classifiers for defect detection** | **Handcrafted features (HOG) + classical ML on industrial images** |
