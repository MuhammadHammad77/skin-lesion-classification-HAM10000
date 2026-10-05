# Lab Assignment — Skin Lesion Boundary Detection Using Canny Edge Detection

## Overview

This lab applies **classical image processing techniques** (no deep learning) to detect and measure skin lesion boundaries in dermoscopic images from the **HAM10000 dataset**. The pipeline uses Gaussian filtering, Canny edge detection, morphological operations, and contour analysis to extract lesion boundaries, measure area/perimeter, and compare preprocessing × edge method combinations.

## Dataset

- **HAM10000** (Human Against Machine with 10000 training images)
- 5 images selected from 5 different lesion classes:
  - Image 1: **nv** (Melanocytic nevi) — ISIC_0025775
  - Image 2: **mel** (Melanoma) — ISIC_0031023
  - Image 3: **bcc** (Basal cell carcinoma) — ISIC_0031513
  - Image 4: **bkl** (Benign keratosis) — ISIC_0025661
  - Image 5: **vasc** (Vascular lesion) — ISIC_0031901

## Pipeline

```
Original Image → Grayscale → Gaussian Filter (5×5) → Canny Edge Detection → Morphological Closing → Largest Contour → Filled Mask → Area & Perimeter
```

## Notebook Structure

| Section | Task | Description |
|---------|------|-------------|
| §1 | Setup & Image Loading | Install dependencies, download HAM10000 via kagglehub, select 5 images |
| §2 | Preprocessing | Convert to grayscale, apply Gaussian blur (5×5 kernel) |
| §3 | Canny Edge Detection | Apply 3 threshold settings: (50,100), (100,200), (150,250) |
| §4 | Threshold Selection | Score each threshold using IoU against Otsu reference mask, select best |
| §5 | Boundary Detection | Morphological closing + contour extraction + green boundary overlay |
| §6 | Area & Perimeter | Measure lesion area (pixels), perimeter (pixels), percentage of image |
| §7 | Full Pipeline Visualization | 5×5 grid: Original → Grayscale → Gaussian → Canny → Boundary |
| §8 | Final Comparison | 4 preprocessing × 2 edge methods = 8 combinations, scored on noise handling, edge quality, boundary detection |
| §9 | Discussion & Answers | Answers to all 6 assignment questions |

## Key Results

### Best Threshold
- **50–100** was selected as best for all 5 images (highest IoU with Otsu reference)
- Mean IoU: 0.055 (low due to inherent difficulty of edge-only segmentation on dermoscopic images)

### Lesion Area & Perimeter

| Image | Class | Area (pixels) | Perimeter (pixels) | Lesion % |
|-------|-------|--------------|-------------------|----------|
| Image 1 | nv | 39,775 | 856.8 | 14.73% |
| Image 2 | mel | 218,752 | 4,040.9 | 81.02% |
| Image 3 | bcc | 2,936 | 271.8 | 1.09% |
| Image 4 | bkl | 36,551 | 2,447.0 | 13.54% |
| Image 5 | vasc | 14,753 | 854.7 | 5.46% |

### Best Overall Method
- **Average + Canny** (Overall score: 0.42)
- Average filter smooths noise while Canny produces thin, continuous edges with fewest fragments

## Key Findings

1. **Canny vs Sobel**: Canny consistently produced better edge quality (fewer fragments, lower density) than Sobel across all preprocessing methods
2. **Preprocessing impact**: Average and Gaussian filtering improved Sobel's noise robustness but had mixed effects on Canny
3. **Boundary detection challenge**: All methods scored low on boundary IoU (<0.12), showing that purely edge-based segmentation struggles with dermoscopic images due to hair, texture, and low contrast
4. **Fallback mechanism**: 3 of 5 images required Otsu thresholding fallback because Canny could not form a closed contour around low-contrast lesions

## Dependencies

```
pip install kagglehub opencv-python-headless numpy pandas matplotlib pillow
```

## How to Run

1. Open `ComputerVisionLab4_LesionBoundary.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Total runtime: ~2–3 minutes (no GPU needed — all CPU-based classical image processing)
4. Outputs are saved to `lab04_outputs/` folder

## Output Files

| File | Description |
|------|-------------|
| `01_originals.png` | 5 selected original images |
| `02_gray_gaussian.png` | Grayscale and Gaussian filtered versions |
| `03_canny_three_thresholds.png` | Canny edge maps at 3 threshold settings |
| `05_lesion_boundary.png` | Detected boundaries overlaid on originals |
| `07_final_pipeline.png` | Complete pipeline visualization (5×5 grid) |
| `08_final_comparison_visual.png` | Visual comparison of all preprocessing × edge combinations |
| `threshold_selection.csv` | Threshold scoring data |
| `table_area_perimeter.csv` | Area and perimeter measurements |
| `final_comparison.csv` | Final comparison scoring table |

## Techniques Used

- **Gaussian Blur**: Low-pass filter to suppress noise before edge detection
- **Canny Edge Detection**: Multi-stage edge detector with hysteresis thresholding
- **Otsu Thresholding**: Automatic threshold selection for reference mask generation
- **Morphological Operations**: Closing (to join edge gaps) and dilation/erosion (to refine contours)
- **Contour Analysis**: `cv2.findContours` for boundary extraction, `cv2.contourArea` and `cv2.arcLength` for measurements
- **IoU (Intersection over Union)**: Quantitative comparison of Canny masks vs Otsu reference masks

## Relationship to Other Labs

| Lab | Focus | Connection |
|-----|-------|------------|
| Lab 1 | CNN classification (DenseNet121, ResNet18, ResNet50) | Same HAM10000 dataset |
| Lab 2 | Effect of image filtering on classification | Same filters (Gaussian, Average, Median) |
| Lab 3 | Edge detection + classification comparison | Same edge detectors (Sobel, Canny) |
| **Lab 4** | **Boundary detection & measurement** | **Applies Canny for segmentation instead of classification** |
