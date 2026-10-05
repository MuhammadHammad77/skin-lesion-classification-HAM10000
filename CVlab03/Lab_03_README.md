# Lab 03 — Edge Detection Techniques and Their Impact on Classification Performance

## Objectives
This laboratory studies classical edge-detection methods and their effect on image classification performance.

The notebook includes:
- Sobel edge detection
- Prewitt edge detection
- Laplacian edge detection
- Laplacian of Gaussian (LoG)
- Canny edge detection
- noise analysis
- smoothing analysis
- Canny parameter analysis
- raw vs filtered vs edge-image classification comparison

## Notebook
`Lab_03_Edge_Detection_Classification.ipynb`

## Dataset
Use the same HAM10000 dataset and the same selected classes used in Labs 01 and 02.

Expected files:
- `HAM10000_metadata.csv`
- HAM10000 image folders

Update the dataset path inside the notebook:

```python
DATA_ROOT = Path("/content/HAM10000")
```

## Important Cross-Lab Settings
Use the same selected classes as Labs 01 and 02:

```python
SELECTED_CLASSES = None
```

or, for example:

```python
SELECTED_CLASSES = ["nv", "mel", "bkl"]
```

Set the best preprocessing/filtering method identified in Lab 02:

```python
BEST_LAB02_FILTER = "Gaussian"
```

Replace this with your actual best Lab 02 filter.

Set the selected CNN models:

```python
CNN_MODEL_NAMES = ["EfficientNetB0", "MobileNetV2"]
```

Change them if your required models are different.

## Task 1 — Comparative Edge Detection
The notebook implements:
- Sobel Gx
- Sobel Gy
- Sobel gradient magnitude
- Prewitt
- Laplacian
- LoG
- Canny

It displays:

Original Image → Sobel → Prewitt → Laplacian → LoG → Canny

for at least three different lesion classes.

## Task 2 — Effect of Noise
Artificial noise is added using:
- Gaussian noise
- Salt-and-pepper noise

Edge detection is tested on:
- original image
- noisy image
- noisy image after Gaussian filtering
- noisy image after Median filtering

The notebook produces Table 1 containing:
- edge detector
- input image
- noise type
- preprocessing
- edge quality
- noise sensitivity
- observations

The analysis considers:
- edge continuity
- edge sharpness
- false edges
- broken edges
- noise sensitivity
- effect of smoothing

## Task 3 — Canny Parameter Analysis
The notebook evaluates multiple Canny configurations, including:

- Low = 30, High = 100
- Low = 50, High = 150
- Low = 100, High = 200
- alternative Gaussian kernel size

It generates Table 2 containing:
- configuration
- low threshold
- high threshold
- kernel size
- edge quality
- number of detected edges
- observations

The notebook also selects a useful Canny configuration using an objective edge-density heuristic.

## Task 4 — Classification Using Edge Maps
Three versions of the dataset are created:

### Set A — Raw Images
Original images.

### Set B — Filtered Images
Uses the best filtering method identified in Lab 02.

### Set C — Edge Images
Uses the selected Canny edge representation.

The same:
- training set
- validation set
- test set
- epochs
- evaluation metrics

are used across all representations.

## Task 5 — Classification Performance Comparison
The notebook evaluates:
- SVM
- Random Forest
- KNN
- CNN Model 1
- CNN Model 2

Metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Inference time

The notebook generates Table 3 for cross-lab comparison.

## Classical Model Features
To keep SVM, Random Forest, and KNN computationally manageable, the notebook uses frozen MobileNetV2 embeddings as common feature vectors.

The same extracted-feature procedure is applied to:
- Raw
- Filtered
- Edge

representations.

## Task 6 — Visual Classification Comparison
For the best observed model, the notebook generates:
- Raw-image confusion matrix
- Filtered-image confusion matrix
- Edge-image confusion matrix
- Accuracy comparison
- Precision comparison
- Recall comparison
- F1-score comparison

## Installation
Typical packages:

```bash
pip install tensorflow opencv-python-headless scikit-learn pandas matplotlib
```

Google Colab with GPU is recommended.

## Running the Notebook
Run from top to bottom:

1. Install/import packages
2. Configure dataset path
3. Load HAM10000
4. Create fixed train/validation/test split
5. Run comparative edge detection
6. Add Gaussian and salt-and-pepper noise
7. Compare smoothing effects
8. Complete Table 1
9. Run Canny parameter analysis
10. Complete Table 2
11. Select the best edge configuration
12. Create Raw, Filtered, and Edge datasets
13. Extract features for classical models
14. Train SVM, Random Forest, and KNN
15. Train the two CNN models
16. Complete Table 3
17. Generate confusion matrices
18. Generate metric comparison charts
19. Use the results to answer discussion questions

## Output Files
The notebook can create:
- `lab03_table1_noise_preprocessing.csv`
- `lab03_table2_canny_parameters.csv`
- `lab03_table3_cross_lab_comparison.csv`
- `lab03_all_representation_results.csv`

## Reproducibility
For a fair comparison, keep constant:
- selected classes
- dataset split
- image size
- number of epochs
- evaluation metrics
- random seed
- training configuration

## Discussion and Viva
The notebook includes a result-driven discussion guide and concise answers for the required viva questions.
