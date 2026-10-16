# Lab Assignment — Landmark Recognition Using the SIFT Algorithm

## Overview

This lab builds a **landmark recognition system** using **SIFT (Scale-Invariant Feature Transform)** for feature extraction, **BFMatcher with KNN** for descriptor matching, **Lowe's Ratio Test** for filtering, and **Homography + RANSAC** for landmark localization. The system identifies famous landmarks from query images by matching them against a reference database.

## Dataset

- **Source:** Wikimedia Commons (public domain images, downloaded automatically via the Wikimedia API)
- **5 Landmarks:**
  - **Eiffel Tower** — Paris, France
  - **Taj Mahal** — Agra, India
  - **Great Wall** — Jinshanling, China
  - **Statue of Liberty** — New York, USA
  - **Colosseum** — Rome, Italy
- **3 reference views per landmark** (15 images total)
- Images are resized to 400x400 for consistency

## Pipeline

```
Query Image → Grayscale → SIFT Keypoints + 128-D Descriptors → BFMatcher (L2) + KNN (k=2) → Lowe's Ratio Test (0.75) → Match Count → Recognition Decision → Homography + RANSAC → Localization
```

## Notebook Structure

| Section | Task | Description |
|---------|------|-------------|
| §1 | Setup & Installation | Install opencv-contrib-python-headless (SIFT support) |
| §2 | Database Building (Task 1) | Download 5 landmarks x 3 views from Wikimedia Commons |
| §3 | SIFT Extraction (Task 2) | Detect keypoints and compute 128-D descriptors |
| §4 | Query Processing (Task 3) | Extract SIFT features from a query image |
| §5 | Feature Matching (Task 4) | BFMatcher + KNN + Lowe's ratio test |
| §6 | Localization (Task 5) | Homography estimation via RANSAC, bounding polygon |
| §7 | Performance Evaluation (Task 6) | Test under rotation, scale, brightness, viewpoint, unknown |
| §8 | Multi-Landmark Test | Recognition accuracy across all 5 landmarks |
| §9 | Manual Upload (Optional) | Test with your own landmark photo |
| §10 | Summary & Discussion | Key findings, limitations, conclusion |

## Key Results

### Feature Matching

| Component | Setting | Purpose |
|-----------|---------|---------|
| BFMatcher | L2 (Euclidean) distance | Compares 128-D SIFT descriptors |
| KNN | k=2 nearest neighbors | Finds two closest matches per descriptor |
| Lowe's Ratio | threshold = 0.75 | Filters ambiguous matches (best/second-best) |
| Match Threshold | 15 minimum matches | Rejects unknown landmarks below this count |

### Performance Under Different Conditions

| Condition | Expected Behavior |
|-----------|-------------------|
| Original | Recognized — baseline reference |
| Rotated (30 deg) | Recognized — SIFT is rotation-invariant |
| Resized (50%) | Recognized — SIFT is scale-invariant |
| Darkened (-40%) | Recognized — SIFT uses gradient orientations, not intensity |
| Different Viewpoint | Recognized — different view of same landmark |
| Unknown Image | Rejected — match count below threshold |

### Localization

| Parameter | Value | Description |
|-----------|-------|-------------|
| Homography | RANSAC (threshold 5.0) | Estimates geometric transformation |
| Min Inliers | 8 | Minimum RANSAC inliers to accept localization |
| Area Validation | 100–95% of image | Rejects invalid bounding polygons |

## Key Findings

1. **SIFT is scale-invariant** — detects features across multiple octaves, so resized images still produce reliable matches
2. **SIFT is rotation-invariant** — each keypoint's dominant orientation is computed and normalized, allowing matching at any angle
3. **Illumination robustness** — SIFT descriptors encode gradient orientations rather than raw pixel intensities, providing moderate robustness to brightness changes
4. **Lowe's ratio test is critical** — without it, BFMatcher returns many false matches; ratio 0.75 provides a good precision-recall balance
5. **Unknown rejection works** — random/noise images produce very few reliable matches, falling below the threshold
6. **Homography + RANSAC** — effectively localizes the landmark even with some outlier matches, by finding the consensus geometric transformation

## Limitations

- SIFT is patented (use ORB or AKAZE for commercial applications)
- Performance depends on image quality and viewpoint similarity
- No learning — adding a new landmark requires manually adding reference images
- Struggles with visually similar landmarks (e.g., similar-looking monuments)
- Occluded or partially visible landmarks may not have enough keypoint matches
- Computational cost scales linearly with database size

## Dependencies

```
pip install opencv-contrib-python-headless numpy pandas matplotlib
```

## How to Run

1. Open `ComputerVisionLab6_SIFT_LandmarkRecognition.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Images download automatically from Wikimedia Commons (no manual upload needed)
4. Total runtime: ~2–3 minutes (CPU only, no GPU needed)
5. Outputs are saved to `lab06_outputs/` folder

## Output Files

| File | Description |
|------|-------------|
| `01_database.png` | Reference image database (5 landmarks x 3 views) |
| `02_sift_keypoints.png` | SIFT keypoints visualized on each landmark |
| `03_query_processing.png` | Query image with detected SIFT keypoints |
| `04_matching.png` | Feature matching visualization (query vs best match) |
| `05_match_chart.png` | Bar chart of match counts per landmark |
| `06_localization.png` | Homography localization with bounding polygon |
| `07_performance.png` | Performance under all 6 test conditions |
| `08_multi_test.png` | Multi-landmark recognition test results |
| `evaluation_results.csv` | Performance evaluation metrics table |

## Techniques Used

- **SIFT (Scale-Invariant Feature Transform):** Detects keypoints at multiple scales using Difference-of-Gaussians and computes 128-dimensional gradient-based descriptors that are invariant to scale, rotation, and partial illumination changes
- **BFMatcher (Brute-Force Matcher):** Compares every query descriptor against every reference descriptor using L2 Euclidean distance
- **KNN Matching (k=2):** Returns the 2 nearest neighbors for each descriptor, enabling the ratio test
- **Lowe's Ratio Test:** Filters unreliable matches by requiring the best match distance to be significantly less than the second-best (ratio < 0.75)
- **Homography Estimation:** Computes the 3x3 perspective transformation matrix that maps reference image coordinates to query image coordinates
- **RANSAC (Random Sample Consensus):** Robust estimation that handles outlier matches by iteratively finding the consensus set of inlier matches
- **Perspective Transform:** Projects reference image corners through the homography to draw the landmark boundary on the query image

## Relationship to Other Labs

| Lab | Focus | Connection |
|-----|-------|------------|
| Lab 1 | CNN classification (DenseNet121, ResNet18, ResNet50) | Deep learning vs handcrafted features (SIFT) |
| Lab 2 | Effect of image filtering on classification | Filtering affects SIFT keypoint detection |
| Lab 3 | Edge detection + classification comparison | Edge-based features vs SIFT gradient descriptors |
| Lab 4 | Boundary detection & measurement | Boundary detection concepts relate to homography localization |
| Lab 5 | HOG + ML classifiers for defect detection | Both HOG and SIFT use gradient-based features |
| **Lab 6** | **SIFT landmark recognition + localization** | **Feature matching + geometric verification** |
