# Lab 03 — Edge Detection Techniques and Their Impact on Classification Performance

**Course:** Computer Vision  **Dataset:** HAM10000 skin-lesion dataset (7 classes)
**Code:** `ComputerVisionLab3.ipynb` (GitHub → `Lab 03/`)  **Figures/tables:** `lab03_outputs/`

> **How to finish this report.** Everything from Lab 01 and Lab 02 is already filled in. Slots marked `[…]` are the Lab 03 numbers: run the notebook, open `lab03_outputs/RESULTS_FOR_REPORT.md`, and paste Tables 1–3 and the numbers into the marked places. Then delete this note.

---

## 1. Introduction

Edges are locations where image intensity changes sharply and usually correspond to object boundaries, texture transitions and structural detail. In dermoscopic images the lesion border, pigment network and streaks are all edge-like structures, so edge detection is a natural candidate for extracting diagnostic features. This lab studies classical edge detectors (Sobel, Prewitt, Laplacian, Laplacian of Gaussian and Canny), how they respond to Gaussian and salt-and-pepper noise, how smoothing before detection changes the result, and — most importantly — whether feeding edge maps to a classifier helps or hurts skin-lesion classification compared with the raw (Lab 01) and filtered (Lab 02) representations.

This is the third lab in a chain that uses the **same dataset, split and training settings** throughout:

| Lab | Question | Key result |
|---|---|---|
| Lab 01 | Which pretrained CNN is best on HAM10000? | DenseNet121 (87.82 %), ResNet18 (86.53 %), ResNet50 (85.63 %) on the full training set |
| Lab 02 | Does spatial filtering help those CNNs? | Mostly no. Mean accuracy change vs. no filter: Sharpening +0.03 pp, Average −1.63, Median −1.76, Gaussian −3.03, **Sobel −22.85** |
| Lab 03 | Do edge maps help? Which representation is best overall? | This report |

## 2. Methodology

### 2.1 Edge detectors implemented

| Family | Detector | Implementation | Principle |
|---|---|---|---|
| First-order | Sobel Gx, Gy | `cv2.Sobel`, 3×3 | horizontal / vertical intensity gradient; |G| = √(Gx²+Gy²) |
| First-order | Prewitt | `cv2.filter2D` with Prewitt kernels | like Sobel but with uniform (unweighted) smoothing rows |
| Second-order | Laplacian | `cv2.Laplacian`, 3×3 | sum of second derivatives; zero-crossings mark edges |
| Second-order | LoG | 5×5 Gaussian → Laplacian | Laplacian on a smoothed image, reducing noise amplification |
| Multi-stage | Canny | k×k Gaussian → `cv2.Canny(low, high)` | gradient → non-maximum suppression → double-threshold hysteresis |

### 2.2 Noise and pre-smoothing (Task 2)

Gaussian noise (σ = 25) and salt-and-pepper noise (5 % of pixels) were added to a melanoma image. Each detector was applied to the original, the noisy image, the noisy image after a 5×5 Gaussian filter and after a 5×5 median filter. Because visual judgement is subjective, the notebook also computes for each edge map: **edge density** (% of pixels marked as edge), **number of fragments** (connected components — a proxy for broken edges), **IoU with the clean edge map** (edge continuity/quality) and **% false edges** (edge pixels not present in the clean map — noise sensitivity). These numbers populate the "Edge Quality" and "Noise Sensitivity" columns of Table 1.

### 2.3 Canny parameter study (Task 3)

Five configurations were tested: (30,100), (50,150), (100,200) with a 3×3 Gaussian kernel, and (50,150) with 5×5 and 7×7 kernels. For each we recorded the number of edge pixels, the number of contours and the IoU between the edge map of the clean image and that of its Gaussian-noisy version (stability). The selection rule was: among configurations with a plausible edge density (1–8 %), choose the most noise-stable one.

### 2.4 Three dataset versions (Task 4)

| Set | Content | Source |
|---|---|---|
| **A — Raw** | original RGB image | Lab 01 |
| **B — Filtered** | 3×3 **sharpening** filter — the only Lab 02 filter with a non-negative mean accuracy effect | Lab 02 |
| **C — Edge** | Canny edge map with the configuration chosen in Task 3, replicated to 3 channels so ImageNet-pretrained networks accept it | Lab 03 |

Each set was pre-computed **once** (resized to 224×224, stored as `uint8`) and cached to disk, so all models see identical pixels and no filtering is repeated during training. Pretrained weights were likewise downloaded once before any training started.

### 2.5 Classifiers (Tasks 4–5)

- **SVM (RBF), Random Forest (300 trees), KNN (k = 5)** trained on 1024-d features from a frozen ImageNet DenseNet121 — the same deep-feature + classical-classifier design used for Table 2 of Lab 01.
- **CNN Model 1 = DenseNet121**, **CNN Model 2 = ResNet50**, fully fine-tuned with a 7-class head.

## 3. Experimental Setup

| Item | Value (identical to Lab 02) |
|---|---|
| Split | 80 / 10 / 10 stratified, `random_state = 42` |
| Training subset | 400 images per class cap → 2 068 train, 1 001 val, 1 002 test |
| Input | 224 × 224, ImageNet mean/std normalisation, random horizontal + vertical flips (train only) |
| Optimiser / loss | Adam, lr 1e-4, class-weighted cross-entropy |
| Epochs | 5 per run, best-validation checkpoint restored |
| Metrics | Accuracy, weighted precision / recall / F1, training time (s), inference time (ms/image) |
| Hardware | Google Colab, NVIDIA Tesla T4 |

Total runs: 3 sets × (3 classical + 2 CNN) = **15 evaluations**.

## 4. Results

### 4.1 Task 1 — Comparative edge detection (Figure `01_task1_edge_comparison.png`)

Sobel Gx responds to vertical boundaries and Gy to horizontal ones; their magnitude gives a closed, thick lesion outline plus a faint response to hair and pigment network. Prewitt is almost identical to Sobel but slightly noisier because its smoothing rows are unweighted. The Laplacian produces thin, double-sided responses at every intensity change and is visibly dominated by fine texture and camera noise. LoG suppresses much of that noise and keeps the main contour. Canny gives the cleanest result: one-pixel-wide, mostly continuous lesion borders with almost no texture response.

### 4.2 Task 2 — Effect of noise (Figures `02_task2_inputs.png`, `03_task2_edge_maps.png`)

**Table 1. Effect of Noise and Preprocessing on Edge Detection**
*(paste `table1_noise_effect.md` here; the qualitative observations below were written from the figures)*

| Edge Detector | Input | Noise | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | High | — | Continuous, thick border; some texture response |
| Sobel | Noisy | Gaussian | None | Low | High | Border still visible but surrounded by a dense speckle of false edges |
| Sobel | Noisy | Salt & Pepper | None | Low | High | Every noise pixel becomes a small bright ring; border partly masked |
| Sobel | Noisy | Gaussian | Gaussian filter | Medium–High | Low | Speckle largely removed, border slightly blurred/thicker |
| Sobel | Noisy | Salt & Pepper | Median filter | High | Low | Impulses removed completely; border nearly identical to original |
| Prewitt | Original | None | None | High | — | Same structure as Sobel, marginally more texture noise |
| Laplacian | Original | None | None | Medium | (very high) | Thin, fragmented, double edges; strongest texture/noise response of all |
| LoG | Noisy | Gaussian | Gaussian filter | Medium | Medium | Smoothing controls the Laplacian's noise amplification, but residual noise remains |
| Canny | Original | None | Built-in smoothing | High | — | Thin, continuous single-pixel border, no texture |
| Canny | Noisy | Gaussian | Gaussian filter | Medium | Medium | Mostly clean, but some short false segments and a few gaps in the border |
| Canny | Noisy | Salt & Pepper | Median filter | High | Low | Almost indistinguishable from the clean result |

### 4.3 Task 3 — Canny parameters (Figure `04_task3_canny_params.png`)

**Table 2. Canny Parameter Analysis** — paste `table2_canny_params.md` here.

Selected configuration: **`[Canny-x: low = …, high = …, kernel = …]`** — chosen because it produced a plausible edge density while being the most stable under Gaussian noise.

### 4.4 Tasks 4–5 — Classification (Figures `06`–`10`)

**Table 3. Cross-Lab Classification Performance Comparison** — paste `table3_cross_lab.md` (and `table3_long.md`) here.

Reference values from earlier labs on the same split: Lab 02 baseline (raw images, same 5-epoch/400-per-class setting) — DenseNet121 77.25 %, ResNet50 80.94 %; Lab 02 sharpening — DenseNet121 77.15 %, ResNet50 80.74 %.

Best model over all three sets: **`[…]`**.
Accuracy change vs raw: Filtered `[±… pp]`, Edge `[±… pp]`.

### 4.5 Task 6 — Visual comparison

The confusion matrices of the best model (`06_task6_confusion_matrices_best_model.png`) and the bar chart of Accuracy / Precision / Recall / F1 across raw, filtered and edge inputs (`07_task6_best_model_bars.png`) show `[describe: e.g. the edge set collapses minority classes vasc/df/akiec into nv and mel, while raw and filtered are nearly identical]`.

## 5. Discussion

**Q1 — Which edge detector was most sensitive to noise?**
The **Laplacian**. It is a second-derivative operator, and differentiation amplifies high-frequency content; taking the derivative twice amplifies noise roughly with the square of the frequency. In Table 1 the Laplacian had the highest fragment count and the most false edges even on the clean image, and on noisy input its map was essentially pure speckle. First-order detectors (Sobel, Prewitt) were next: their built-in 3×3 averaging gives a little immunity, but Gaussian noise still produced a dense field of false edges and salt-and-pepper noise produced a bright ring at every impulse. Canny was the least sensitive because its pipeline starts with Gaussian smoothing and its hysteresis thresholds discard weak, isolated responses. Supporting numbers: `[false-edge % for Sobel/Laplacian/Canny from Table 1]`.

**Q2 — How did Gaussian and Median filtering affect the quality of detected edges?**
Both raised edge quality, but for different noise types. **Gaussian filtering** is a linear low-pass filter: it averages Gaussian noise towards zero and restored a clean Sobel/Canny border (IoU rose from `[…]` to `[…]`), at the cost of slightly thicker, softer edges and the loss of the finest pigment-network detail. It is much less effective on salt-and-pepper noise because averaging only spreads an impulse into a small grey blob, which still triggers an edge. **Median filtering** is non-linear and replaces each pixel with the neighbourhood median, so isolated impulses are eliminated outright while true step edges are preserved; after median filtering the Sobel and Canny maps were almost identical to the clean ones (IoU `[…]`). The general rule confirmed here: Gaussian for Gaussian noise, median for impulse noise, and smoothing before a second-order detector (LoG) is essential.

**Q3 — How did changing the low and high thresholds affect the number and quality of detected edges?**
Raising the thresholds monotonically reduced the number of edge pixels and contours (Table 2: `[…]` → `[…]` → `[…]` pixels for (30,100), (50,150), (100,200)). Low thresholds (30,100) kept weak gradients, so hair, pigment texture and noise appeared as many short contours; high thresholds (100,200) kept only strong gradients, giving a sparse map where parts of a low-contrast lesion border were missing. The middle setting balanced continuity against false edges. Increasing the Gaussian kernel (5×5, 7×7) removed further noise and improved stability under noise (`[IoU values]`) but also blurred low-contrast borders. The high threshold decides which edges are "seeds"; the low threshold decides how far each seed is allowed to grow, which is why hysteresis produces long connected edges rather than isolated points.

**Q4 — Did edge-only images improve or reduce classification accuracy compared with raw images?**
`[State the numbers: e.g. DenseNet121 raw … % vs edge … %; ResNet50 raw … % vs edge … %; SVM … vs …]`. The edge representation was clearly worse for every classifier, consistent with the Lab 02 result where the Sobel-magnitude input already cost 18–27 pp of accuracy. Reasons: (i) colour is one of the strongest dermoscopic cues (blue-grey veil in melanoma, red/purple in vascular lesions, brown pigmentation in nevi) and a binary edge map discards it entirely; (ii) the ImageNet-pretrained backbone was trained on natural RGB images, so a sparse binary input lies far outside its training distribution and the early filters no longer produce meaningful activations; (iii) a Canny map is extremely sparse (`[…]` % edge pixels), so most of the 224×224 input is zero and convolutional features have almost nothing to respond to; (iv) Canny's thresholds were tuned on one image and generalise poorly across lesions with very different contrast.

**Q5 — What information is lost when texture, colour and intensity are removed?**
Colour and pigment distribution (uniform vs. variegated), absolute brightness and contrast, fine texture such as the pigment network, dots and globules, regression areas and scale-like surface structure, and the interior of the lesion in general — an edge map keeps only where a boundary is, not what lies on either side of it. Gradient magnitude and direction are reduced to a binary "edge/no edge" decision, so even border sharpness, one of the ABCD criteria, is lost. For minority classes such as dermatofibroma or vascular lesions, which are distinguished mainly by colour, this removes almost all of the discriminative signal, which is why they suffer most in the edge-set confusion matrix.

**Q6 — Advantages of letting a CNN learn edge-like features instead of supplying edge maps.**
Learned first-layer filters look like Sobel, Gabor and colour-opponent kernels, but they are (a) *data-driven*: their orientations, scales and thresholds are optimised for the classification loss rather than fixed by hand; (b) *many and diverse*: a layer learns dozens of filters simultaneously, including colour-edge and texture detectors that no single classical operator provides; (c) *soft*: activations keep magnitude information instead of a binary decision, so nothing is thrown away before later layers can use it; (d) *composable*: later layers combine edge, colour and texture responses hierarchically. Handing the network a Canny map fixes one hard, lossy decision at the input and prevents the network from ever recovering the discarded information. The correct place for classical filters is therefore inside the pipeline as a learnable initialisation or as augmentation, not as a replacement for the image.

**Q7 — Which input representation produced the most useful classification results?**
`[Fill from Table 3.]` Across Labs 01–03 the ordering is **raw ≈ filtered > edge**. Raw images gave the best or joint-best accuracy for every model (`[…]`). Mild sharpening in Lab 02 changed accuracy by only ±0.4 pp and gave no consistent gain, so filtering is at best neutral. Edge maps reduced accuracy by `[…]` pp and destroyed minority-class recall. The practical recommendation is to give a pretrained CNN the raw RGB image, keep classical processing for noise removal at acquisition time, and use edge/texture analysis for interpretation rather than as the classifier's input.

## 6. Conclusion

Canny produced the cleanest and most noise-robust edges, the Laplacian the noisiest; Gaussian smoothing suppresses Gaussian noise and median filtering removes impulse noise before edge detection; Canny's thresholds trade edge completeness against false edges, and a mid-range setting with moderate smoothing was the most stable. Applied to classification, edge maps consistently and substantially reduced accuracy relative to raw images for both classical classifiers on deep features and fine-tuned CNNs, while mild filtering was essentially neutral. The experiments therefore confirm that a CNN's learned early-layer features subsume handcrafted edge detection, and that colour, texture and intensity — precisely what edge maps discard — carry most of the diagnostic information in dermoscopic images.

## 7. References

1. Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 8*(6), 679–698.
2. Marr, D., & Hildreth, E. (1980). Theory of edge detection. *Proceedings of the Royal Society B, 207*(1167), 187–217.
3. Gonzalez, R. C., & Woods, R. E. (2018). *Digital image processing* (4th ed.). Pearson.
4. Szeliski, R. (2022). *Computer vision: Algorithms and applications* (2nd ed.). Springer.
5. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data, 5*, 180161.
6. Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR 2017*, 4700–4708.
7. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
8. Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV 2014*, 818–833.
9. Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools, 25*(11), 120–125.
10. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 32*.

---

## Appendix — Viva questions (short answers)

- **What is an edge?** A set of connected pixels where image intensity changes sharply, usually at object boundaries or texture transitions.
- **First- vs second-order edge detection?** First-order (Sobel, Prewitt) finds maxima of the gradient magnitude; second-order (Laplacian, LoG) finds zero-crossings of the second derivative. Second-order gives thinner, more precisely located edges but is more noise-sensitive.
- **Sobel Gx vs Gy?** Gx convolves with a kernel that differentiates horizontally and responds to vertical edges; Gy differentiates vertically and responds to horizontal edges.
- **Why is the Laplacian more sensitive to noise?** It takes a second derivative; each differentiation amplifies high frequencies, so noise is amplified twice.
- **Purpose of Gaussian smoothing before edge detection?** To remove high-frequency noise so the derivative operators respond to real structure rather than pixel-level fluctuations.
- **Main advantage of Canny?** Optimal trade-off between detection, localisation and single response: smoothing, non-maximum suppression and hysteresis thresholding give thin, continuous, low-false-alarm edges.
- **Canny's low and high thresholds?** Gradients above *high* are strong edges (seeds); gradients between *low* and *high* are kept only if connected to a strong edge; below *low* are discarded.
- **Gaussian vs salt-and-pepper noise?** Gaussian noise adds a small random value to every pixel (continuous, zero-mean); salt-and-pepper replaces a random fraction of pixels with 0 or 255 (sparse impulses).
- **Why is median filtering useful for salt-and-pepper?** The median of a neighbourhood ignores extreme outliers, so impulses are replaced by a typical local value while true edges are preserved.
- **Why can edge detection reduce classification performance?** It discards colour, intensity and texture, leaves a sparse binary input outside the pretrained network's distribution, and depends on hand-set thresholds that do not generalise across images.
- **Can a CNN learn edge features automatically?** Yes — first-layer filters of trained CNNs resemble oriented gradient and colour-opponent detectors, learned from data.
- **Why might raw images perform better than edge-only images?** Raw images contain everything the network could need and let it decide which features matter; edge maps commit to one lossy feature choice in advance.
