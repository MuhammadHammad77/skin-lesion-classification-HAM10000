# Lab 02 — Effect of Image Filtering on Skin-Lesion Classification

Investigates how spatial-domain image filters affect pretrained CNNs on the HAM10000
skin-lesion dataset.

## Experiment design

- **Dataset:** HAM10000, 7 classes (`akiec, bcc, bkl, df, mel, nv, vasc`)
- **Split:** 80 / 10 / 10 stratified, `random_state=42` — identical to Lab 01
- **Models (best three from Lab 01):** DenseNet121, ResNet18, ResNet50
- **Filters:** none (baseline), Average (5×5), Gaussian (5×5), Median (5),
  Sharpening (3×3), Sobel (3×3 gradient magnitude)
- **Runs:** 3 models × 6 conditions = 18
- **Fixed across all runs:** 224×224 input, batch 32, Adam lr=1e-4, 5 epochs,
  class-weighted cross-entropy, same augmentation, same seed, same test set

## Lab 01 baseline (models selected from)

| Rank | Model | Accuracy | F1 | AUC |
|---|---|---|---|---|
| 1 | DenseNet121 | 87.82% | 87.73% | 97.78% |
| 2 | ResNet18 | 86.53% | 86.59% | 96.94% |
| 3 | ResNet50 | 85.63% | 85.67% | 97.74% |

## How to run

1. Open `ComputerVisionLab2.ipynb` in Google Colab.
2. `Runtime → Change runtime type → GPU (T4)`.
3. `Runtime → Run all`.

The dataset downloads automatically via `kagglehub` (no Kaggle API key needed for
this public dataset). Run time is roughly 1.5–2 hours on a T4 with the default
balanced training subset.

### Configuration

All knobs are in the **configuration cell** in Section 1:

```python
EPOCHS        = 5      # same for every run
USE_SUBSET    = True   # set False to train on the complete dataset
MAX_PER_CLASS = 400    # cap per class, training split only
```

Set `USE_SUBSET = False` for the full dataset (expect 5–6 hours).

### Resuming after a disconnect

Results are checkpointed to `lab02_results.csv` and `lab02_histories.json` after
every run, and completed runs are skipped. If Colab disconnects, just re-run the
experiment cell in Section 7.

## Outputs

| File | Contents |
|---|---|
| `lab02_comparison_table.csv` | Main required comparison table (18 rows) |
| `lab02_per_class_metrics.csv` | Per-class precision / recall / F1 for every run |
| `lab02_results.csv` | Raw metrics + runtime per run |
| `lab02_all_results.xlsx` | All of the above as one workbook |
| `checkpoints/` | Best weights per (model, filter) |

## Notebook structure

1. Setup & configuration
2. Dataset preparation (download, split, class distribution)
3. Image filtering — OpenCV implementations + visual examples
4. Dataset & dataloaders
5. Model loading
6. Training + evaluation function
7. Run all 18 experiments
8. Comparison table + change vs baseline
9. Visualization — bar charts, heatmap, curves, confusion matrices, ROC
10. Comparative analysis + answers to the ten questions
