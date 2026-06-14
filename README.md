# 🔭 MAGIC Gamma Telescope — Binary Classification

> *Can a machine learn to see what the human eye cannot? Separating gamma rays from cosmic noise using supervised learning.*

This project is a **comparative study of six supervised machine learning algorithms** applied to the binary classification problem of distinguishing gamma-ray events from hadronic background noise in data recorded by atmospheric Cherenkov telescopes.

The work was developed as an assignment for the **Machine Learning** course — Engineering in Computer Science and Artificial Intelligence, Sapienza University of Rome (A.Y. 2025/2026).

---

## 📑 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models](#models)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Author](#author)

---

## Problem Statement

When a high-energy gamma ray reaches Earth's atmosphere, it produces a cascade of particles — a **shower** — that emits faint blue Cherenkov radiation. The MAGIC telescope captures an image of this shower, from which geometric features (Hillas parameters) are extracted.

The classification challenge is to tell apart two types of events:

| Class | Description |
|---|---|
| **Gamma (signal)** | Showers produced by gamma rays — the astrophysical signal of interest |
| **Hadron (background)** | Showers produced by ordinary cosmic rays — noise to be rejected |

Misclassifying a hadron as gamma (false positive) introduces contamination into the signal sample, which is more harmful than missing a true gamma event. For this reason, **ROC-AUC** is the primary evaluation metric rather than accuracy alone.

---

## Dataset

**Source:** [UCI Machine Learning Repository — MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

| Property | Value |
|---|---|
| Instances | 19,020 |
| Features | 10 (all continuous — Hillas parameters) |
| Target | Binary: gamma (1) / hadron (0) |
| Missing values | None |
| Class distribution | Gamma: 12,332 (64.8%) · Hadron: 6,688 (35.2%) |

### Hillas Parameters

| Feature | Description |
|---|---|
| `fLength` | Major axis of the shower ellipse (mm) |
| `fWidth` | Minor axis — gamma showers tend to be narrower |
| `fSize` | log₁₀ of total photon count — correlated with energy |
| `fConc` | Ratio of two brightest pixels to fSize — light concentration |
| `fConc1` | Ratio of brightest pixel to fSize |
| `fAsym` | Distance from brightest pixel to ellipse centre (mm) |
| `fM3Long` | Third moment along major axis — longitudinal asymmetry |
| `fM3Trans` | Third moment along minor axis — transverse asymmetry |
| `fAlpha` | Angle of major axis relative to source direction (°) |
| `fDist` | Distance of ellipse centre from origin (mm) |

The most discriminative features are **fAlpha** (gamma events peak sharply near 0°, hadrons are uniformly distributed 0°–90°), **fLength**, **fWidth**, and **fConc**.

---

## Project Workflow

```
1. Exploratory Data Analysis
   ├── Class distribution (countplot)
   ├── Feature histograms (gamma vs hadron overlay)
   ├── Correlation heatmap
   └── Pairplot of selected Hillas parameters

2. Data Preprocessing
   ├── Missing value check (dataset is complete)
   ├── Train/test split — 75% / 25%, stratified
   └── Z-score standardisation (StandardScaler fit on train only)

3. Model Training
   ├── 5-fold stratified cross-validation (3-fold for SVMs)
   ├── Hyperparameter tuning: GridSearchCV (Naive Bayes)
   │                          RandomizedSearchCV (all others)
   └── Optimisation metric: ROC-AUC

4. Evaluation
   ├── Classification report (precision, recall, F1, accuracy)
   ├── Confusion matrices
   ├── ROC curves and AUC
   ├── Learning curves
   ├── Feature importance plots
   ├── Decision boundaries (PCA projection)
   ├── Training vs validation metrics
   └── Training time comparison
```

---

## Models

| Model | Type | Notes |
|---|---|---|
| **Naive Bayes** (Gaussian) | Probabilistic | Assumes feature independence — baseline model |
| **Logistic Regression** | Linear | Tuned: `C`, `penalty`, `solver`, `class_weight` |
| **Decision Tree** | Non-linear | Tuned: `max_depth`, `min_samples_split/leaf`, `criterion` |
| **Random Forest** | Ensemble | Tuned: `n_estimators`, `max_depth`, `min_samples_*`, `max_features` |
| **SVM — Linear kernel** | Linear | Tuned: `C`, `class_weight` |
| **SVM — RBF kernel** | Non-linear | Tuned: `C`, `gamma`, `class_weight` |

All models use `class_weight='balanced'` where supported, to compensate for the moderate class imbalance (65/35 split).

---

## Results

### ROC-AUC Summary

| Model | ROC-AUC | Accuracy | F1-Score | Training Time |
|---|---|---|---|---|
| **Random Forest** | **0.930** | 0.868 | 0.900 | 216 s |
| **SVM — RBF kernel** | 0.925 | 0.866 | 0.898 | 71 s |
| **Decision Tree** | 0.890 | 0.834 | 0.872 | 4 s |
| **Logistic Regression** | 0.845 | 0.789 | 0.835 | 1 s |
| **SVM — Linear** | 0.845 | 0.792 | 0.838 | 199 s |
| **Naive Bayes** | 0.762 | 0.727 | 0.813 | 5 s |

### Key Findings

**Random Forest** is the best overall model (ROC-AUC 0.930). As an ensemble of 150 trees, it captures non-linear interactions between Hillas parameters — patterns that are clearly visible in the pairplot but inaccessible to linear models. Hyperparameter tuning reduced the train/validation gap from 7–9% to 5.5%, confirming good generalisation.

**SVM with RBF kernel** is the best alternative (ROC-AUC 0.925, only 0.005 below Random Forest), with the smallest overfitting gap of all models (1.8%). Its main limitation is computational cost, as the kernel matrix grows quadratically with the number of samples.

**Linear models** (Logistic Regression, Linear SVM) plateau around ROC-AUC 0.845 and show clear underfitting — their linear decision boundary cannot capture the curved relationships visible between features such as `fSize` and `fConc`.

**Naive Bayes** is the weakest model (ROC-AUC 0.762). The feature independence assumption is strongly violated in this dataset, leading to 62% of hadrons being misclassified as gamma events.

**Feature importance** is consistent across models: `fAlpha` is the single most important feature (importance ≈ 0.29–0.37), followed by `fLength`, `fWidth`, and `fSize`. This matches the underlying physics: gamma events come from point-like sources and have very small angles, while hadrons arrive isotropically.

---

## Repository Structure

```
.
├── notebook.ipynb       # Full analysis: EDA, preprocessing, training, evaluation
├── report.pdf           # Written report with methodology and results
└── README.md
```

The notebook is self-contained. All data is loaded directly from the UCI repository using `pandas.read_csv`.

---

## Requirements

- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

---

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/ilaria-casolino/<repo-name>.git
cd <repo-name>
```

2. **Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

3. **Download the dataset**

The notebook loads the dataset automatically from the UCI repository. Make sure you have an internet connection on first run, or download it manually:

```
https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip
```

Place the file `magic04.data` in the same directory as the notebook and update the path in the first code cell if needed.

4. **Run the notebook**

```bash
jupyter notebook notebook.ipynb
```

> ⚠️ **Note:** Training SVMs on the full dataset (14,265 samples) is computationally intensive. The Linear SVM takes approximately 3 minutes and the RBF SVM approximately 1 minute. All other models complete in under 5 seconds.

---

## Author

**Ilaria Casolino** — [GitHub](https://github.com/ilaria-casolino)

Engineering in Computer Science and Artificial Intelligence
Sapienza University of Rome — A.Y. 2025/2026

---

*Dataset source: P. Savicky, J. Kovarik, J. Dedic — MAGIC Gamma Telescope, UCI Machine Learning Repository, 2007.*
