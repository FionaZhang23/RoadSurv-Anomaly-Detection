# RoadSurv-Anomaly-Detection

## 📌 Overview

This project is a proof-of-concept system for **anomaly detection in road surveillance imagery**, focused on distinguishing non-dog images from a large collection of dog-only images. It was developed for CSC373/673: Data Mining at Wake Forest University (Spring 2025).

The task involves evaluating and comparing multiple detection methods (naïve, baseline, and advanced models), reporting performance using AUC (Area Under ROC Curve), and saving a reproducible pipeline.

---

## 🎯 Objectives

- Train models to distinguish anomalous (non-dog) images from normal (dog) images
- Evaluate models using AUC score
- Engineer visual features (RGB stats, histograms, image embeddings)
- Build a scikit-learn compatible pipeline and export with `joblib`
- Save output scores and reports to `output/` directory

  The solution was developed as part of CSC373/673: Data Mining at Wake Forest University (Spring 2025).
---

## 🧠 Models Used

| Model             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Naïve**         | Assigns random scores to each image                                         |
| **Baseline**      | Computes distance between each image's RGB mean and the training mean RGB  |
| **Advanced**      | Isolation Forest, or embedding-based outlier detection using cleanlab       |

---

## 🧪 Evaluation Metric

- AUC (Area Under ROC Curve), implemented via `sklearn.metrics.roc_auc_score`
- Goal: maximize AUC by increasing separation between in-domain (dogs) and outliers (non-dogs)

---

## 🧠 Methodology

| Phase | Task                                           |
|-------|------------------------------------------------|
| I     | Preprocess raw images and extract features     |
| II    | Train naïve and baseline anomaly detectors     |
| III   | Engineer embeddings/histograms for advanced models |
| IV    | Evaluate and compare AUC scores                |
| V     | Save final model pipeline with `joblib.dump`   |

---

## 📂 Project Structure

```bash
data/                             # Raw data files
└── ground_truth.txt              # Labels (0=dog, 1=anomaly) for evaluation

output/                           # Reports and model results
├── data_quality_report.txt       # Optional summary of input data
└── output_report.txt             # AUC scores for all models (naive, baseline, best)

scripts/                          # All Python scripts
├── __pycache__/                  # Bytecode cache
├── assignment_3.py               # Main entry script (must run final model here)
├── check_quality.py              # Optional data diagnostics
├── code_snippets.py              # Provided image helper functions
├── Transformer.py                # Custom feature transformers
└── utils.py                      # Helper functions (e.g., scoring, plotting)

README.md                         # Project overview and instructions
