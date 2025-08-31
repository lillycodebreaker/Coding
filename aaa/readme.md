# AAA Life – Lapse Prediction Mock Case

This repository contains a single, self-contained Python script for an interview-style case study:
**`aaa_lapse_model.py`**. It demonstrates an end-to-end workflow for predicting **12-month policy lapses**
and turning those predictions into **actionable agent/customer briefs**.

---

## ✨ What the script does

- Loads a **CSV dataset** (or **generates synthetic data** if none is provided).
- Builds preprocessing with **imputation** and **categorical encoding**.
- Trains two models: **Logistic Regression** (baseline) and **HistGradientBoosting** (tabular SOTA in sklearn).
- Evaluates with **ROC-AUC** and **PR-AUC** (better for class imbalance).
- Selects an **operating threshold by cost/utility** (benefit of retention vs. cost of outreach).
- Exports **feature importances**, **top-risk briefs** (agent + customer), and a **summary.json**.

Outputs are written to `aaa_outputs/`.

---

## 🧱 Requirements

- Python **3.9+**
- Packages: `numpy`, `pandas`, `scikit-learn`

> Install (user env):  
> ```bash
> pip install -U numpy pandas scikit-learn
> ```

---

## 🚀 Quick start

**Option A — Synthetic data**
```bash
python aaa_lapse_model.py

