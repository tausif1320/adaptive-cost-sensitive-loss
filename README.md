# Adaptive Cost-Sensitive Learning for Extremely Imbalanced Data

## One-line summary (what this project actually does)

This project shows how **optimizing for the wrong metric can hide real losses**, and how an adaptive, cost-aware learning approach can **reduce real-world decision cost** in highly imbalanced problems like fraud detection.

---

## Why this project matters (the real problem)

In many real-world classification systems, **mistakes are not equally expensive**.

For example, in credit card fraud detection:
- A **False Negative** (missing fraud) can directly cause financial loss.
- A **False Positive** (flagging a genuine transaction) is usually just an inconvenience.

Despite this, most machine learning models are still:
- trained using accuracy or log-loss,
- evaluated using ROC-AUC,
- or adjusted using static class weights.

These approaches often look good on paper but fail to answer a simple business question:

> *How much does this model actually cost when it makes mistakes?*

This project was built to address that gap.

---

## Key idea (in simple terms)

Instead of treating all errors equally, this project trains models to:

1. **Care more about costly mistakes**  
   Errors on fraud cases are penalized more than errors on normal transactions.

2. **Focus increasingly on hard cases over time**  
   Samples that remain difficult across training epochs gradually receive higher importance.

This is implemented through an **Adaptive Cost-Sensitive Loss** that dynamically combines:
- business cost (FN ≫ FP),
- and sample hardness tracked using an exponential moving average.

The goal is not higher accuracy.  
The goal is **lower real-world cost**.

---

## Dataset

- Credit Card Fraud Detection dataset (Kaggle)
- Binary classification: fraud vs non-fraud
- Extreme imbalance: ~0.17% fraud cases
- A setting where accuracy alone is misleading

The dataset is not included in the repository due to size constraints.
Place it at:
data/raw/creditcard.csv

## Project structure

adaptive-cost-sensitive-loss/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_data_overview.ipynb # EDA & imbalance analysis
│ ├── 02_baseline_models.ipynb # Classical baselines
│ ├── 03_torch_baseline.ipynb # Neural models + adaptive loss
│ └── 04_results_summary.ipynb # Final comparison & visuals
├── src/
│ ├── data/
│ ├── models/
│ ├── losses/
│ └── utils/
├── experiments/
│ └── final_model_summary.csv
├── requirements.txt
└── README.md



---

## Modeling approach (what was actually done)

### 1. Classical baselines
Strong reference models were implemented, including:
- Logistic Regression
- Random Forest
- Cost-weighted variants
- Oversampling-based variants

These models provide realistic baselines rather than weak comparisons.

---

### 2. Neural baseline
A simple PyTorch MLP was trained using:
- Binary Cross-Entropy loss
- Class-weighted BCE

These approaches improve recall but rely on **static weighting**, which does not adapt during training.

---

### 3. Adaptive Cost-Sensitive Loss (proposed)

The proposed loss function introduces two ideas:

**Cost awareness**
- Fraud-related errors are penalized more heavily.
- Training aligns directly with business priorities.

**Hardness awareness**
- Some samples remain difficult across epochs.
- An exponential moving average of per-sample loss tracks this difficulty.
- Persistently hard samples gradually influence training more.

This creates a learning signal that **adapts over time**, rather than remaining fixed.

---

## Evaluation strategy (what really matters)

Models are evaluated using:
- Precision, Recall, F1-score
- ROC-AUC and PR-AUC
- **Business Cost**:

Total Cost = 10 × False Negatives + 1 × False Positives


Models are compared primarily using **total cost**, not accuracy.

---

## Real-world impact (numbers that matter)

Compared to a strong Random Forest baseline, the adaptive model achieved:

- **~23% reduction in false negatives**
- **~2–3% lower total business cost**
- **~87% recall** under extreme class imbalance
- A controlled increase in false positives, resulting in net cost reduction

In high-volume transaction systems, even small percentage improvements in missed fraud detection can translate into significant financial savings.

---



## How to run

```bash
git clone https://github.com/tausif1320/adaptive-cost-sensitive-loss.git
cd adaptive-cost-sensitive-loss
pip install -r requirements.txt
