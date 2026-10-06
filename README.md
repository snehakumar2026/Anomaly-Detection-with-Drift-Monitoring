# AIML: Fraud/Anomaly Detection with Drift Monitoring

An end-to-end fraud detection pipeline addressing severe class imbalance, business cost asymmetries, and continuous feature drift monitoring.

---

## 1. Project Overview & Scenario
In real-world fintech applications, transaction patterns continuously shift due to evolving fraud strategies and seasonal customer behaviors. A static classifier trained once inevitably decays over time. This project deploys a machine learning system that:
- Accurately identifies fraudulent transactions under severe class imbalance (~0.17% fraud rate).
- Balances False Negatives vs. False Positives using asymmetric business cost optimization.
- Actively tracks data drift between training and incoming transaction streams to alert when retraining is required.

---

## 2. Technical Decisions & Approach

### Extreme Class Imbalance Handling
- **Dataset:** 284,807 total transactions with only 492 fraud cases (0.17% positive class).
- **Choice:** Used **Class Weighting (`class_weight='balanced'`)** in Random Forest rather than naive oversampling.
- **Justification:** Naive oversampling risks overfitting and distorts probability calibrations. Class weighting adjusts the loss function to heavily penalize fraud misclassification without artificially inflating dataset size.

### Supervised vs. Unsupervised Modeling
- **Supervised Model:** Random Forest Classifier (balanced weights). Learns labeled feature boundaries.
- **Unsupervised Anomaly Detector:** Isolation Forest (`contamination=0.002`). Identifies anomalies purely as outliers without label supervision.
- **Comparison:** The supervised Random Forest vastly outperformed the Isolation Forest (PR-AUC 0.8560 vs. 0.1548), demonstrating that supervised models are essential when labeled data exists, while unsupervised methods serve as fallbacks for unlabeled streams.

### Evaluation Metric: PR-AUC over ROC-AUC
- Standard accuracy yields >99.8% by predicting zero fraud, masking total model failure.
- ROC-AUC incorporates True Negatives heavily in the denominator ($FPR = FP / (FP + TN)$), producing deceptively high scores (>0.95) even with poor fraud capture.
- **PR-AUC (Precision-Recall AUC)** measures the direct trade-off between Precision (positive predictive value) and Recall (fraud detection rate), making it the gold standard for severe imbalance.

### Cost-Sensitive Threshold Optimization
- **Default Threshold (0.50):** Misses too many rare fraud cases.
- **Cost Formulation:** Assigned $Cost = 10 \times FN + 1 \times FP$ (assuming missed fraud is 10x more detrimental than a blocked legitimate user).
- **Outcome:** The optimal threshold shifted down to **0.2461**, boosting recall and minimizing net financial loss.

### Drift Detection & Alerting Logic
- Split data chronologically (first 70% historical training, last 30% simulated future transactions).
- Calculated the standardized mean absolute difference across feature distributions.
- **Baseline Drift:** Measured on split training slices (~0.05).
- **Observed Drift:** Reached **0.1476** on the later window, exceeding the `0.10` alert threshold and successfully triggering an automated retraining flag.

---

## 3. Results Summary

| Model / Experiment | Metric | Value |
| :--- | :--- | :--- |
| Random Forest (Validation) | PR-AUC | **0.8560** |
| Isolation Forest (Validation) | PR-AUC | **0.1548** |
| Random Forest (Later Window) | PR-AUC | **0.8216** (Performance degradation) |
| Cost-Optimal Threshold | Threshold | **0.2461** (Minimizes $10 \cdot FN + 1 \cdot FP$) |
| Feature Drift Score | Drift Score | **0.1476** (Exceeds 0.10 threshold) |
| Pipeline Alert Status | Alert | **⚠️ Retraining Required** |

---

## 4. Setup & Installation

### Prerequisites
- Python 3.9+
- Git

### Installation
1. Clone this repository:
   ```bash
   git clone <YOUR-GITHUB-REPO-URL>
   cd fraud_drift_project

