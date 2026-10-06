import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# STEP 1: LOAD & INSPECT DATA
# -------------------------------------------------------------
print("--> Loading dataset...")
df = pd.read_csv("data/creditcard.csv")
print(f"Total rows: {len(df)}, Total features: {df.shape[1]}")
print(f"Base fraud rate: {df['Class'].mean() * 100:.3f}%\n")

# -------------------------------------------------------------
# STEP 2: TIME-BASED SPLITTING (SIMULATE DRIFT)
# -------------------------------------------------------------
# Sort chronologically by Time
df = df.sort_values("Time").reset_index(drop=True)

# First 70% is historical training window; last 30% simulates future/incoming data
split_idx = int(0.7 * len(df))
train_window = df.iloc[:split_idx].copy()
later_window = df.iloc[split_idx:].copy()

# Separate features from target
feature_cols = [col for col in df.columns if col not in ["Time", "Class"]]

X_train_full = train_window[feature_cols]
y_train_full = train_window["Class"]

X_later = later_window[feature_cols]
y_later = later_window["Class"]

# Stratified train/validation split within the training window
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=42,
)

# -------------------------------------------------------------
# STEP 3: TRAIN SUPERVISED MODEL (RANDOM FOREST)
# -------------------------------------------------------------
print("--> Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# Calculate Validation PR-AUC
y_val_prob = rf_clf.predict_proba(X_val)[:, 1]
prec_val, rec_val, _ = precision_recall_curve(y_val, y_val_prob)
pr_auc_val = auc(rec_val, prec_val)
print(f"Validation PR-AUC: {pr_auc_val:.4f}\n")

# -------------------------------------------------------------
# STEP 4: TRAIN UNSUPERVISED MODEL (ISOLATION FOREST)
# -------------------------------------------------------------
print("--> Training Isolation Forest (Unsupervised)...")
iso_forest = IsolationForest(
    n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1
)
iso_forest.fit(X_train)

# Anomaly score: invert decision function so higher score = higher fraud likelihood
val_anomaly_scores = -iso_forest.decision_function(X_val)
prec_iso, rec_iso, _ = precision_recall_curve(y_val, val_anomaly_scores)
pr_auc_iso = auc(rec_iso, prec_iso)
print(f"Isolation Forest PR-AUC: {pr_auc_iso:.4f}\n")


# -------------------------------------------------------------
# STEP 5: COST-SENSITIVE THRESHOLD OPTIMIZATION
# -------------------------------------------------------------
def calculate_cost(y_true, y_prob, threshold, cost_fn=10, cost_fp=1):
  y_pred = (y_prob >= threshold).astype(int)
  _, fp, fn, _ = confusion_matrix(y_true, y_pred).ravel()
  return (cost_fn * fn) + (cost_fp * fp)


def get_performance_metrics(y_true, y_prob, threshold):
  y_pred = (y_prob >= threshold).astype(int)
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  return {"TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall}


# Search thresholds between 0.01 and 0.90
candidate_thresholds = np.linspace(0.01, 0.90, 90)
costs = [
    calculate_cost(y_val, y_val_prob, t, cost_fn=10, cost_fp=1)
    for t in candidate_thresholds
]
best_threshold = candidate_thresholds[int(np.argmin(costs))]

print(f"Optimal Threshold (Minimizing Cost): {best_threshold:.3f}")
val_metrics = get_performance_metrics(y_val, y_val_prob, best_threshold)
print(f"Validation metrics at optimal threshold: {val_metrics}\n")

# -------------------------------------------------------------
# STEP 6: EVALUATE ON LATER (FUTURE) WINDOW
# -------------------------------------------------------------
y_later_prob = rf_clf.predict_proba(X_later)[:, 1]
prec_later, rec_later, _ = precision_recall_curve(y_later, y_later_prob)
pr_auc_later = auc(rec_later, prec_later)
later_metrics = get_performance_metrics(y_later, y_later_prob, best_threshold)

print(f"Later-Window PR-AUC: {pr_auc_later:.4f}")
print(f"Later-Window metrics at optimal threshold: {later_metrics}\n")


# -------------------------------------------------------------
# STEP 7: DRIFT MONITORING & ALERTING
# -------------------------------------------------------------
def compute_feature_drift(X_ref, X_new):
  means_ref = X_ref.mean()
  stds_ref = X_ref.std() + 1e-8
  means_new = X_new.mean()
  diff = np.abs((means_new - means_ref) / stds_ref)
  return diff.mean()


# In-sample baseline drift
X_a, X_b = train_test_split(X_train, test_size=0.5, random_state=42)
baseline_drift = compute_feature_drift(X_a, X_b)

# Production drift (Training vs Later window)
production_drift = compute_feature_drift(X_train, X_later)

ALERT_THRESHOLD = 0.10

print("--- DRIFT MONITORING REPORT ---")
print(f"Baseline In-Sample Drift: {baseline_drift:.4f}")
print(f"Production Drift Score:   {production_drift:.4f}")
print(f"Alert Threshold:          {ALERT_THRESHOLD:.4f}")

if production_drift > ALERT_THRESHOLD:
  print(
      "[!] ALERT: Significant drift detected. Triggering model retraining"
      " workflow."
  )
else:
  print("[*] OK: Data distributions remain within expected bounds.")
