# train_ml_model.py
"""
Step 4: Trade-oriented training & evaluation
- RandomForest dengan class_weight
- Confidence-based filtering untuk sinyal BUY/SELL
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# ============ CONFIG ============
CSV_INPUT = "xau_M5_ml_dataset.csv"
MODEL_OUTPUT = "rf_scalping_model_v2.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Threshold confidence untuk eksekusi trade
CONF_TRADE_THRESHOLD = 0.50   # 60% confidence minimal
# ===============================

print("[INFO] Loading ML dataset...")
df = pd.read_csv(CSV_INPUT)

if "label" not in df.columns:
    raise ValueError("Label column not found in dataset!")

print(f"[INFO] Original columns: {df.columns.tolist()}")
print(f"[INFO] Dataset shape: {df.shape}")

y = df["label"]
X = df.drop("label", axis=1)

# Drop kolom waktu kalau ada
for col in ["time", "time_local", "Unnamed: 0"]:
    if col in X.columns:
        print(f"[INFO] Dropping non-numeric column: {col}")
        X = X.drop(columns=[col])

# Drop kolom non-numerik lain (jaga-jaga)
non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
if non_numeric_cols:
    print(f"[INFO] Dropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(columns=non_numeric_cols)

print(f"[INFO] Final feature columns ({len(X.columns)}): {X.columns.tolist()}")
print(f"[INFO] Cleaned feature shape: {X.shape}")

# Split train/test
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print("[INFO] Data split complete:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============ Train model dengan class_weight ============
print("[INFO] Training RandomForest model with class_weight...")

class_weight = {
    -1: 3.0,   # SELL lebih dibobot
    0: 1.0,    # NO_TRADE normal
    1: 3.0     # BUY lebih dibobot
}

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=18,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight=class_weight
)

model.fit(X_train_scaled, y_train)

# ============ Baseline evaluasi (tanpa filter) ============
print("\n[RESULT] Baseline Performance (no confidence filter):")
y_pred = model.predict(X_test_scaled)

labels_order = [-1, 0, 1]
target_names = ["SELL(-1)", "NO_TRADE(0)", "BUY(1)"]

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=labels_order,
        target_names=target_names,
        digits=4
    )
)

print("\nConfusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred, labels=labels_order))

# ============ Trade-oriented evaluation (confidence filter) ============
print(f"\n[RESULT] Trade-oriented Evaluation (CONF_TRADE_THRESHOLD={CONF_TRADE_THRESHOLD})")

proba = model.predict_proba(X_test_scaled)
classes = list(model.classes_)

# map class -> index
class_to_idx = {c: i for i, c in enumerate(classes)}

y_pred_filtered = []

for i in range(len(y_test)):
    base_pred = y_pred[i]
    base_pred_idx = class_to_idx[base_pred]
    base_conf = proba[i, base_pred_idx]

    # Kalau prediksi BUY/SELL tapi confidence < threshold -> jadikan NO_TRADE
    if base_pred != 0 and base_conf < CONF_TRADE_THRESHOLD:
        y_pred_filtered.append(0)
    else:
        y_pred_filtered.append(base_pred)

y_pred_filtered = np.array(y_pred_filtered)

print("\nClassification Report (with confidence filter):")
print(
    classification_report(
        y_test,
        y_pred_filtered,
        labels=labels_order,
        target_names=target_names,
        digits=4
    )
)

print("\nConfusion Matrix (rows=true, cols=pred, with filter):")
print(confusion_matrix(y_test, y_pred_filtered, labels=labels_order))

# Hitung berapa banyak trade yang dieksekusi (pred != 0)
num_trades = np.sum(y_pred_filtered != 0)
total_samples = len(y_pred_filtered)
print(f"\n[STATS] Number of trade signals (BUY/SELL only): {num_trades} / {total_samples} samples")

# ============ Save model ============
joblib.dump(
    {"model": model, "scaler": scaler, "feature_names": X.columns.tolist()},
    MODEL_OUTPUT
)
print(f"\n[SAVED] Trade-oriented model & scaler saved to: {MODEL_OUTPUT}")
