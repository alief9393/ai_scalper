# train_ml_model_timesplit.py
"""
Train RandomForest untuk XAUUSD M5 dengan TIME-BASED split (no leakage)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# ============ CONFIG ============
CSV_INPUT = "xau_M5_ml_dataset.csv"
MODEL_OUTPUT = "rf_scalping_model_timesplit.pkl"
RANDOM_STATE = 42
TEST_RATIO = 0.2
CONF_TRADE_THRESHOLD = 0.50
# ===============================

print("[INFO] Loading ML dataset...")
df = pd.read_csv(CSV_INPUT)

if "label" not in df.columns:
    raise ValueError("Label column not found in dataset!")

# Pastikan sort by time
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

print(f"[INFO] Dataset shape: {df.shape}")

y = df["label"]
X = df.drop("label", axis=1)

# Drop kolom waktu dan non-numeric
for col in ["time", "time_local", "Unnamed: 0"]:
    if col in X.columns:
        X = X.drop(columns=[col])

non_numeric = X.select_dtypes(include=["object"]).columns.tolist()
if non_numeric:
    X = X.drop(columns=non_numeric)

print(f"[INFO] Final feature columns ({len(X.columns)}): {X.columns.tolist()}")

# TIME-BASED SPLIT
n = len(df)
split_idx = int(n * (1 - TEST_RATIO))

X_train = X.iloc[:split_idx].copy()
y_train = y.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:].copy()

print(f"[INFO] Time-based split: train={X_train.shape}, test={X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weight biar BUY/SELL lebih diperhatikan
class_weight = {-1: 3.0, 0: 1.0, 1: 3.0}

print("[INFO] Training RandomForest...")
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

# Baseline test performance (no filter)
print("\n[RESULT] Baseline Test Performance (time-based split):")
y_pred = model.predict(X_test_scaled)
labels_order = [-1, 0, 1]
target_names = ["SELL(-1)", "NO_TRADE(0)", "BUY(1)"]

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels_order,
        target_names=target_names,
        digits=4
    )
)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=labels_order))

# Save model + scaler + split index
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "feature_names": X.columns.tolist(),
        "split_idx": split_idx,
        "conf_threshold": CONF_TRADE_THRESHOLD,
    },
    MODEL_OUTPUT
)
print(f"\n[SAVED] Model & scaler saved to: {MODEL_OUTPUT}")
print(f"[INFO] split_idx (start of test section): {split_idx}")
