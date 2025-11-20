"""
train_ml_model_v3_balanced.py

Versi balanced:
- Pakai dataset advanced: xau_M5_ml_dataset_ADV.csv
- Time-based split (70% train, 30% test)
- SMOTE oversampling untuk seimbangkan kelas (0,1,2)
- Train RandomForest & XGBoost di data balanced
- Simpan model v3_balanced sebagai: rf_scalping_model_v3_balanced.pkl dan xgb_scalping_model_v3_balanced.pkl
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

import joblib

# ===== LOAD DATASET =====
df = pd.read_csv("xau_M5_ml_dataset_ADV.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Label mapping: from original [-1, 0, 1] -> [0, 1, 2]
label_mapping = {-1: 0, 0: 1, 1: 2}
df["label"] = df["label"].map(label_mapping)

X = df.drop(columns=["time", "label"])
y = df["label"]

print(f"[INFO] Dataset shape: {X.shape}, Features: {len(X.columns)}")

# ===== TIME-BASED SPLIT =====
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

# ===== SCALING =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== SMOTE OVERSAMPLING =====
print("\n[INFO] BEFORE SMOTE class distribution (y_train):")
print(y_train.value_counts())

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("\n[INFO] AFTER SMOTE class distribution (y_train_bal):")
print(pd.Series(y_train_bal).value_counts())

# ===== RANDOM FOREST (BALANCED) =====
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight=None,   # class_weight off karena sudah SMOTE
    random_state=42,
    n_jobs=-1
)

print("\n[TRAINING] RandomForest (SMOTE balanced)...")
rf_model.fit(X_train_bal, y_train_bal)
rf_pred = rf_model.predict(X_test_scaled)

print("\n[RF BALANCED] Classification Report:")
print(classification_report(y_test, rf_pred))

print("[RF BALANCED] Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# ===== XGBOOST (BALANCED) =====
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

print("\n[TRAINING] XGBoost (SMOTE balanced)...")
xgb_model.fit(X_train_bal, y_train_bal)
xgb_pred = xgb_model.predict(X_test_scaled)

print("\n[XGB BALANCED] Classification Report:")
print(classification_report(y_test, xgb_pred))

print("[XGB BALANCED] Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

# ===== SAVE MODELS =====
joblib.dump({
    "model": rf_model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "conf_threshold": 0.5
}, "rf_scalping_model_v3_balanced.pkl")

joblib.dump({
    "model": xgb_model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "conf_threshold": 0.5
}, "xgb_scalping_model_v3_balanced.pkl")

print("\n[SAVED] rf_scalping_model_v3_balanced.pkl & xgb_scalping_model_v3_balanced.pkl")
