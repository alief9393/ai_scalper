"""
train_ml_model_v3.py
Model training menggunakan dataset advanced:
- Menggunakan XGBoost dan RandomForest (perbandingan)
- Balanced classes
- Time-based split
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Load dataset
df = pd.read_csv("xau_M5_ml_dataset_ADV.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

label_mapping = {-1: 0, 0: 1, 1: 2}
df["label"] = df["label"].map(label_mapping)

# Drop 'time' → tidak dilatih
X = df.drop(columns=["time", "label"])
y = df["label"]

print(f"[INFO] Dataset shape: {X.shape}, Features: {len(X.columns)}")

# Time-based split (70% awal data = training, 30% akhir = test)
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== RandomForest Model =====
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\n[TRAINING] RandomForest with advanced features...")
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print("\n[RF] Classification Report:")
print(classification_report(y_test, rf_pred))

# ===== XGBoost Model =====
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

print("\n[TRAINING] XGBoost with advanced features...")
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)

print("\n[XGBoost] Classification Report:")
print(classification_report(y_test, xgb_pred))

# ===== Save models =====
joblib.dump({
    "model": rf_model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "conf_threshold": 0.5
}, "rf_scalping_model_v3.pkl")

joblib.dump({
    "model": xgb_model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "conf_threshold": 0.5
}, "xgb_scalping_model_v3.pkl")

print("\n[SAVED] rf_scalping_model_v3.pkl & xgb_scalping_model_v3.pkl")
