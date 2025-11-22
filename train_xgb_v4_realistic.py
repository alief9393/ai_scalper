"""
train_xgb_v4_realistic.py

Train XGBoost multi-class classifier (SELL / NO_TRADE / BUY)
menggunakan label_realistic dari:
  xau_M5_ml_dataset_ADV_realistic_labels.csv

Output:
  xgb_scalping_model_v4_realistic.pkl
  berisi:
    - model       : XGBClassifier
    - scaler      : StandardScaler
    - feature_names : list nama kolom fitur
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ======== CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV_realistic_labels.csv"
OUTPUT_PKL = "xgb_scalping_model_v4_realistic.pkl"

LABEL_COL = "label_realistic"   # 0=SELL,1=NO_TRADE,2=BUY

# Kolom yang TIDAK dijadikan fitur
EXCLUDE_COLS = {
    "time",
    LABEL_COL,
    "real_buy_pips",
    "real_sell_pips",
}


def main():
    print("[INFO] Loading dataset with realistic labels...")
    df = pd.read_csv(CSV_INPUT)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # Pilih fitur (semua numeric kecuali yang di-exclude)
    feature_names = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Num features: {len(feature_names)}")
    print(f"[INFO] Features: {feature_names[:10]} ...")

    X = df[feature_names].values
    y = df[LABEL_COL].values

    # Split train/val (pakai shuffle=False kalau mau strict time-based)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # XGBoost model (hyperparam bisa lo tuning belakangan)
    print("[INFO] Training XGB v4 (realistic labels)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    # Evaluation
    y_pred = model.predict(X_val_scaled)
    print("\n=== CLASSIFICATION REPORT (XGB v4 realistic) ===")
    print(classification_report(y_val, y_pred, digits=3))

    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_val, y_pred))

    # Save
    out = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    joblib.dump(out, OUTPUT_PKL)
    print(f"\n[SAVED] XGB model → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
