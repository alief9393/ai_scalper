"""
train_xgb_v4_realistic_fixed.py

Train XGBoost multi-class classifier (SELL / NO_TRADE / BUY)
menggunakan label_realistic dari:
  xau_M5_ml_dataset_ADV_realistic_labels.csv

Label:
  - 0 = SELL
  - 1 = NO_TRADE
  - 2 = BUY

Perbedaan utama vs v4 lama:
- TIDAK pakai train_test_split(shuffle=True)
- Pakai SEQUENTIAL SPLIT (time-series friendly):
    80% data awal  -> TRAIN
    20% data akhir -> VALIDATION

Output:
  xgb_scalping_model_v4_realistic_fixed.pkl
  berisi:
    - model         : XGBClassifier
    - scaler        : StandardScaler
    - feature_names : list nama kolom fitur
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ======== CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV_realistic_labels.csv"
OUTPUT_PKL = "xgb_scalping_model_v4_realistic_fixed.pkl"

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

    # Pastikan time urut
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # Pilih fitur numeric selain EXCLUDE_COLS
    feature_names = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and np.issubdtype(df[c].dtype, np.number)
    ]

    print(f"[INFO] Total rows: {len(df)}")
    print(f"[INFO] Num features: {len(feature_names)}")
    print(f"[INFO] First 10 features: {feature_names[:10]}")

    X = df[feature_names].values
    y = df[LABEL_COL].values

    # Info distribusi label
    unique, counts = np.unique(y, return_counts=True)
    print("\n=== LABEL DISTRIBUTION (Full) ===")
    print(dict(zip(unique, counts)))  # 0,1,2

    # ===== TIME-SERIES SEQUENTIAL SPLIT (NO SHUFFLE) =====
    test_ratio = 0.2
    split_idx = int(len(X) * (1.0 - test_ratio))

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_val = X[split_idx:]
    y_val = y[split_idx:]

    print(f"\n[INFO] Train size: {len(X_train)}, Val size: {len(X_val)}")
    if "time" in df.columns:
        t_start_train = df.iloc[0]["time"]
        t_end_train   = df.iloc[split_idx - 1]["time"]
        t_start_val   = df.iloc[split_idx]["time"]
        t_end_val     = df.iloc[len(df) - 1]["time"]
        print(f"[INFO] Train period: {t_start_train}  →  {t_end_train}")
        print(f"[INFO] Val   period: {t_start_val}  →  {t_end_val}")
    print("[INFO] Sequential split applied (no shuffling, time-order preserved).")

    # ===== SCALING =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    # ===== TRAIN XGBOOST MODEL =====
    print("\n[INFO] Training XGB v4 FIXED (realistic labels)...")
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

    # ===== EVALUATION =====
    y_pred = model.predict(X_val_scaled)
    print("\n=== CLASSIFICATION REPORT (XGB v4 realistic FIXED) ===")
    print(classification_report(y_val, y_pred, digits=3))

    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_val, y_pred))

    # Save model + scaler + feature names
    out = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "info": "XGB v4 realistic with sequential time-series split (no shuffle)",
        "split_index": split_idx,
        "test_ratio": test_ratio,
    }
    joblib.dump(out, OUTPUT_PKL)
    print(f"\n[SAVED] XGB model (fixed) → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
