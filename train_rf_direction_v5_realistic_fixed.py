"""
train_rf_direction_v5_realistic_fixed.py

Train RandomForest khusus BUY vs SELL
dari dataset dengan label_realistic:
  - 0 = SELL
  - 2 = BUY
(1 = NO_TRADE tidak dipakai di RF)

Perbedaan utama vs v5:
- TIDAK pakai train_test_split(shuffle=True)
- Pakai SEQUENTIAL SPLIT (time-series friendly):
    80% data awal  -> TRAIN
    20% data akhir -> VALIDATION

Output:
  rf_direction_model_v5_realistic_fixed.pkl
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ======== CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV_realistic_labels.csv"
OUTPUT_PKL = "rf_direction_model_v5_realistic_fixed.pkl"

LABEL_COL = "label_realistic"
EXCLUDE_COLS = {
    "time",
    LABEL_COL,
    "real_buy_pips",
    "real_sell_pips",
}


def main():
    print("[INFO] Loading dataset with realistic labels...")
    df = pd.read_csv(CSV_INPUT)

    # Pastikan time terurut
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # Ambil hanya BUY (2) & SELL (0) untuk RF direction
    df_dir = df[df[LABEL_COL].isin([0, 2])].copy()
    print(f"[INFO] Directional subset size: {len(df_dir)}")

    # Fitur numeric selain EXCLUDE_COLS
    feature_names = [
        c for c in df_dir.columns
        if c not in EXCLUDE_COLS and np.issubdtype(df_dir[c].dtype, np.number)
    ]

    print(f"[INFO] Num features: {len(feature_names)}")
    print(f"[INFO] First 10 features: {feature_names[:10]}")

    X = df_dir[feature_names].values
    y = df_dir[LABEL_COL].values   # 0 = SELL, 2 = BUY

    # Remap ke 0 / 1 (lebih clean)
    # 0 = SELL, 1 = BUY
    y_bin = np.where(y == 2, 1, 0)

    # ======== TIME-SERIES SEQUENTIAL SPLIT (NO SHUFFLE) ========
    test_ratio = 0.2
    split_idx = int(len(X) * (1.0 - test_ratio))

    X_train = X[:split_idx]
    y_train = y_bin[:split_idx]

    X_val = X[split_idx:]
    y_val = y_bin[split_idx:]

    print(f"[INFO] Train size: {len(X_train)}, Val size: {len(X_val)}")
    if "time" in df_dir.columns:
        t_start_train = df_dir.iloc[0]["time"]
        t_end_train = df_dir.iloc[split_idx - 1]["time"]
        t_start_val = df_dir.iloc[split_idx]["time"]
        t_end_val = df_dir.iloc[len(df_dir) - 1]["time"]
        print(f"[INFO] Train period: {t_start_train}  →  {t_end_train}")
        print(f"[INFO] Val   period: {t_start_val}  →  {t_end_val}")
    print("[INFO] Sequential split applied (no shuffling, time-order preserved).")

    # ======== TRAIN RF MODEL ========
    print("[INFO] Training RandomForest v5 FIXED (direction realistic)...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    rf.fit(X_train, y_train)

    # ======== VALIDATION REPORT ========
    y_pred = rf.predict(X_val)
    print("\n=== CLASSIFICATION REPORT (RF v5 realistic FIXED) ===")
    print(classification_report(y_val, y_pred, digits=3))

    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_val, y_pred))

    # (Optional) lihat class distribution
    unique, counts = np.unique(y_bin, return_counts=True)
    print("\n=== LABEL DISTRIBUTION (Full Directional Set) ===")
    print(dict(zip(unique, counts)))  # 0=SELL, 1=BUY

    # Save model + metadata
    out = {
        "model": rf,
        "feature_names": feature_names,
        "label_mapping": {0: "SELL", 1: "BUY"},
        "split_index": split_idx,
        "test_ratio": test_ratio,
        "info": "RF direction v5 realistic with sequential time-series split (no shuffle)",
    }
    joblib.dump(out, OUTPUT_PKL)
    print(f"\n[SAVED] RF direction model (fixed) → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
