"""
train_rf_direction_v5_realistic.py

Train RandomForest khusus BUY vs SELL
dari dataset dengan label_realistic:
  - 0 = SELL
  - 2 = BUY
(1 = NO_TRADE tidak dipakai di RF)

Output:
  rf_direction_model_v5_realistic.pkl
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ======== CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV_realistic_labels.csv"
OUTPUT_PKL = "rf_direction_model_v5_realistic.pkl"

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

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    # Ambil hanya BUY (2) & SELL (0)
    df_dir = df[df[LABEL_COL].isin([0, 2])].copy()
    print(f"[INFO] Directional subset size: {len(df_dir)}")

    feature_names = [
        c for c in df_dir.columns
        if c not in EXCLUDE_COLS and np.issubdtype(df_dir[c].dtype, np.number)
    ]

    print(f"[INFO] Num features: {len(feature_names)}")
    print(f"[INFO] Features: {feature_names[:10]} ...")

    X = df_dir[feature_names].values
    y = df_dir[LABEL_COL].values   # 0 = SELL, 2 = BUY

    # Optional: remap to 0 / 1 for cleaner RF
    y_bin = np.where(y == 2, 1, 0)  # 0=SELL, 1=BUY

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_bin, test_size=0.2, shuffle=False, random_state=42
    )

    print("[INFO] Training RandomForest v5 (direction realistic)...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=50,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    print("\n=== CLASSIFICATION REPORT (RF v5 realistic) ===")
    print(classification_report(y_val, y_pred, digits=3))

    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_val, y_pred))

    # Save (ingat: ini binary 0/1, nanti di hybrid kita mapping balik ke 0/2)
    out = {
        "model": rf,
        "feature_names": feature_names,
        "label_mapping": {0: "SELL", 1: "BUY"},
    }
    joblib.dump(out, OUTPUT_PKL)
    print(f"\n[SAVED] RF direction model → {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
