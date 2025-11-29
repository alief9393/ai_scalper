# apply_xgb_signals_xgb.py

import numpy as np
import pandas as pd
import xgboost as xgb

INPUT_CSV  = "XAUUSD_M15_WITH_SIGNALS.csv"
DATA_NPZ   = "xgb_intraday_dataset.npz"
MODEL_PATH = "xgb_intraday.model"
META_PATH  = "xgb_intraday_meta.npz"
OUTPUT_CSV = "XAUUSD_M15_WITH_XGB_SIGNALS_FULL.csv"

THR_LONG  = 0.70
THR_SHORT = 0.30  # 1 - THR_LONG kalau mau simetris

def main():
    # Load DF
    df = pd.read_csv(INPUT_CSV, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if "set" not in df.columns:
        raise ValueError("Kolom 'set' tidak ditemukan di CSV.")

    # Load meta dataset (feature_names) supaya konsisten
    data = np.load(DATA_NPZ, allow_pickle=True)
    feature_names = list(data["feature_names"])

    # Pastikan semua fitur ada di DF
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in DF: {missing}")

    # Build DMatrix untuk semua bar (boleh full, nanti backtest tetap pakai set='test')
    X_all = df[feature_names].to_numpy(dtype=np.float32)
    dmat_all = xgb.DMatrix(X_all, feature_names=feature_names)

    # Load model
    bst = xgb.Booster()
    bst.load_model(MODEL_PATH)
    print(f"[INFO] Loaded XGB model from {MODEL_PATH}")

    proba_up = bst.predict(dmat_all)
    print("[INFO] XGB proba shape:", proba_up.shape)

    # Tambah kolom proba & signal
    df["xgb_proba_up"] = proba_up
    df["xgb_signal"] = 0

    df.loc[df["xgb_proba_up"] >= THR_LONG,  "xgb_signal"] = 1
    df.loc[df["xgb_proba_up"] <= THR_SHORT, "xgb_signal"] = -1

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved -> {OUTPUT_CSV}")

    # Stats di test set
    if "set" in df.columns:
        df_test = df[df["set"] == "test"].copy()
        print("[INFO] xgb_signal distribution (test set):")
        print(df_test["xgb_signal"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
