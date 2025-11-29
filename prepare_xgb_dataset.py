# prepare_xgb_dataset.py

import pandas as pd
import numpy as np

INPUT_FILE = "XAUUSD_M15_WITH_SIGNALS.csv"
OUTPUT_NPZ = "xgb_intraday_dataset.npz"

def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Pastikan 'set' & 'target' ada
    assert "set" in df.columns, "Kolom 'set' tidak ditemukan."
    assert "target" in df.columns, "Kolom 'target' tidak ditemukan."

    # Buang bar tanpa target
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    # List kolom yang jelas BUKAN fitur input
    exclude_cols = {
        "time",
        "date",
        "set",
        "target",
        "future_ret",
        "signal",
        "proba_up",
        "dl_signal",
        "dl_proba_up",
    }

    # Ambil hanya kolom numerik yang bukan di exclude
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    print("[INFO] Feature columns used for XGB ({}):".format(len(feature_cols)))
    for c in feature_cols:
        print("  -", c)

    # Split berdasarkan 'set'
    df_train = df[df["set"] == "train"].copy().sort_values("time")
    df_val   = df[df["set"] == "val"].copy().sort_values("time")
    df_test  = df[df["set"] == "test"].copy().sort_values("time")

    print("\n[INFO] Rows per set:")
    print("  Train:", len(df_train))
    print("  Val  :", len(df_val))
    print("  Test :", len(df_test))

    # X dan y per set
    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train["target"].to_numpy(dtype=np.int64)

    X_val   = df_val[feature_cols].to_numpy(dtype=np.float32)
    y_val   = df_val["target"].to_numpy(dtype=np.int64)

    X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test  = df_test["target"].to_numpy(dtype=np.int64)

    print("\n[INFO] Shapes:")
    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val  :", X_val.shape,   " y_val  :", y_val.shape)
    print("  X_test :", X_test.shape,  " y_test :", y_test.shape)

    # Simpan ke NPZ
    np.savez_compressed(
        OUTPUT_NPZ,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=np.array(feature_cols),
    )

    print("\n[OK] Saved XGB dataset ->", OUTPUT_NPZ)


if __name__ == "__main__":
    main()
