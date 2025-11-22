# prepare_dl_dataset.py

import pandas as pd
import numpy as np

INPUT_FILE = "XAUUSD_M15_WITH_SIGNALS.csv"
OUTPUT_FILE = "dl_intraday_dataset_seq64.npz"

# Panjang sequence history untuk DL (jumlah candle M15 ke belakang)
SEQ_LEN = 64  # ~16 jam history per sample


def build_sequences(feat_array, target_array, seq_len):
    """
    feat_array: shape (N, num_features)
    target_array: shape (N,)
    seq_len: panjang window
    Return:
        X_seq: (num_samples, seq_len, num_features)
        y_seq: (num_samples,)
    """
    X_list = []
    y_list = []
    n = len(feat_array)

    for i in range(0, n - seq_len):
        x_win = feat_array[i : i + seq_len]
        y = target_array[i + seq_len - 1]  # label di bar terakhir window
        X_list.append(x_win)
        y_list.append(y)

    if not X_list:
        return np.empty((0, seq_len, feat_array.shape[1])), np.empty((0,))

    X_seq = np.stack(X_list, axis=0)
    y_seq = np.array(y_list)
    return X_seq, y_seq


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Pastikan kolom 'set' dan 'target' ada
    assert "set" in df.columns, "Kolom 'set' tidak ditemukan di CSV."
    assert "target" in df.columns, "Kolom 'target' tidak ditemukan di CSV."

    # Drop bar yang ada NaN di target
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    # Feature columns:
    # Semua kolom numerik kecuali yang jelas bukan fitur input
    exclude_cols = [
        "time",
        "date",
        "target",
        "future_ret",
        "set",
        "signal",
        "proba_up",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("[INFO] Feature columns used for DL ({}):".format(len(feature_cols)))
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

    # Ambil fitur dan target per set
    X_train_df = df_train[feature_cols].copy()
    X_val_df   = df_val[feature_cols].copy()
    X_test_df  = df_test[feature_cols].copy()

    y_train = df_train["target"].values
    y_val   = df_val["target"].values
    y_test  = df_test["target"].values

    # Normalisasi pakai mean/std dari TRAIN saja
    feat_mean = X_train_df.mean(axis=0)
    feat_std  = X_train_df.std(axis=0).replace(0, 1.0)  # hindari std=0

    X_train_norm = (X_train_df - feat_mean) / feat_std
    X_val_norm   = (X_val_df   - feat_mean) / feat_std
    X_test_norm  = (X_test_df  - feat_mean) / feat_std

    # Convert ke numpy
    X_train_arr = X_train_norm.values.astype(np.float32)
    X_val_arr   = X_val_norm.values.astype(np.float32)
    X_test_arr  = X_test_norm.values.astype(np.float32)

    y_train_arr = y_train.astype(np.int64)
    y_val_arr   = y_val.astype(np.int64)
    y_test_arr  = y_test.astype(np.int64)

    print("\n[INFO] Building sequences with SEQ_LEN =", SEQ_LEN)

    X_train_seq, y_train_seq = build_sequences(X_train_arr, y_train_arr, SEQ_LEN)
    X_val_seq,   y_val_seq   = build_sequences(X_val_arr,   y_val_arr,   SEQ_LEN)
    X_test_seq,  y_test_seq  = build_sequences(X_test_arr,  y_test_arr,  SEQ_LEN)

    print("[INFO] Sequence shapes:")
    print("  X_train:", X_train_seq.shape, " y_train:", y_train_seq.shape)
    print("  X_val  :", X_val_seq.shape,   " y_val  :", y_val_seq.shape)
    print("  X_test :", X_test_seq.shape,  " y_test :", y_test_seq.shape)

    # Save ke NPZ (bisa diload di PyTorch/Keras)
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        feature_names=np.array(feature_cols),
        feat_mean=feat_mean.values,
        feat_std=feat_std.values,
        seq_len=np.array([SEQ_LEN]),
    )

    print("\n[OK] Saved DL dataset ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()
