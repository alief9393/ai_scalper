# apply_threshold_signals.py

import pandas as pd
from xgboost import XGBClassifier

FILE_INPUT  = "XAUUSD_M15_FEATURES_LABELED.csv"
MODEL_PATH  = "xgb_intraday_model.json"
FILE_OUTPUT = "XAUUSD_M15_WITH_SIGNALS.csv"

THR_LONG  = 0.65
THR_SHORT = 1.0 - THR_LONG   # 0.35


def main():
    # Load data
    df = pd.read_csv(FILE_INPUT, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    # Tambah kolom 'set' (info aja, bukan fitur)
    df["set"] = "train"
    df.loc[train_end:val_end-1, "set"] = "val"
    df.loc[val_end:, "set"] = "test"

    # Siapkan X (fitur) sama seperti waktu training
    drop_cols = ["time", "date", "target", "future_ret", "set"]
    # Kalau file input ini sudah pernah dihasilin sebelumnya & punya kolom proba_up/signal, drop juga:
    extra_drop = ["proba_up", "signal"]
    for c in extra_drop:
        if c in df.columns:
            drop_cols.append(c)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Pastikan semua kolom fitur numerik
    X = X.astype(float)

    # Load model
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # Prediksi probabilitas kelas 1 (UP) untuk seluruh dataset
    proba_up = model.predict_proba(X)[:, 1]
    df["proba_up"] = proba_up

    # Terapkan threshold → bikin kolom signal
    def make_signal(p):
        if p >= THR_LONG:
            return 1   # long
        elif p <= THR_SHORT:
            return -1  # short
        else:
            return 0   # flat / no trade

    df["signal"] = df["proba_up"].apply(make_signal)

    # Simpan
    df.to_csv(FILE_OUTPUT, index=False)
    print(f"[OK] Saved -> {FILE_OUTPUT}")
    print("[INFO] Signal distribution (fraction):")
    print(df["signal"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
