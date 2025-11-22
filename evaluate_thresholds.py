# evaluate_thresholds.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier

FILE_INPUT = "XAUUSD_M15_FEATURES_LABELED.csv"
MODEL_PATH = "xgb_intraday_model.json"

# list threshold untuk diuji (probabilitas kelas 1 / UP)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def load_data():
    df = pd.read_csv(FILE_INPUT, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    y = df["target"].astype(int)

    drop_cols = [
        "time",
        "date",
        "target",
        "future_ret",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

    print("[INFO] Data loaded and split (time-based):")
    print("  Total :", n)
    print("  Train :", X_train.shape[0])
    print("  Val   :", X_val.shape[0])
    print("  Test  :", X_test.shape[0])

    return X_train, y_train, X_val, y_val, X_test, y_test, X.columns


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"[INFO] Loaded model from {MODEL_PATH}")
    return model


def evaluate_thresholds(y_true, proba, thresholds):
    """
    y_true: array of true labels (0/1)
    proba : predicted probability of class 1 (UP)
    thresholds: list of thr_long values
    """
    results = []

    for thr in thresholds:
        thr_long = thr          # BUY jika p >= thr_long
        thr_short = 1.0 - thr   # SELL jika p <= thr_short (simetris)

        long_mask  = proba >= thr_long
        short_mask = proba <= thr_short
        trade_mask = long_mask | short_mask

        n_total   = len(y_true)
        n_trades  = trade_mask.sum()
        n_longs   = long_mask.sum()
        n_shorts  = short_mask.sum()

        if n_trades == 0:
            precision_all = np.nan
        else:
            correct_trades = (
                ((y_true == 1) & long_mask) |
                ((y_true == 0) & short_mask)
            ).sum()
            precision_all = correct_trades / n_trades

        if n_longs == 0:
            precision_long = np.nan
        else:
            precision_long = (y_true[long_mask] == 1).mean()

        if n_shorts == 0:
            precision_short = np.nan
        else:
            precision_short = (y_true[short_mask] == 0).mean()

        coverage = n_trades / n_total  # berapa % bar yang di-trade

        results.append({
            "thr_long": thr_long,
            "thr_short": thr_short,
            "n_trades": int(n_trades),
            "n_longs": int(n_longs),
            "n_shorts": int(n_shorts),
            "coverage": coverage,
            "precision_all": precision_all,
            "precision_long": precision_long,
            "precision_short": precision_short,
        })

    return pd.DataFrame(results)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = load_data()
    model = load_model()

    # Prediksi probabilitas di test set
    proba_test = model.predict_proba(X_test)[:, 1]

    print("\n[INFO] Evaluating thresholds on TEST set...")
    df_res = evaluate_thresholds(y_test.values, proba_test, THRESHOLDS)

    # Format angka biar enak dibaca
    df_res["coverage_pct"] = df_res["coverage"] * 100.0
    for col in ["precision_all", "precision_long", "precision_short"]:
        df_res[col] = (df_res[col] * 100.0).round(2)
    df_res["coverage_pct"] = df_res["coverage_pct"].round(2)

    display_cols = [
        "thr_long", "thr_short",
        "n_trades", "n_longs", "n_shorts",
        "coverage_pct",
        "precision_all", "precision_long", "precision_short",
    ]

    print("\n[THRESHOLD EVALUATION RESULTS] (Test Set)")
    print(df_res[display_cols].to_string(index=False))

    # Simpan ke CSV kalau mau dianalisa di luar
    df_res.to_csv("threshold_evaluation_results.csv", index=False)
    print("\n[OK] Saved threshold evaluation -> threshold_evaluation_results.csv")


if __name__ == "__main__":
    main()
