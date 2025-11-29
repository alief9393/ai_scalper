import pandas as pd
import numpy as np

# ================== CONFIG ==================
DL_FILE   = "XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv"
XGB_FILE  = "XAUUSD_M15_WITH_XGB_SIGNALS_FULL.csv"
OUT_FILE  = "XAUUSD_M15_WITH_HYBRID_SIGNALS_FULL.csv"

# Bobot hybrid (bisa di-tuning nanti)
W_DL  = 0.4
W_XGB = 0.6

# Threshold untuk hybrid score
THR_LONG  = 0.70
THR_SHORT = 0.30

# Optional: minimal confidence (jarak dari 0.5)
USE_CONF_FILTER   = True
CONF_MARGIN_HYBRID = 0.20   # misal butuh |p - 0.5| >= 0.20


def main():
    print("[INFO] Loading DL + XGB CSV...")
    df_dl = pd.read_csv(DL_FILE, parse_dates=["time"])
    df_xgb = pd.read_csv(XGB_FILE, parse_dates=["time"])

    # Pastikan sorted
    df_dl = df_dl.sort_values("time").reset_index(drop=True)
    df_xgb = df_xgb.sort_values("time").reset_index(drop=True)

    # Ambil kolom penting dari XGB (supaya nggak duplikat OHLC dsb)
    xgb_cols_keep = ["time", "xgb_signal", "xgb_proba_up"]
    missing_xgb = [c for c in xgb_cols_keep if c not in df_xgb.columns]
    if missing_xgb:
        raise ValueError(f"Missing columns in XGB CSV: {missing_xgb}")

    df_xgb_small = df_xgb[xgb_cols_keep].copy()

    # Merge by time (inner join supaya candle yang nggak ada salah satu side ke-drop)
    print("[INFO] Merging on 'time'...")
    df = pd.merge(
        df_dl,
        df_xgb_small,
        on="time",
        how="inner",
        suffixes=("", "_xgb"),
    )

    print(f"[INFO] After merge rows: {len(df)}")
    if "set" in df_dl.columns and "set" in df.columns:
        print(df["set"].value_counts())

    # Cek kolom proba DL
    if "dl_proba_up" not in df.columns:
        raise ValueError("Kolom 'dl_proba_up' tidak ditemukan di DL CSV.")

    # Ambil proba
    dl_p  = df["dl_proba_up"].astype(float)
    xgb_p = df["xgb_proba_up"].astype(float)

    # Siapkan hybrid score
    hybrid_score = pd.Series(0.5, index=df.index, dtype=float)

    # Kasus: keduanya ada
    mask_both = dl_p.notna() & xgb_p.notna()
    hybrid_score[mask_both] = W_DL * dl_p[mask_both] + W_XGB * xgb_p[mask_both]

    # Kasus fallback: cuma DL
    mask_dl_only = dl_p.notna() & ~xgb_p.notna()
    hybrid_score[mask_dl_only] = dl_p[mask_dl_only]

    # Kasus fallback: cuma XGB
    mask_xgb_only = xgb_p.notna() & ~dl_p.notna()
    hybrid_score[mask_xgb_only] = xgb_p[mask_xgb_only]

    df["hybrid_proba_up"] = hybrid_score

    # ====== Generate hybrid_signal ======
    hybrid_signal = np.zeros(len(df), dtype=int)

    # Confidence filter (optional)
    if USE_CONF_FILTER:
        conf = (hybrid_score - 0.5).abs()
        mask_conf_ok = conf >= CONF_MARGIN_HYBRID
    else:
        mask_conf_ok = pd.Series(True, index=df.index)

    # Long / Short decision
    long_mask  = (hybrid_score >= THR_LONG) & mask_conf_ok
    short_mask = (hybrid_score <= THR_SHORT) & mask_conf_ok

    hybrid_signal[long_mask]  = 1
    hybrid_signal[short_mask] = -1

    df["hybrid_signal"] = hybrid_signal

    # ====== (Optional) Sanity check vs DL/XGB signal ======
    if "dl_signal" in df.columns and "xgb_signal" in df.columns:
        agree_mask = (df["dl_signal"] == df["xgb_signal"]) & (df["dl_signal"] != 0)
        both_flat  = (df["dl_signal"] == 0) & (df["xgb_signal"] == 0)

        print("\n[INFO] Signal agreement (non-zero only):")
        print("  Both non-zero & agree :", agree_mask.sum())
        print("  Both zero (flat)      :", both_flat.sum())

    # Distribusi signal di test set
    if "set" in df.columns:
        df_test = df[df["set"] == "test"].copy()
        print("\n[INFO] hybrid_signal distribution (TEST):")
        print(
            df_test["hybrid_signal"]
            .value_counts(normalize=True)
            .rename("proportion")
        )
    else:
        print("\n[INFO] hybrid_signal distribution (ALL):")
        print(
            df["hybrid_signal"]
            .value_counts(normalize=True)
            .rename("proportion")
        )

    # ====== Save ======
    df.to_csv(OUT_FILE, index=False)
    print(f"\n[OK] Saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()
