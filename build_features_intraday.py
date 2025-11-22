import pandas as pd
import numpy as np

# Input dari step sebelumnya
FILE_M15 = "XAUUSD_M15_ICM.csv"
FILE_H1  = "XAUUSD_H1_ICM.csv"
FILE_D1  = "XAUUSD_D1_ICM.csv"

# Output fitur
OUT_FEATURES = "XAUUSD_M15_FEATURES.csv"


def add_m15_features(df):
    """
    Fitur level M15 (local intraday)
    """
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Return & log return
    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))

    # Range intrabar
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    # Rolling volatility (realized vol style)
    df["rv_5"]  = df["log_ret_1"].rolling(5).std()
    df["rv_20"] = df["log_ret_1"].rolling(20).std()

    # Simple ATR-like di M15 (high-low + gap prev close)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14_m15"] = tr.rolling(14).mean()

    # EMA & momentum M15
    df["ema_20_m15"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50_m15"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_ratio_20_50_m15"] = df["ema_20_m15"] / df["ema_50_m15"]

    # Simple RSI 14 (M15)
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    df["rsi_14_m15"] = rsi.values

    return df


def add_h1_context(df_m15, df_h1):
    """
    Tambah konteks H1 (trend, volatility) ke bar M15
    pakai merge_asof (align ke candle H1 terakhir <= time M15).
    """
    df_h1 = df_h1.copy().sort_values("time")
    df_m15 = df_m15.copy().sort_values("time")

    # Fitur H1 dasar
    df_h1["ret_1_h1"] = df_h1["close"].pct_change()
    df_h1["log_ret_1_h1"] = np.log(df_h1["close"] / df_h1["close"].shift(1))

    # EMA H1 sebagai trend context
    df_h1["ema_50_h1"] = df_h1["close"].ewm(span=50, adjust=False).mean()
    df_h1["ema_100_h1"] = df_h1["close"].ewm(span=100, adjust=False).mean()
    df_h1["ema_ratio_50_100_h1"] = df_h1["ema_50_h1"] / df_h1["ema_100_h1"]

    # Volatility di H1
    hl = df_h1["high"] - df_h1["low"]
    hc = (df_h1["high"] - df_h1["close"].shift(1)).abs()
    lc = (df_h1["low"]  - df_h1["close"].shift(1)).abs()
    tr_h1 = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_h1["atr_14_h1"] = tr_h1.rolling(14).mean()

    # Align ke M15 (asof merge: H1 row terakhir <= time M15)
    df_ctx = pd.merge_asof(
        df_m15.sort_values("time"),
        df_h1.sort_values("time"),
        on="time",
        direction="backward",
        suffixes=("", "_h1ctx"),
    )

    return df_ctx


def add_daily_context(df_m15, df_d1):
    """
    Tambah konteks harian (pivot, daily return, daily ATR).
    Merge ke M15 via date (floor ke hari).
    """
    df_m15 = df_m15.copy()
    df_d1 = df_d1.copy()

    # Tambah kolom date
    df_m15["date"] = df_m15["time"].dt.date
    df_d1["date"] = df_d1["time"].dt.date

    # Daily returns & range
    df_d1["daily_ret"] = df_d1["close"].pct_change()
    df_d1["daily_range"] = (df_d1["high"] - df_d1["low"]) / df_d1["close"].shift(1)

    # Daily ATR (14 hari)
    hl = df_d1["high"] - df_d1["low"]
    hc = (df_d1["high"] - df_d1["close"].shift(1)).abs()
    lc = (df_d1["low"]  - df_d1["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_d1["atr_14_d1"] = tr.rolling(14).mean()

    # Classic daily pivot
    P = (df_d1["high"].shift(1) + df_d1["low"].shift(1) + df_d1["close"].shift(1)) / 3.0
    df_d1["pivot_p"] = P
    df_d1["pivot_r1"] = 2 * P - df_d1["low"].shift(1)
    df_d1["pivot_s1"] = 2 * P - df_d1["high"].shift(1)
    df_d1["pivot_r2"] = P + (df_d1["high"].shift(1) - df_d1["low"].shift(1))
    df_d1["pivot_s2"] = P - (df_d1["high"].shift(1) - df_d1["low"].shift(1))

    # Pilih kolom penting daily
    daily_cols = [
        "date",
        "daily_ret",
        "daily_range",
        "atr_14_d1",
        "pivot_p",
        "pivot_r1",
        "pivot_s1",
        "pivot_r2",
        "pivot_s2",
    ]
    df_d1_small = df_d1[daily_cols].drop_duplicates(subset=["date"])

    # Merge ke M15 via 'date'
    df_merged = df_m15.merge(df_d1_small, on="date", how="left")

    return df_merged


def main():
    # Load data utama
    df_m15 = pd.read_csv(FILE_M15, parse_dates=["time"])
    df_h1  = pd.read_csv(FILE_H1,  parse_dates=["time"])
    df_d1  = pd.read_csv(FILE_D1,  parse_dates=["time"])

    # Step 1: fitur lokal M15
    print("[INFO] Building M15 local features...")
    df_feat = add_m15_features(df_m15)

    # Step 2: konteks H1
    print("[INFO] Adding H1 context...")
    df_feat = add_h1_context(df_feat, df_h1)

    # Step 3: konteks Daily (pivot, ATR harian, dll)
    print("[INFO] Adding Daily context...")
    df_feat = add_daily_context(df_feat, df_d1)

    # Drop bar awal yang banyak NaN (indikator rolling)
    df_feat = df_feat.sort_values("time").reset_index(drop=True)
    df_feat = df_feat.dropna().reset_index(drop=True)

    print("[INFO] Final feature rows:", len(df_feat))
    print("[INFO] Range:", df_feat["time"].min(), "->", df_feat["time"].max())

    # Simpan
    df_feat.to_csv(OUT_FEATURES, index=False)
    print(f"[OK] Saved features -> {OUT_FEATURES}")


if __name__ == "__main__":
    main()
