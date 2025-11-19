# xau_feature_engineering.py
"""
Step 2: Feature Engineering + Labeling untuk XAUUSD M5 ML Scalping

- Input: CSV history XAUUSD M5 (hasil export dari MT5)
- Output: CSV dataset siap ML (fitur + label)
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path

# ================== CONFIG ==================

# Nama file CSV mentah yang sudah kamu export dari MT5
CSV_INPUT = "xau_M5_history_365d.csv"   # ganti kalau beda nama
CSV_OUTPUT = "xau_M5_ml_dataset.csv"

# Labeling horizon (berapa candle ke depan yang kita lihat)
HORIZON = 3  # 3 candle M5 = 15 menit ke depan

# Threshold untuk menganggap sinyal BUY/SELL (dalam %)
TP_THRESHOLD = 0.0008   # 0.08% ke atas -> BUY
SL_THRESHOLD = -0.0008  # -0.08% ke bawah -> SELL

# ============================================

def load_raw_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV input tidak ditemukan: {csv_path}")

    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Pastikan kolom waktu ada
    if "time" not in df.columns:
        raise ValueError("Kolom 'time' tidak ditemukan di CSV. Pastikan export dari MT5 sesuai.")
    
    # Parse datetime
    df["time"] = pd.to_datetime(df["time"])
    
    # Kalau ada kolom time_local, kita pakai juga (kalau ga ada, nanti pakai time utc saja)
    if "time_local" in df.columns:
        df["time_local"] = pd.to_datetime(df["time_local"], errors="coerce")
    else:
        df["time_local"] = df["time"]

    # Sort by time & set index
    df = df.sort_values("time")
    df.set_index("time", inplace=True)

    # Basic check
    print(f"[INFO] Loaded rows: {len(df)}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adding technical features...")

    # 1. Return sederhana
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # 2. Moving Averages
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_fast_slow_diff"] = df["ema_10"] - df["ema_50"]

    # 3. RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # 4. Bollinger Bands (versi robust, gak ngandelin nama kolom fix)
    bb = ta.bbands(df["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        cols = list(bb.columns)

        def find_col(prefix: str):
            for c in cols:
                if isinstance(c, str) and c.startswith(prefix):
                    return c
            return None

        low_col = find_col("BBL_")
        mid_col = find_col("BBM_")
        high_col = find_col("BBU_")

        if low_col and mid_col and high_col:
            df["bb_low"] = bb[low_col]
            df["bb_mid"] = bb[mid_col]
            df["bb_high"] = bb[high_col]
            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
            df["bb_pos"] = (df["close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])
        else:
            print(f"[WARN] BB columns not found as expected. Got: {cols}")
    else:
        print("[WARN] Bollinger Bands gagal dihitung, cek data.")

    # 5. ATR (volatilitas)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # 6. Stochastic (momentum)
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    if stoch is not None and not stoch.empty:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]
    else:
        print("[WARN] Stoch gagal dihitung, cek data.")

    # 7. Volume-based features (kalau ada tick_volume)
    if "tick_volume" in df.columns:
        df["vol_ema_20"] = ta.ema(df["tick_volume"], length=20)
        df["vol_ratio"] = df["tick_volume"] / df["vol_ema_20"]

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adding time/session features...")

    # Pastikan time_local bukan NaT
    df["time_local"] = pd.to_datetime(df["time_local"], errors="coerce")
    tl = df["time_local"].copy()

    df["hour"] = tl.dt.hour
    df["minute"] = tl.dt.minute
    df["dayofweek"] = tl.dt.dayofweek  # 0=Mon

    # Encode jam sebagai sin-cos (cyclical feature)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Session flag kasar (pakai kira-kira Jakarta time)
    # Ini cuma indikasi, ga harus super presisi.
    # Asia: 06-15, London: 15-22, NY: 20-05 (kurang lebih)
    df["sess_asia"] = ((df["hour"] >= 6) & (df["hour"] < 15)).astype(int)
    df["sess_london"] = ((df["hour"] >= 15) & (df["hour"] < 22)).astype(int)
    df["sess_ny"] = ((df["hour"] >= 20) | (df["hour"] < 5)).astype(int)

    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Adding labels (BUY / SELL / NO_TRADE)...")

    # Future close setelah HORIZON candle
    df["future_close"] = df["close"].shift(-HORIZON)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]

    # Inisialisasi label: 0 = NO_TRADE
    df["label"] = 0

    # BUY: kalau future_return >= TP_THRESHOLD
    df.loc[df["future_return"] >= TP_THRESHOLD, "label"] = 1

    # SELL: kalau future_return <= SL_THRESHOLD
    df.loc[df["future_return"] <= SL_THRESHOLD, "label"] = -1

    # Drop baris yang future_close-nya NaN (bagian paling akhir)
    df = df.dropna(subset=["future_close", "future_return"])

    print("[INFO] Label distribution (count):")
    print(df["label"].value_counts())
    print("[INFO] Label distribution (ratio):")
    print(df["label"].value_counts(normalize=True))

    return df


def build_dataset():
    # 1. Load raw
    df = load_raw_data(CSV_INPUT)

    # 2. Tambah fitur teknikal
    df = add_technical_features(df)

    # 3. Tambah fitur waktu/sesi
    df = add_time_features(df)

    # 4. Tambah label
    df = add_labels(df)

    # 5. Drop baris yang ada NaN di fitur penting
    feature_cols = [
        "open", "high", "low", "close",
        "ret_1", "ret_3", "ret_6",
        "ema_10", "ema_20", "ema_50", "ema_fast_slow_diff",
        "rsi_14",
        "bb_low", "bb_mid", "bb_high", "bb_width", "bb_pos",
        "atr_14",
        "stoch_k", "stoch_d",
        "hour", "dayofweek", "hour_sin", "hour_cos",
        "sess_asia", "sess_london", "sess_ny",
    ]

    # Tambah volume features kalau ada
    if "tick_volume" in df.columns:
        feature_cols += ["tick_volume", "vol_ema_20", "vol_ratio"]

    # Pastikan ada di df
    feature_cols = [c for c in feature_cols if c in df.columns]

    all_cols = feature_cols + ["label"]

    dataset = df[all_cols].dropna()
    print(f"[INFO] Final dataset rows: {len(dataset)}")
    print(f"[INFO] Feature count: {len(feature_cols)}")

    # Save ke CSV
    dataset.to_csv(CSV_OUTPUT, index=True)
    print(f"[INFO] Saved ML dataset to: {CSV_OUTPUT}")

    # Tampilkan sample
    print("[INFO] Sample dataset head:")
    print(dataset.head())


if __name__ == "__main__":
    build_dataset()
