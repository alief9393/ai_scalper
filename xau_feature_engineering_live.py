# xau_feature_engineering_live.py
"""
Feature Engineering for LIVE Data
- Accepts raw OHLC data from MT5
- Adds technical indicators & session/time features
- Output must match EXACTLY the feature set used during training
"""

import pandas as pd
import numpy as np
import pandas_ta as ta

def build_live_features(df):
    """
    Input:
        df => raw OHLCV DataFrame from MT5:
            ['time', 'open', 'high', 'low', 'close', 'tick_volume']

    Output:
        df_feat => DataFrame with FULL ML features:
            Same columns used in training
    """

    df = df.copy()

    # === 1. Ensure time format ===
    df["time"] = pd.to_datetime(df["time"])

    # === 2. Returns ===
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # === 3. EMA-based trend ===
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_fast_slow_diff"] = df["ema_10"] - df["ema_50"]

    # === 4. RSI ===
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # === 5. Bollinger Bands ===
    bb = ta.bbands(df["close"], length=20, std=2)

    # Cari nama kolom yang sesuai prefix
    bb_low_col = [c for c in bb.columns if c.startswith("BBL_")][0]
    bb_mid_col = [c for c in bb.columns if c.startswith("BBM_")][0]
    bb_high_col = [c for c in bb.columns if c.startswith("BBU_")][0]

    df["bb_low"] = bb[bb_low_col]
    df["bb_mid"] = bb[bb_mid_col]
    df["bb_high"] = bb[bb_high_col]
    df["bb_width"] = df["bb_high"] - df["bb_low"]
    df["bb_pos"] = (df["close"] - df["bb_low"]) / df["bb_width"]

    # === 6. ATR (volatility) ===
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # === 7. Stochastic ===
    stoch = ta.stoch(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # === 8. Time & Session Features ===
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Session Flags -> XAUUSD typical sessions
    df["sess_asia"] = df["hour"].between(0, 6).astype(int)
    df["sess_london"] = df["hour"].between(7, 13).astype(int)
    df["sess_ny"] = df["hour"].between(14, 20).astype(int)

    # === 9. Volume Ratios ===
    df["vol_ema_20"] = ta.ema(df["tick_volume"], length=20)
    df["vol_ratio"] = df["tick_volume"] / df["vol_ema_20"]

    # === 10. Clean missing values ===
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    return df
