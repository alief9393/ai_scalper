"""
feature_engineering_advanced.py

Advanced feature engineering untuk meningkatkan akurasi model scalping:
- Price Action Features (Wick, Body, Candle Strength)
- Volatility Regime (ATR-based)
- VWAP (Institutional Value Level)
- Session Strength (Market Session Impact)
- Multi-timeframe EMA Trend Alignment (M5, M15, H1)

Designed to improve BUY/SELL signal detection quality.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta


def add_advanced_features(df):
    df = df.copy()

    # ===== 1) Price Action Features =====
    df["upper_wick"] = df["high"] - np.maximum(df["open"], df["close"])
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["body_size"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]

    # Rasio body ke total range (candle strength)
    df["body_ratio"] = df["body_size"] / (df["candle_range"] + 1e-6)
    df["upper_wick_ratio"] = df["upper_wick"] / (df["candle_range"] + 1e-6)
    df["lower_wick_ratio"] = df["lower_wick"] / (df["candle_range"] + 1e-6)

    # ===== 2) Volatility Regime (ATR Strength) =====
    df["ATR_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ATR_50"] = ta.atr(df["high"], df["low"], df["close"], length=50)
    df["vol_regime"] = df["ATR_14"] / (df["ATR_50"] + 1e-6)

    # ===== 3) VWAP (Volume Weighted Avg Price) =====
    df["vwap"] = (df["close"] * df["tick_volume"]).cumsum() / (df["tick_volume"].cumsum() + 1e-6)
    df["distance_from_vwap"] = df["close"] - df["vwap"]

    # ===== 4) Market Session Strength =====
    df["hour"] = df["time"].dt.hour
    df["is_asian"] = df["hour"].between(0, 7).astype(int)
    df["is_london"] = df["hour"].between(7, 15).astype(int)
    df["is_newyork"] = df["hour"].between(12, 21).astype(int)

    # ===== 5) Multi Timeframe EMA (Trend Alignment) =====
    df["ema_fast"] = ta.ema(df["close"], length=10)
    df["ema_mid"] = ta.ema(df["close"], length=30)
    df["ema_slow"] = ta.ema(df["close"], length=50)
    df["ema_trend_strength"] = df["ema_fast"] - df["ema_slow"]

    # ===== Handle NaN =====
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print("[INFO] Advanced features added (PA, VWAP, EMA Trend, ATR Regime).")
    return df

