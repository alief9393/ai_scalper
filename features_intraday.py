# features_intraday.py

import pandas as pd
import numpy as np

def add_m15_features(df):
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))

    df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
    df["oc_range"] = (df["close"] - df["open"]) / df["open"]

    df["rv_5"]  = df["log_ret_1"].rolling(5).std()
    df["rv_20"] = df["log_ret_1"].rolling(20).std()

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14_m15"] = tr.rolling(14).mean()

    df["ema_20_m15"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50_m15"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_ratio_20_50_m15"] = df["ema_20_m15"] / df["ema_50_m15"]

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
    df_h1 = df_h1.copy().sort_values("time")
    df_m15 = df_m15.copy().sort_values("time")

    df_h1["ret_1_h1"] = df_h1["close"].pct_change()
    df_h1["log_ret_1_h1"] = np.log(df_h1["close"] / df_h1["close"].shift(1))

    df_h1["ema_50_h1"] = df_h1["close"].ewm(span=50, adjust=False).mean()
    df_h1["ema_100_h1"] = df_h1["close"].ewm(span=100, adjust=False).mean()
    df_h1["ema_ratio_50_100_h1"] = df_h1["ema_50_h1"] / df_h1["ema_100_h1"]

    hl = df_h1["high"] - df_h1["low"]
    hc = (df_h1["high"] - df_h1["close"].shift(1)).abs()
    lc = (df_h1["low"]  - df_h1["close"].shift(1)).abs()
    tr_h1 = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_h1["atr_14_h1"] = tr_h1.rolling(14).mean()

    df_ctx = pd.merge_asof(
        df_m15.sort_values("time"),
        df_h1.sort_values("time"),
        on="time",
        direction="backward",
        suffixes=("", "_h1ctx"),
    )
    return df_ctx


def add_daily_context(df_m15, df_d1):
    df_m15 = df_m15.copy()
    df_d1 = df_d1.copy()

    df_m15["date"] = df_m15["time"].dt.date
    df_d1["date"] = df_d1["time"].dt.date

    df_d1["daily_ret"] = df_d1["close"].pct_change()
    df_d1["daily_range"] = (df_d1["high"] - df_d1["low"]) / df_d1["close"].shift(1)

    hl = df_d1["high"] - df_d1["low"]
    hc = (df_d1["high"] - df_d1["close"].shift(1)).abs()
    lc = (df_d1["low"]  - df_d1["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df_d1["atr_14_d1"] = tr.rolling(14).mean()

    P = (df_d1["high"].shift(1) + df_d1["low"].shift(1) + df_d1["close"].shift(1)) / 3.0
    df_d1["pivot_p"] = P
    df_d1["pivot_r1"] = 2 * P - df_d1["low"].shift(1)
    df_d1["pivot_s1"] = 2 * P - df_d1["high"].shift(1)
    df_d1["pivot_r2"] = P + (df_d1["high"].shift(1) - df_d1["low"].shift(1))
    df_d1["pivot_s2"] = P - (df_d1["high"].shift(1) - df_d1["low"].shift(1))

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
    df_merged = df_m15.merge(df_d1_small, on="date", how="left")

    return df_merged
