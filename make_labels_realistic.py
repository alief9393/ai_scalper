"""
make_labels_realistic.py

Bikin label BUY / SELL / NO_TRADE yang realistis berdasarkan:
- Entry di close[i] atau open[i+1] (pilih via ENTRY_MODE)
- TP/SL fixed (dalam "pips harga" kayak backtest lo)
- SL dicek dulu baru TP (konservatif)
- MAX_BARS_HOLD forward-only (gak ada lookahead curang)

Output:
- xau_M5_ml_dataset_ADV_realistic_labels.csv
  dengan kolom baru:
    - label_realistic (0=SELL,1=NO_TRADE,2=BUY)
    - real_buy_pips
    - real_sell_pips
"""

import pandas as pd
import numpy as np

# ========= CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV.csv"
CSV_OUTPUT = "xau_M5_ml_dataset_ADV_realistic_labels.csv"

TP_PIPS = 1.0
SL_PIPS = 0.5
MAX_BARS_HOLD = 10

# ENTRY_MODE:
# "close"     -> entry di CLOSE[i], mirip live bot lo sekarang
# "next_open" -> entry di OPEN[i+1], versi ultra-realistic
ENTRY_MODE = "close"   # bisa lo ganti ke "next_open" buat eksperimen


def simulate_trade(df, i, direction, entry_mode="close"):
    """
    Simulasikan 1 trade (BUY/SELL) mulai dari index i,
    dengan:
      - entry di close[i] atau open[i+1]
      - SL-first, lalu TP
      - max hold = MAX_BARS_HOLD bar
    Return:
      profit_pips (float)
    """
    n = len(df)
    if entry_mode not in ("close", "next_open"):
        raise ValueError("entry_mode harus 'close' atau 'next_open'")

    # Tentukan entry index & harga
    if entry_mode == "close":
        if i >= n - 1:
            return 0.0  # gak cukup bar ke depan
        entry_idx = i
        entry_price = float(df.loc[entry_idx, "close"])
        start_j = i + 1
    else:  # "next_open"
        if i + 1 >= n:
            return 0.0
        entry_idx = i + 1
        entry_price = float(df.loc[entry_idx, "open"])
        start_j = entry_idx

    # TP/SL level
    if direction == "BUY":
        tp_level = entry_price + TP_PIPS
        sl_level = entry_price - SL_PIPS
    else:  # SELL
        tp_level = entry_price - TP_PIPS
        sl_level = entry_price + SL_PIPS

    hit = None
    last_j = start_j

    # Loop forward sampai MAX_BARS_HOLD bar atau habis data
    for j in range(start_j, min(start_j + MAX_BARS_HOLD, n)):
        high = float(df.loc[j, "high"])
        low = float(df.loc[j, "low"])

        if direction == "BUY":
            # Konservatif: cek SL dulu
            if low <= sl_level:
                hit = "SL"
                last_j = j
                break
            if high >= tp_level:
                hit = "TP"
                last_j = j
                break
        else:  # SELL
            if high >= sl_level:
                hit = "SL"
                last_j = j
                break
            if low <= tp_level:
                hit = "TP"
                last_j = j
                break

        last_j = j

    # Hitung profit dalam "pips harga"
    if hit == "TP":
        profit_pips = TP_PIPS
    elif hit == "SL":
        profit_pips = -SL_PIPS
    else:
        # time exit di bar terakhir yg kita cek
        last_price = float(df.loc[last_j, "close"])
        if direction == "BUY":
            profit_pips = last_price - entry_price
        else:
            profit_pips = entry_price - last_price

    return profit_pips


def main():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(CSV_INPUT)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    n = len(df)
    print(f"[INFO] Total rows: {n}")

    buy_pips_list = []
    sell_pips_list = []
    labels = []

    for i in range(n):
        # Untuk bar-bar paling akhir yang ga cukup ruang ke depan, kasih NO_TRADE
        if i >= n - 2:
            buy_pips = 0.0
            sell_pips = 0.0
            label = 1  # NO_TRADE
        else:
            buy_pips = simulate_trade(df, i, direction="BUY", entry_mode=ENTRY_MODE)
            sell_pips = simulate_trade(df, i, direction="SELL", entry_mode=ENTRY_MODE)

            # Keputusan label:
            # - Kalau BUY jelas menang dan SELL nggak bagus -> label BUY (2)
            # - Kalau SELL jelas menang dan BUY nggak bagus -> label SELL (0)
            # - Kalau dua-duanya jelek / abu-abu -> NO_TRADE (1)

            # Bisa pakai threshold kecil supaya noise +/- kecil dianggap NO_TRADE
            THRESH = 0.1  # 0.1 "pips harga" -> lo bisa adjust nanti

            good_buy = buy_pips > THRESH
            good_sell = sell_pips > THRESH

            if good_buy and not good_sell:
                label = 2  # BUY
            elif good_sell and not good_buy:
                label = 0  # SELL
            else:
                label = 1  # NO_TRADE

        buy_pips_list.append(buy_pips)
        sell_pips_list.append(sell_pips)
        labels.append(label)

        if (i + 1) % 5000 == 0:
            print(f"[INFO] Processed {i+1}/{n} rows...")

    df["real_buy_pips"] = buy_pips_list
    df["real_sell_pips"] = sell_pips_list
    df["label_realistic"] = labels

    print("[INFO] Label distribution (label_realistic):")
    print(df["label_realistic"].value_counts().sort_index())
    # 0 = SELL, 1 = NO_TRADE, 2 = BUY

    df.to_csv(CSV_OUTPUT, index=False)
    print(f"[SAVED] → {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
