# live_engine_lstm_intraday.py
#
# DL Intraday v1 Live Engine (XAUUSD M15, LSTM + ATR SL/TP)
# - Connect MT5
# - Ambil M15/H1/D1
# - Bangun fitur (sama seperti training pipeline)
# - LSTM inference -> signal + SL/TP (ATR 1.2 / 2.0)
# - Kirim order (kalau DRY_RUN=False)
#
# NOTE:
#   - Pastikan features_intraday.py ada di folder yang sama:
#       from features_intraday import add_m15_features, add_h1_context, add_daily_context

import time
from datetime import datetime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import torch
import torch.nn as nn

from features_intraday import add_m15_features, add_h1_context, add_daily_context

# =============== CONFIG =====================

SYMBOL    = "XAUUSD"          # sesuaikan dengan nama simbol di MT5
TIMEFRAME = mt5.TIMEFRAME_M15

DATA_META_FILE = "dl_intraday_dataset_seq64.npz"
MODEL_PATH     = "lstm_intraday_best.pt"

# Threshold final DL Intraday v1
THR_LONG  = 0.70
THR_SHORT = 0.30

# ATR SL/TP config (best grid)
ATR_COL      = "atr_14_m15"
ATR_MULT_SL  = 1.2    # SL = 1.2 * ATR_M15
RR_TP        = 2.0    # TP = 2.0 * SL = 2.4 * ATR_M15

LOT_SIZE     = 0.01   # default lot untuk 1 posisi
MAGIC_NUMBER = 987654  # bebas tapi unik untuk EA ini

DRY_RUN      = True   # True: hanya log, tidak kirim order. False: kirim order beneran.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============== MODEL DEF ==================


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits


def load_meta_and_model():
    data = np.load(DATA_META_FILE, allow_pickle=True)
    feature_names = list(data["feature_names"])
    seq_len = int(data["seq_len"][0])
    input_dim = len(feature_names)

    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Loaded meta/model:")
    print("  seq_len      :", seq_len)
    print("  n_features   :", input_dim)
    print("  device       :", DEVICE)

    return feature_names, seq_len, model


# =============== MT5 UTILS ==================


def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print("[INFO] Connected to MT5")

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        raise RuntimeError(f"Symbol {SYMBOL} not found in MT5")

    if not symbol_info.visible:
        mt5.symbol_select(SYMBOL, True)
    print("[INFO] Symbol ready:", SYMBOL)


def get_latest_m15_df(n_bars=500):
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos M15 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_latest_h1_df(n_bars=300):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos H1 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_latest_d1_df(n_bars=200):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_D1, 0, n_bars)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos D1 failed: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_symbol_tick():
    info = mt5.symbol_info_tick(SYMBOL)
    if info is None:
        raise RuntimeError(f"symbol_info_tick failed for {SYMBOL}")
    return info.bid, info.ask


def has_open_position():
    positions = mt5.positions_get(symbol=SYMBOL)
    return positions is not None and len(positions) > 0


def send_order(direction, volume, entry_price, sl_price, tp_price):
    if direction == "LONG":
        order_type = mt5.ORDER_TYPE_BUY
        price = entry_price
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = entry_price

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 50,
        "magic": MAGIC_NUMBER,
        "comment": "DL_INTRADAY_V1",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if DRY_RUN:
        print("[DRY_RUN] Order not sent. Request would be:")
        print(" ", request)
        return

    result = mt5.order_send(request)
    print("[ORDER]", result)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("[WARN] Order failed:", result.retcode)


# =========== FEATURE BUILDER LIVE ==========


def build_features_for_m15(df_m15_raw):
    """
    Live version of your offline pipeline:
      M15 -> add_m15_features
      + H1 -> add_h1_context
      + D1 -> add_daily_context
    """
    df_m15 = df_m15_raw.copy().sort_values("time").reset_index(drop=True)

    df_h1 = get_latest_h1_df(n_bars=300)
    df_d1 = get_latest_d1_df(n_bars=200)

    df_feat = add_m15_features(df_m15)
    df_feat = add_h1_context(df_feat, df_h1)
    df_feat = add_daily_context(df_feat, df_d1)

    df_feat = df_feat.sort_values("time").reset_index(drop=True)
    df_feat = df_feat.dropna().reset_index(drop=True)

    return df_feat


# =============== INFERENCE =================


def build_sequence_from_df(df_feat, feature_names, seq_len):
    if len(df_feat) < seq_len:
        raise ValueError(f"Data kurang, butuh {seq_len} bar, cuma ada {len(df_feat)}")

    df_seq = df_feat.sort_values("time").iloc[-seq_len:]
    missing = [f for f in feature_names if f not in df_seq.columns]
    if missing:
        raise ValueError(f"Missing feature columns in live DF: {missing}")

    X = df_seq[feature_names].to_numpy(dtype=np.float32)
    X = np.expand_dims(X, axis=0)
    last_row = df_seq.iloc[-1]
    return X, last_row


def infer_signal(model, X, last_row, bid, ask):
    X_t = torch.from_numpy(X).float().to(DEVICE)
    with torch.no_grad():
        logits = model(X_t)
        prob = torch.softmax(logits, dim=1)
        proba_up = float(prob[0, 1].cpu().item())

    # Signal
    if proba_up >= THR_LONG:
        signal = 1
    elif proba_up <= THR_SHORT:
        signal = -1
    else:
        signal = 0

    # Entry price (real: BUY di Ask, SELL di Bid)
    if signal == 1:
        entry_price = float(ask)
        direction_str = "LONG"
    elif signal == -1:
        entry_price = float(bid)
        direction_str = "SHORT"
    else:
        entry_price = None
        direction_str = "FLAT"

    atr_val = None
    sl_price = None
    tp_price = None
    sl_dist = None
    tp_dist = None

    if signal != 0:
        if ATR_COL not in last_row.index:
            raise ValueError(f"{ATR_COL} tidak ada di live features")
        atr_val = float(last_row[ATR_COL])
        sl_dist = ATR_MULT_SL * atr_val
        tp_dist = RR_TP * sl_dist

        if signal == 1:
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

    return {
        "proba_up": proba_up,
        "signal": signal,
        "direction": direction_str,
        "entry_price": entry_price,
        "atr_m15": atr_val,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "decision_time": last_row["time"],
    }


# =============== MAIN LOOP =================


def main_loop():
    feature_names, seq_len, model = load_meta_and_model()
    connect_mt5()

    print("[INFO] Starting main loop...")
    last_bar_time = None

    while True:
        try:
            df_m15 = get_latest_m15_df(n_bars=500)
            current_bar_time = df_m15["time"].iloc[-1]

            # Deteksi bar baru (close M15)
            if last_bar_time is not None and current_bar_time == last_bar_time:
                time.sleep(5)
                continue

            last_bar_time = current_bar_time
            print(f"\n[INFO] New M15 bar detected: {current_bar_time}")

            df_feat = build_features_for_m15(df_m15)
            if len(df_feat) < seq_len:
                print("[WARN] Feature DF belum cukup panjang untuk seq_len, skip bar ini.")
                time.sleep(5)
                continue

            X, last_row = build_sequence_from_df(df_feat, feature_names, seq_len)
            bid, ask = get_symbol_tick()

            result = infer_signal(model, X, last_row, bid, ask)

            print("  Decision time:", result["decision_time"])
            print(f"  proba_up     : {result['proba_up']:.4f}")
            print(f"  signal       : {result['signal']} ({result['direction']})")

            if result["signal"] == 0:
                print("  -> FLAT, no trade.")
            else:
                print(f"  entry_price  : {result['entry_price']:.2f}")
                print(f"  ATR(M15)     : {result['atr_m15']:.2f}")
                print(f"  SL price     : {result['sl_price']:.2f}")
                print(f"  TP price     : {result['tp_price']:.2f}")

                if has_open_position():
                    print("  -> Existing position detected, skip new order.")
                else:
                    print("  -> Sending order..." if not DRY_RUN else "  -> DRY_RUN: would send order now.")
                    send_order(result["direction"], LOT_SIZE, result["entry_price"],
                               result["sl_price"], result["tp_price"])

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")
            break
        except Exception as e:
            print("[ERROR] Exception in main loop:", e)
            time.sleep(10)

    mt5.shutdown()
    print("[INFO] MT5 shutdown.")


if __name__ == "__main__":
    main_loop()
