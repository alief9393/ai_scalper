"""
ml_signal_live_bridge.py

Bridge LIVE / DEMO untuk hybrid ML scalping XAU:
- Baca candle M5 dari MT5 (Exness / broker lain)
- Bangun fitur pakai feature_engineering_advanced.add_advanced_features
- Jalankan XGB (filter) + RF (direction) => hybrid_signal (-1,0,1)
- Session filter (London/NY), cooldown, dynamic lot, margin check
- Eksekusi order BUY/SELL dengan TP/SL fixed (mirip backtest)

PENTING:
- Fitur yang dipakai model harus bisa dihasilkan dari:
  OHLCV + add_advanced_features(). Kalau di pipeline dataset lo
  ada indikator ekstra (RSI, EMA, dsb), tambahin juga di
  build_features_from_candles().
"""

import time
from datetime import datetime
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import joblib
import pandas_ta as ta 

from feature_engineering_advanced import add_advanced_features

# ================= CONFIG =================
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
BARS_HISTORY = 2000          # cukup untuk hitung indikator
POLL_SECONDS = 5             # cek tiap 5 detik

# Model & scaler
XGB_MODEL = "xgb_scalping_model_v3_tuned.pkl"
RF_MODEL  = "rf_scalping_model_v3_balanced.pkl"

# Hybrid config (samakan dengan backtest)
CONF_XGB = 0.50
TP_PIPS = 1.0
SL_PIPS = 0.5
MAX_BARS_HOLD = 10   # di live tidak dipakai utk exit, tapi dibiarkan sama

USE_SESSION_FILTER = True
TRADE_LONDON = True
TRADE_NEWYORK = False
COOLDOWN_BARS = 3    # berdasarkan index bar, sama kayak backtest

# Account / leverage (untuk hitung lot; equity real ambil dari MT5)
LEVERAGE = 1000
CONTRACT_SIZE = 100      # XAU 100oz
RISK_PERCENT = 5.0
MIN_LOT = 0.01
MAX_LOT = 0.50
MARGIN_USAGE_LIMIT = 0.8

# Cost (lebih ke info/logging, bukan ngeblok eksekusi)
hidden_spread_per_001 = 0.037
commission_per_001    = 0.06
cost_per_001 = hidden_spread_per_001 + commission_per_001

MAGIC = 987654321

# True  -> hanya print signal & calon order, TIDAK kirim order ke MT5
# False -> kirim order beneran
USE_DEMO_MODE = False


# ================= MT5 CONNECTOR =================

def connect_to_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"[MT5] initialize() failed: {mt5.last_error()}")
    acc_info = mt5.account_info()
    if acc_info is None:
        raise RuntimeError(f"[MT5] account_info() is None: {mt5.last_error()}")
    print(f"[MT5] Connected. Login={acc_info.login}, Balance={acc_info.balance}, Equity={acc_info.equity}")


def shutdown_mt5():
    mt5.shutdown()
    print("[MT5] Disconnected.")


def get_symbol_info(symbol=SYMBOL):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"[MT5] Symbol {symbol} not found.")
    if not info.visible:
        mt5.symbol_select(symbol, True)
    return info


# ================= SESSION & LOT LOGIC =================

def in_trade_session(t: datetime) -> bool:
    """Samakan jam session dengan backtest."""
    if not USE_SESSION_FILTER:
        return True
    hour = t.hour
    in_london = 7 <= hour < 17
    in_ny = 12 <= hour < 22
    if TRADE_LONDON and in_london:
        return True
    if TRADE_NEWYORK and in_ny:
        return True
    return False


def calc_dynamic_lot(equity: float, entry_price: float) -> float:
    """Copas logic dari backtest untuk konsistensi."""
    if equity <= 0:
        return 0.0

    # 0.01 lot XAU ≈ $1 per 1.0 "pip"
    pip_value_per_001 = 1.0

    risk_amount = equity * (RISK_PERCENT / 100.0)
    loss_per_001 = SL_PIPS * pip_value_per_001

    if loss_per_001 <= 0:
        return 0.0

    units_001 = risk_amount / loss_per_001
    lot_by_risk = units_001 * 0.01

    # margin based
    max_lot_by_margin = (equity * MARGIN_USAGE_LIMIT) * LEVERAGE / (CONTRACT_SIZE * entry_price)

    lot = min(lot_by_risk, max_lot_by_margin, MAX_LOT)
    lot = max(lot, 0.0)

    if lot < MIN_LOT:
        return 0.0

    return lot


# ================= FEATURE PIPELINE =================

def build_features_from_candles(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    df_raw: data OHLCV dari MT5, kolom minimal:
      ['time','open','high','low','close','tick_volume', ...]
    Output: df_features dengan semua fitur dasar + advanced
            (harus mengandung semua feature_names model).
    """
    df = df_raw.copy()

    # Pastikan time dalam datetime
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], unit="s")

    # Pastikan tick_volume ada
    if "tick_volume" not in df.columns and "volume" in df.columns:
        df["tick_volume"] = df["volume"]

    # ========= BASE FEATURES (yang sekarang missing) =========

    # Returns (simple % change)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # EMAs
    df["ema_10"] = ta.ema(df["close"], length=10)
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_fast_slow_diff"] = df["ema_10"] - df["ema_50"]

    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)

    # Bollinger Bands (20, 2)
    bb = ta.bbands(df["close"], length=20, std=2)

    if isinstance(bb, pd.DataFrame):
        low_col_candidates  = [c for c in bb.columns if "BBL_" in c]
        mid_col_candidates  = [c for c in bb.columns if "BBM_" in c]
        high_col_candidates = [c for c in bb.columns if "BBU_" in c]

        low_col  = low_col_candidates[0]  if low_col_candidates  else None
        mid_col  = mid_col_candidates[0]  if mid_col_candidates  else None
        high_col = high_col_candidates[0] if high_col_candidates else None

        if low_col and mid_col and high_col:
            df["bb_low"]  = bb[low_col]
            df["bb_mid"]  = bb[mid_col]
            df["bb_high"] = bb[high_col]

            df["bb_width"] = (df["bb_high"] - df["bb_low"]) / (df["bb_mid"] + 1e-6)
            df["bb_pos"]   = (df["close"] - df["bb_low"]) / (
                (df["bb_high"] - df["bb_low"]) + 1e-6
            )

    # ATR 14 (versi base)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Stochastic (k & d)
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    if stoch is not None:
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # Time features
    df["dayofweek"] = df["time"].dt.dayofweek
    hour = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Session flags (versi base)
    df["sess_asia"] = hour.between(0, 7).astype(int)
    df["sess_london"] = hour.between(7, 15).astype(int)
    df["sess_ny"] = hour.between(12, 21).astype(int)

    # Volume regime
    df["vol_ema_20"] = ta.ema(df["tick_volume"], length=20)
    df["vol_ratio"] = df["tick_volume"] / (df["vol_ema_20"] + 1e-6)

    # ========= ADVANCED FEATURES (module lo sendiri) =========
    df = add_advanced_features(df)

    # Buang bar awal yang masih NaN
    df = df.dropna().reset_index(drop=True)

    return df


# ================= LOAD MODEL =================

print("[INFO] Loading models...")
xgb_data = joblib.load(XGB_MODEL)
rf_data  = joblib.load(RF_MODEL)

xgb_model = xgb_data["model"]
rf_model  = rf_data["model"]
scaler    = xgb_data["scaler"]
feature_names = xgb_data["feature_names"]

print(f"[INFO] Feature count: {len(feature_names)}")


# ================= HYBRID SIGNAL =================

def generate_hybrid_signal(df_features: pd.DataFrame):
    """
    Ambil sinyal untuk bar CLOSED terakhir di df_features.
    Return dict:
      {
        "time": datetime,
        "signal": -1 / 0 / 1,
        "xgb_cls": int,
        "xgb_conf": float,
        "rf_cls": int,
        "rf_conf": float,
      }
    """
    if df_features.empty:
        return None

    # Pastikan semua feature tersedia
    missing = [f for f in feature_names if f not in df_features.columns]
    if missing:
        print("[ERROR] Missing features:", missing)
        return None

    X = df_features[feature_names]          # <- biarin tetap DataFrame
    X_scaled = scaler.transform(X)          # scaler masih bisa handle DataFrame

    xgb_pred = xgb_model.predict(X_scaled)
    xgb_proba = xgb_model.predict_proba(X_scaled)

    i = len(df_features) - 1
    base_cls = int(xgb_pred[i])            # 0=SELL,1=NO_TRADE,2=BUY
    base_conf = float(xgb_proba[i, base_cls])

    t = df_features.iloc[i]["time"]

    xgb_cls = base_cls
    xgb_conf = base_conf
    rf_cls = 1
    rf_conf = 0.0
    final_signal = 0

    # Filter utama: XGB bilang no-trade / confidence kurang
    if base_cls == 1 or base_conf < CONF_XGB:
        return {
            "time": t,
            "signal": 0,
            "xgb_cls": xgb_cls,
            "xgb_conf": xgb_conf,
            "rf_cls": rf_cls,
            "rf_conf": rf_conf,
        }

    # RF decide arah (0=SELL,1=NO_TRADE,2=BUY)
    rf_proba_row = rf_model.predict_proba(X_scaled[i:i+1])[0]
    rf_cls = int(np.argmax(rf_proba_row))
    rf_conf = float(rf_proba_row[rf_cls])

    if rf_cls == 0:
        final_signal = -1
    elif rf_cls == 2:
        final_signal = 1
    else:
        final_signal = 0

    return {
        "time": t,
        "signal": final_signal,
        "xgb_cls": xgb_cls,
        "xgb_conf": xgb_conf,
        "rf_cls": rf_cls,
        "rf_conf": rf_conf,
    }


# ================= ORDER EXECUTION =================

def send_order(direction: str, lot: float, tp_pips: float, sl_pips: float):
    """
    direction: "BUY" / "SELL"
    TP/SL: dalam "pip harga" (1.0 = 1.0 di price), sama kayak backtest.
    """
    info = get_symbol_info()
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("[ORDER] No tick info:", mt5.last_error())
        return None

    price = tick.ask if direction == "BUY" else tick.bid

    # Di backtest, TP/SL langsung: entry_price ± TP_PIPS
    if direction == "BUY":
        tp_price = price + tp_pips
        sl_price = price - sl_pips
    else:
        tp_price = price - tp_pips
        sl_price = price + sl_pips

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": 20,
        "magic": MAGIC,
        "comment": "ml_hybrid_live",
        "type_filling": mt5.ORDER_FILLING_FOK,
        "type_time": mt5.ORDER_TIME_GTC,
    }

    if USE_DEMO_MODE:
        print(f"[DEMO ORDER] {direction} {lot:.3f} @ {price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}")
        return None

    result = mt5.order_send(request)
    if result is None:
        print("[ORDER] order_send returned None:", mt5.last_error())
        return None

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ORDER] FAILED retcode={result.retcode}, comment={result.comment}")
    else:
        print(f"[ORDER] SUCCESS ticket={result.order}, {direction} {lot:.3f} @ {price:.2f}")
    return result


# ================= MAIN LOOP =================

def main_loop():
    connect_to_mt5()
    info = get_symbol_info()
    print(f"[INFO] Symbol {SYMBOL} digits={info.digits}, point={info.point}")

    last_bar_time = None
    last_trade_bar_index = -9999

    while True:
        try:
            # Ambil history M5 terakhir
            rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS_HISTORY)
            if rates is None:
                print("[LOOP] Failed to get rates:", mt5.last_error())
                time.sleep(POLL_SECONDS)
                continue

            df_raw = pd.DataFrame(rates)
            df_raw["time"] = pd.to_datetime(df_raw["time"], unit="s")

            if df_raw.empty:
                time.sleep(POLL_SECONDS)
                continue

            # Hanya pakai closed bar (exclude bar yang masih jalan)
            if len(df_raw) < 2:
                time.sleep(POLL_SECONDS)
                continue

            df_closed = df_raw.iloc[:-1].copy()
            current_last_time = df_closed["time"].iloc[-1]

            # Kalau belum ada bar baru, skip
            if last_bar_time is not None and current_last_time <= last_bar_time:
                time.sleep(POLL_SECONDS)
                continue

            last_bar_time = current_last_time

            # Build fitur
            df_feat = build_features_from_candles(df_closed)
            if df_feat.empty:
                print("[LOOP] df_feat empty (indikator belum cukup data?)")
                time.sleep(POLL_SECONDS)
                continue

            signal_info = generate_hybrid_signal(df_feat)
            if signal_info is None:
                time.sleep(POLL_SECONDS)
                continue

            bar_time = signal_info["time"]
            signal = signal_info["signal"]

            print(
                f"[SIGNAL] {bar_time} | sig={signal} | "
                f"xgb_cls={signal_info['xgb_cls']} ({signal_info['xgb_conf']:.3f}) | "
                f"rf_cls={signal_info['rf_cls']} ({signal_info['rf_conf']:.3f})"
            )

            # No trade
            if signal == 0:
                time.sleep(POLL_SECONDS)
                continue

            # Session filter
            if not in_trade_session(bar_time):
                print("[FILTER] Out of session, skip.")
                time.sleep(POLL_SECONDS)
                continue

            # Cooldown pakai index bar (mirip backtest)
            bar_index = len(df_feat) - 1
            if bar_index <= last_trade_bar_index + COOLDOWN_BARS:
                print("[FILTER] Cooldown active, skip.")
                time.sleep(POLL_SECONDS)
                continue

            # Equity real dari MT5
            acc = mt5.account_info()
            if acc is None:
                print("[ERROR] account_info is None:", mt5.last_error())
                time.sleep(POLL_SECONDS)
                continue
            current_equity = acc.equity

            entry_price = float(df_feat["close"].iloc[-1])
            lot = calc_dynamic_lot(current_equity, entry_price)
            if lot <= 0:
                print(f"[FILTER] Lot <= 0 (equity={current_equity:.2f}), skip.")
                time.sleep(POLL_SECONDS)
                continue

            direction = "BUY" if signal == 1 else "SELL"
            print(f"[TRADE] {direction} signal | equity={current_equity:.2f} | lot={lot:.3f}")

            send_order(direction, lot, TP_PIPS, SL_PIPS)

            last_trade_bar_index = bar_index

            time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("[MAIN] KeyboardInterrupt, exiting...")
            break
        except Exception as e:
            print("[ERROR] Exception in main_loop:", e)
            time.sleep(POLL_SECONDS)

    shutdown_mt5()


if __name__ == "__main__":
    main_loop()
