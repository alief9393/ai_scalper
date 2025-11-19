"""
mt5_trading_bridge.py
Live trading bridge between MT5 & ML Signal Engine

Features:
- Ambil candle real-time dari MT5 (XAUUSD, M5)
- Kirim data ke ML model => BUY / SELL / NO_TRADE
- Open order otomatis (0.02 lot misalnya)
- Auto-apply SL/TP
- Hindari multiple open positions
- Logging hasil trading
"""

import time
import MetaTrader5 as mt5
import pandas as pd
from ml_live_signal import MLSignalEngine
import pytz
from datetime import datetime, timedelta


# ===== USER CONFIG =====
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M5
LOT_SIZE = 0.05           # Realistic for $100-$200
TP_PIPS = 1.0             # price-based like backtest
SL_PIPS = 0.5
MAGIC = 20251118
MAX_SPREAD = 4.0          # filter execution (pips)
CHECK_INTERVAL = 60       # seconds between signals
ALLOW_ONE_TRADE_ONLY = True


# ===== MT5 CONNECT =====
def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    print("[INFO] Connected to MT5")


# ===== GET LATEST DATA FROM MT5 =====
def get_latest_data():
    utc_from = datetime.now(pytz.utc) - timedelta(days=30)
    utc_to = datetime.now(pytz.utc)

    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, utc_from, utc_to)
    if rates is None:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# ===== CHECK OPEN POSITIONS =====
def has_open_trade():
    if not ALLOW_ONE_TRADE_ONLY:
        return False
    positions = mt5.positions_get(symbol=SYMBOL)
    return len(positions) > 0


# ===== CALCULATE SL/TP LEVELS =====
def calc_levels(direction, entry_price):
    if direction == "BUY":
        sl = entry_price - SL_PIPS
        tp = entry_price + TP_PIPS
    else:
        sl = entry_price + SL_PIPS
        tp = entry_price - TP_PIPS
    return sl, tp


# ===== SEND ORDER TO MT5 =====
def send_order(direction):
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        print("[ERROR] Failed to get tick info")
        return False

    price = tick.ask if direction == "BUY" else tick.bid
    sl, tp = calc_levels(direction, price)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "magic": MAGIC,
        "deviation": 10,
        "comment": f"ML {direction}",
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] order_send failed: {result.retcode}, {result.comment}")
        return False

    print(f"[TRADE EXECUTED] {direction} @ {price} | SL={sl} | TP={tp}")
    return True


# ===== MAIN LOOP =====
def main():
    connect_mt5()
    engine = MLSignalEngine()

    while True:
        print("\n[INFO] Checking for new signal...")

        # Jangan ambil sinyal baru kalau masih ada posisi floating
        if has_open_trade():
            print("[INFO] Open trade found → skip signal check.")
            time.sleep(CHECK_INTERVAL)
            continue

        from xau_feature_engineering_live import build_live_features

        df_raw = get_latest_data()
        df_live = build_live_features(df_raw)

        if df_live is None or len(df_live) < 50:
            print("[WARNING] MT5 data fetch issue.")
            time.sleep(CHECK_INTERVAL)
            continue

        # === (IMPORTANT) Append Features ===
        # Normally we would reapply feature_engineering_live
        # But here we assume your df already includes full features.
        # If not, we will build xau_feature_engineering_live.py later.

        # Use last row to predict
        signal_data = engine.predict_signal(df_live)

        print(f"[SIGNAL] {signal_data}")

        if signal_data["signal"] in ["BUY", "SELL"]:
            print(f"[ACTION] Sending order: {signal_data['signal']}")
            send_order(signal_data["signal"])
        else:
            print("[NO TRADE] Confidence not high enough.")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
