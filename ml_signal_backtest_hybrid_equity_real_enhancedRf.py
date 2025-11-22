"""
ml_signal_backtest_hybrid_equity_real.py

Hybrid Backtest (XGB filter + RF direction) with:
- Dynamic lot based on equity & risk per trade
- Leverage & margin check (simulate cannot overlot)
- Optional date range filter
- Session filter (London & NY)
- Trade cooldown
- Confidence logging (XGB & RF)

Final signal format:
- -1 = SELL
-  0 = NO_TRADE
-  1 = BUY
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ========= CONFIG =========
CSV_INPUT = "xau_M5_ml_dataset_ADV.csv"
XGB_MODEL = "xgb_scalping_model_v4_realistic_fixed.pkl"
RF_MODEL  = "rf_direction_model_v5_realistic_fixed.pkl"

# --- Date filter ---
USE_DATE_FILTER = True
START_DATE = "2025-10-01"
END_DATE   = "2025-11-18"

CONF_XGB = 0.30         # Confidence threshold XGB
TP_PIPS = 1.0
SL_PIPS = 0.5
MAX_BARS_HOLD = 10

# Session filter
USE_SESSION_FILTER = True
TRADE_LONDON = True
TRADE_NEWYORK = True

# Trade cooldown (bars after a trade closes)
COOLDOWN_BARS = 3

# Exness-like cost
hidden_spread_per_001 = 0.037
commission_per_001    = 0.06
cost_per_001 = hidden_spread_per_001 + commission_per_001   # per 0.01 lot

# Account / leverage setup
START_BALANCE = 20.0
LEVERAGE = 1000
CONTRACT_SIZE = 100     # XAU contract size (100 oz)
RISK_PERCENT = 5.0      # % equity per trade (risk to SL)
MIN_LOT = 0.01
MAX_LOT = 1.0          # safety cap (lo udah set ini)
MARGIN_USAGE_LIMIT = 0.8  # max 80% equity boleh dipakai jadi margin


def calc_dynamic_lot(equity, entry_price):
    """Hitung lot berdasarkan risk per trade & leverage/margin."""
    if equity <= 0:
        return 0.0

    # 0.01 lot XAU ≈ $1 per 1.0 "pip" (move)
    pip_value_per_001 = 1.0  # $ per pip per 0.01 lot

    risk_amount = equity * (RISK_PERCENT / 100.0)
    loss_per_001 = SL_PIPS * pip_value_per_001  # loss jika SL kena per 0.01 lot

    if loss_per_001 <= 0:
        return 0.0

    # Berapa unit 0.01 lot yang boleh (berdasarkan risk)
    units_001 = risk_amount / loss_per_001
    lot_by_risk = units_001 * 0.01

    # Margin-based max lot
    # Margin = lot * CONTRACT_SIZE * price / LEVERAGE
    max_lot_by_margin = (equity * MARGIN_USAGE_LIMIT) * LEVERAGE / (CONTRACT_SIZE * entry_price)

    lot = min(lot_by_risk, max_lot_by_margin, MAX_LOT)
    lot = max(lot, 0.0)

    if lot < MIN_LOT:
        if equity < 200:
            return MIN_LOT
        return 0.0

    return lot

def normalize_lot(lot):
    MIN_LOT = 0.01   
    LOT_STEP = 0.01 
    MAX_LOT = 100

    if lot < MIN_LOT:
        return 0.0

    lot = round(lot / LOT_STEP) * LOT_STEP
    return min(max(lot, MIN_LOT), MAX_LOT)

def in_trade_session(t):
    """Filter jam trading (simple): London & NY sessions."""
    if not USE_SESSION_FILTER:
        return True
    hour = t.hour  # asumsi timezone sama dengan data CSV (broker time)
    in_london = 7 <= hour < 17   # approx London session
    in_ny     = 12 <= hour < 22  # approx New York session

    if TRADE_LONDON and in_london:
        return True
    if TRADE_NEWYORK and in_ny:
        return True
    return False


print("[INFO] Loading models & dataset...")
df = pd.read_csv(CSV_INPUT)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# ====== APPLY DATE FILTER (IF ENABLED) ======
if USE_DATE_FILTER:
    df_range = df[(df["time"] >= START_DATE) & (df["time"] <= END_DATE)].copy()
    print(f"[INFO] Date filter ON: {START_DATE} to {END_DATE}")
    print(f"[INFO] Rows in range: {len(df_range)}")
    df = df_range.reset_index(drop=True)
else:
    print("[INFO] Date filter OFF: using full dataset")
    print(f"[INFO] Total rows: {len(df)}")

xgb_data = joblib.load(XGB_MODEL)
rf_data  = joblib.load(RF_MODEL)

xgb_model = xgb_data["model"]
rf_model  = rf_data["model"]
scaler    = xgb_data["scaler"]
feature_names = xgb_data["feature_names"]

X = df[feature_names]
X_scaled = scaler.transform(X)
prices = df["close"].values

# ========= HYBRID SIGNAL + CONFIDENCE LOGGING =========
print("[INFO] Generating hybrid signals...")

xgb_pred = xgb_model.predict(X_scaled)             # 0=SELL,1=NO_TRADE,2=BUY
xgb_proba = xgb_model.predict_proba(X_scaled)      # [N, 3]

hybrid_signal = np.zeros(len(xgb_pred), dtype=int)  # -1,0,1
xgb_conf_arr = np.zeros(len(xgb_pred))
xgb_cls_arr = np.zeros(len(xgb_pred), dtype=int)
rf_cls_arr = np.zeros(len(xgb_pred), dtype=int)
rf_conf_arr = np.zeros(len(xgb_pred))

for i in range(len(xgb_pred)):
    base_cls = int(xgb_pred[i])      # 0,1,2
    base_conf = float(xgb_proba[i, base_cls])

    xgb_cls_arr[i] = base_cls
    xgb_conf_arr[i] = base_conf

    # ===== Hybrid Decision v2 =====
    if base_cls == 1 and base_conf >= CONF_XGB:
        # XGB confident NO_TRADE → Skip trade
        continue

    # For BUY / SELL with strong confidence → follow XGB
    if base_cls in [0, 2] and base_conf >= CONF_XGB:
        hybrid_signal[i] = -1 if base_cls == 0 else 1
        continue

    # Otherwise → uncertain or NO_TRADE → let RF decide direction
    rf_proba_row = rf_model.predict_proba(X_scaled[i:i+1])[0]
    rf_cls = int(np.argmax(rf_proba_row))
    rf_conf = float(rf_proba_row[rf_cls])

    RF_CONF_MIN = 0.60  # Threshold RF, bisa tuning

    # Final mapping with RF confidence filtering
    if rf_cls == 0 and rf_conf >= RF_CONF_MIN:
        hybrid_signal[i] = -1
    elif rf_cls == 2 and rf_conf >= RF_CONF_MIN:
        hybrid_signal[i] = 1
    else:
        hybrid_signal[i] = 0


print("[INFO] Hybrid signal distribution (final):")
unique, counts = np.unique(hybrid_signal, return_counts=True)
print(dict(zip(unique, counts)))

# ========= EQUITY BACKTEST =========

equity = [START_BALANCE]
max_equity = START_BALANCE
max_dd = 0.0
results = []
last_trade_exit_idx = -9999

for i in range(len(hybrid_signal) - MAX_BARS_HOLD):
    signal = hybrid_signal[i]
    if signal == 0:
        continue

    # Session filter
    current_time = df.loc[i, "time"]
    if not in_trade_session(current_time):
        continue

    # Trade cooldown (skip jika belum lewat COOLDOWN_BARS dari trade sebelumnya)
    if i <= last_trade_exit_idx + COOLDOWN_BARS:
        continue

    current_equity = equity[-1]
    entry_price = prices[i]
    direction = "BUY" if signal == 1 else "SELL"

    lot = calc_dynamic_lot(current_equity, entry_price)
    lot = normalize_lot(lot)
    if lot <= 0:
        continue

    margin_required = lot * CONTRACT_SIZE * entry_price / LEVERAGE
    if margin_required > current_equity * MARGIN_USAGE_LIMIT:
        continue

    # TP / SL price
    tp_level = entry_price + TP_PIPS if direction == "BUY" else entry_price - TP_PIPS
    sl_level = entry_price - SL_PIPS if direction == "BUY" else entry_price + SL_PIPS

    hit = None
    exit_price = entry_price
    j = i  # init j buat fallback

    for j in range(i+1, min(i + MAX_BARS_HOLD, len(prices))):
        high = df.loc[j, "high"]
        low = df.loc[j, "low"]

        if direction == "BUY":
            if high >= tp_level:
                hit = "TP"; exit_price = tp_level; break
            if low <= sl_level:
                hit = "SL"; exit_price = sl_level; break
        else:
            if low <= tp_level:
                hit = "TP"; exit_price = tp_level; break
            if high >= sl_level:
                hit = "SL"; exit_price = sl_level; break

    # Raw pip profit
    if hit == "TP":
        profit_raw_pips = TP_PIPS
    elif hit == "SL":
        profit_raw_pips = -SL_PIPS
    else:
        last_price = prices[j]
        if direction == "BUY":
            profit_raw_pips = last_price - entry_price
        else:
            profit_raw_pips = entry_price - last_price

    # 0.01 lot XAU ≈ $1 per 1.0 pip
    pip_value_per_lot = 100.0  # 1.0 lot = 100 * $1
    profit_gross = profit_raw_pips * pip_value_per_lot * lot

    cost_trade = cost_per_001 * (lot / 0.01)
    profit_net = profit_gross - cost_trade

    new_equity = current_equity + profit_net
    equity.append(new_equity)

    if new_equity > max_equity:
        max_equity = new_equity
    dd = max_equity - new_equity
    if dd > max_dd:
        max_dd = dd

    last_trade_exit_idx = j

    results.append([
        i,
        df.loc[i, "time"],
        direction,
        lot,
        entry_price,
        exit_price,
        profit_raw_pips,
        profit_net,
        new_equity,
        margin_required,
        xgb_cls_arr[i],
        xgb_conf_arr[i],
        rf_cls_arr[i],
        rf_conf_arr[i],
        signal
    ])

results_df = pd.DataFrame(results, columns=[
    "index","time","signal_dir","lot","entry","exit",
    "profit_pips","profit_after_cost","equity","margin_required",
    "xgb_cls","xgb_conf","rf_cls","rf_conf","final_signal"
])

total = len(results_df)
wins = len(results_df[results_df["profit_after_cost"] > 0])
losses = len(results_df[results_df["profit_after_cost"] < 0])

winrate = wins / total * 100 if total else 0.0
avg_profit = results_df["profit_after_cost"].mean() if total else 0.0
final_balance = equity[-1]
net_profit = final_balance - START_BALANCE

gross_profit = results_df[results_df["profit_after_cost"] > 0]["profit_after_cost"].sum()
gross_loss   = results_df[results_df["profit_after_cost"] < 0]["profit_after_cost"].sum()
profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf

print("\n===== HYBRID REALISTIC EQUITY BACKTEST RESULT =====")
print(f"Date Range:       {START_DATE if USE_DATE_FILTER else 'FULL'} to {END_DATE if USE_DATE_FILTER else 'FULL'}")
print(f"Starting Balance: ${START_BALANCE:.2f}")
print(f"Final Balance:    ${final_balance:.2f}")
print(f"Net Profit:       ${net_profit:.2f}")
print(f"Total Trades:     {total}")
print(f"Win Trades:       {wins}")
print(f"Loss Trades:      {losses}")
print(f"Winrate:          {winrate:.2f}%")
print(f"Average Profit:   {avg_profit:.3f} $/trade")
print(f"Max Drawdown:     ${max_dd:.2f}")
print(f"Profit Factor:    {profit_factor:.2f}")

results_df.to_csv("ml_backtest_hybrid_equity_real_override.csv", index=False)
print("\n[SAVED] → ml_backtest_hybrid_equity_real_override.csv")
