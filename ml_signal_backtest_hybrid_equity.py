"""
Hybrid Backtest with Equity Simulation
- Include raw pip profit
- Include Exness cost (spread + commission)
- Track equity growth, drawdown, winrate
"""

import pandas as pd
import numpy as np
import joblib

# === CONFIG ===
CSV_INPUT = "xau_M5_ml_dataset_ADV.csv"
XGB_MODEL = "xgb_scalping_model_v3_tuned.pkl"
RF_MODEL  = "rf_scalping_model_v3_balanced.pkl"

CONF_XGB = 0.60     # Confidence threshold
TP_PIPS = 1.0
SL_PIPS = 0.5
MAX_BARS_HOLD = 10

# === EXNESS REAL COST CONFIG ===
starting_balance = 100.0
lot_multiplier = 5.0   # 0.01 lot per 10x
hidden_spread_per_001 = 0.037
commission_per_001 = 0.06
cost_per_trade = hidden_spread_per_001 + commission_per_001

# === LOAD MODELS & DATA ===
print("[INFO] Loading models & dataset...")
df = pd.read_csv(CSV_INPUT)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

xgb_data = joblib.load(XGB_MODEL)
rf_data  = joblib.load(RF_MODEL)

xgb_model = xgb_data["model"]
rf_model  = rf_data["model"]
scaler    = xgb_data["scaler"]
feature_names = xgb_data["feature_names"]

X_scaled = scaler.transform(df[feature_names])
prices = df["close"].values

# === HYBRID SIGNAL GENERATION ===
print("[INFO] Generating hybrid signals...")
xgb_pred = xgb_model.predict(X_scaled)
xgb_proba = xgb_model.predict_proba(X_scaled)

hybrid_signal = np.zeros(len(xgb_pred), dtype=int)

for i in range(len(xgb_pred)):
    base_cls = int(xgb_pred[i])  # 0=SELL,1=NO_TRADE,2=BUY
    base_conf = float(xgb_proba[i, base_cls])

    if base_cls == 1 or base_conf < CONF_XGB:
        continue

    rf_cls = int(rf_model.predict(X_scaled[i:i+1])[0])

    # Final mapping: -1 = SELL, 1 = BUY, 0 = NO_TRADE
    if rf_cls == 0:
        hybrid_signal[i] = -1
    elif rf_cls == 2:
        hybrid_signal[i] = 1
    else:
        hybrid_signal[i] = 0

print("[INFO] Hybrid signal distribution (final):")
unique, counts = np.unique(hybrid_signal, return_counts=True)
print(dict(zip(unique, counts)))

# === BACKTEST WITH EQUITY SIMULATION ===
results = []
equity = [starting_balance]
max_equity = starting_balance
max_dd = 0

for i in range(len(hybrid_signal) - MAX_BARS_HOLD):
    signal = hybrid_signal[i]
    if signal == 0:
        continue

    entry = prices[i]
    direction = "BUY" if signal == 1 else "SELL"

    tp_level = entry + TP_PIPS if direction == "BUY" else entry - TP_PIPS
    sl_level = entry - SL_PIPS if direction == "BUY" else entry + SL_PIPS

    hit = None
    exit_price = entry

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

    profit_raw = TP_PIPS if hit=="TP" else -SL_PIPS if hit=="SL" else 0
    profit_after_cost = profit_raw * lot_multiplier - cost_per_trade

    new_equity = equity[-1] + profit_after_cost
    equity.append(new_equity)

    if new_equity > max_equity:
        max_equity = new_equity
    dd = max_equity - new_equity
    if dd > max_dd:
        max_dd = dd

    results.append([i, direction, entry, exit_price, profit_raw, profit_after_cost, new_equity])

results_df = pd.DataFrame(results, columns=["index","signal","entry","exit","profit_raw","profit_after_cost","equity"])

total = len(results_df)
wins = len(results_df[results_df.profit_after_cost > 0])
losses = len(results_df[results_df.profit_after_cost < 0])
winrate = wins / total * 100 if total else 0
avg_profit = results_df["profit_after_cost"].mean() if total else 0

print("\n===== HYBRID EQUITY BACKTEST RESULT =====")
print(f"Starting Balance: ${starting_balance:.2f}")
print(f"Final Balance:    ${equity[-1]:.2f}")
print(f"Net Profit:       ${equity[-1] - starting_balance:.2f}")
print(f"Total Trades:     {total}")
print(f"Win Trades:       {wins}")
print(f"Loss Trades:      {losses}")
print(f"Winrate:          {winrate:.2f}%")
print(f"Average Profit:   {avg_profit:.3f} $/trade")
print(f"Max Drawdown:     ${max_dd:.2f}")
print(f"Profit Factor:    {results_df[results_df.profit_after_cost>0]['profit_after_cost'].sum() / abs(results_df[results_df.profit_after_cost<0]['profit_after_cost'].sum()):.2f}")

results_df.to_csv("ml_backtest_hybrid_equity.csv", index=False)
print("\n[SAVED] → ml_backtest_hybrid_equity.csv")
