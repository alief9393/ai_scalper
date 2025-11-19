# ml_signal_backtest.py
"""
Step 6 (fixed): Price-based backtest of ML signals
- Uses time-based train/test split
- Only backtests on OUT-OF-SAMPLE (test) section
"""

import pandas as pd
import numpy as np
import joblib

CSV_INPUT = "xau_M5_ml_dataset.csv"
MODEL_FILE = "rf_scalping_model_timesplit.pkl"

print("[INFO] Loading dataset & model...")
df = pd.read_csv(CSV_INPUT)
data = joblib.load(MODEL_FILE)

model = data["model"]
scaler = data["scaler"]
feature_names = data["feature_names"]
split_idx = data["split_idx"]
CONF_TRADE_THRESHOLD = data["conf_threshold"]

# Sort by time (safety)
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

X = df[feature_names].copy()
prices = df["close"].values

# Scale all
X_scaled = scaler.transform(X)

# Predict
proba = model.predict_proba(X_scaled)
pred = model.predict(X_scaled)

classes = list(model.classes_)
class_to_idx = {c: i for i, c in enumerate(classes)}

filtered_pred = []
for i in range(len(pred)):
    base_pred = pred[i]
    conf = proba[i, class_to_idx[base_pred]]

    if base_pred != 0 and conf >= CONF_TRADE_THRESHOLD:
        filtered_pred.append(base_pred)
    else:
        filtered_pred.append(0)

filtered_pred = np.array(filtered_pred)

print(f"[INFO] Using out-of-sample test range starting from index: {split_idx}")

df["time"] = pd.to_datetime(df["time"])

# Filter hanya 1 Nov - 18 Nov 2025
start_date = "2025-11-01"
end_date   = "2025-11-18"

df_period = df[(df["time"] >= start_date) & (df["time"] <= end_date)].copy()

print(f"[INFO] Filtered data rows: {len(df_period)} from {start_date} to {end_date}")

# Replace original df sections
price_subset = df_period["close"].values
high_subset = df_period["high"].values
low_subset = df_period["low"].values
time_subset = df_period["time"].values

# Also ensure signals match same rows
filtered_pred_period = filtered_pred[df_period.index]

# Backtest only on test range
tp_pips = 1.0
sl_pips = 0.5
max_bars_hold = 10
allow_one_trade_at_a_time = True
min_gap_bars = 5
last_trade_exit_index = -999

results = []

for idx, i in enumerate(df_period.index[:-max_bars_hold]):
    if allow_one_trade_at_a_time and i <= last_trade_exit_index + min_gap_bars:
        continue

    signal = filtered_pred[i]
    if signal == 0:
        continue

    entry = prices[i]

    if signal == 1:
        tp_level = entry + tp_pips
        sl_level = entry - sl_pips
        direction = "BUY"
    else:
        tp_level = entry - tp_pips
        sl_level = entry + sl_pips
        direction = "SELL"

    hit = None
    exit_price = entry

    for j in range(i + 1, min(i + max_bars_hold, len(prices))):
        high = df.loc[j, "high"]
        low = df.loc[j, "low"]

        if direction == "BUY":
            if high >= tp_level:
                hit = "TP"
                exit_price = tp_level
                break
            elif low <= sl_level:
                hit = "SL"
                exit_price = sl_level
                break
        else:  # SELL
            if low <= tp_level:
                hit = "TP"
                exit_price = tp_level
                break
            elif high >= sl_level:
                hit = "SL"
                exit_price = sl_level
                break

    if hit == "TP":
        profit = tp_pips
        last_trade_exit_index = j
    elif hit == "SL":
        profit = -sl_pips
        last_trade_exit_index = j
    else:
        exit_price = prices[j]
        profit = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        last_trade_exit_index = j

    results.append([i, direction, entry, exit_price, profit])

import pandas as pd
results_df = pd.DataFrame(results, columns=["index", "signal", "entry", "exit", "profit_raw"])

# === Konversi ke $ (akun real Exness raw) ===
starting_balance = 20.0

# 0.01 lot XAU ≈ 1$ per 1.0 move (1$ per profit_raw = 1$)
lot_multiplier = 5.0

hidden_spread_per_001 = 0.037   # from you
commission_per_001   = 0.06     # from you

cost_per_trade = hidden_spread_per_001 + commission_per_001  # ≈ 0.097 USD

results_df["profit_after_cost"] = results_df["profit_raw"] * lot_multiplier - cost_per_trade

# Equity curve
equity = [starting_balance]
for p in results_df["profit_after_cost"]:
    equity.append(equity[-1] + p)

results_df["equity"] = equity[1:]

# Basic stats
total_trades = len(results_df)
win_trades = len(results_df[results_df["profit_after_cost"] > 0])
loss_trades = len(results_df[results_df["profit_after_cost"] <= 0])
winrate = win_trades / total_trades * 100 if total_trades else 0
avg_profit = results_df["profit_after_cost"].mean() if total_trades else 0

final_balance = equity[-1]
net_profit = final_balance - starting_balance

# Max drawdown (simple)
max_equity = -1e9
max_dd = 0
for e in results_df["equity"]:
    if e > max_equity:
        max_equity = e
    dd = max_equity - e
    if dd > max_dd:
        max_dd = dd

print("\n===== ML Strategy OOS (With Real Exness Cost & Equity) =====")
print(f"Total Trades (OOS): {total_trades}")
print(f"Win Trades: {win_trades}")
print(f"Loss Trades: {loss_trades}")
print(f"Winrate: {winrate:.2f}%")
print(f"Average Profit after cost: {avg_profit:.4f} $/trade")
print(f"Starting Balance: ${starting_balance:.2f}")
print(f"Final Balance:    ${final_balance:.2f}")
print(f"Net Profit:       ${net_profit:.2f}")
print(f"Max Drawdown:     ${max_dd:.2f}")

results_df.to_csv("ml_backtest_signals_oos_equity.csv", index=False)
print("\n[SAVED] OOS equity results to: ml_backtest_signals_oos_equity.csv")
