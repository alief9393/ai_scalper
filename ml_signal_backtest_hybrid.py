"""
ml_signal_backtest_hybrid.py

Hybrid Backtest:
- Model 1 (XGB tuned) → Filter NO_TRADE (regime filter + confidence)
- Model 2 (RF balanced) → Decide BUY or SELL only
- Output sinyal final: -1 (SELL), 0 (NO_TRADE), 1 (BUY)
"""

import pandas as pd
import joblib
import numpy as np

# ===== CONFIG =====
CSV_INPUT = "xau_M5_ml_dataset_ADV.csv"        # dataset advanced
XGB_MODEL = "xgb_scalping_model_v3_tuned.pkl"
RF_MODEL  = "rf_scalping_model_v3_balanced.pkl"

CONF_XGB = 0.60   # Threshold confidence XGBoost (boleh nanti dicoba 0.55–0.65)
# (optional) bisa tambahin CONF_RF kalau mau filter RF juga

tp_pips = 1.0
sl_pips = 0.5
max_bars_hold = 10

# ===== LOAD DATA & MODEL =====
print("[INFO] Loading dataset & hybrid models...")
df = pd.read_csv(CSV_INPUT)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

xgb_data = joblib.load(XGB_MODEL)
rf_data  = joblib.load(RF_MODEL)

xgb_model = xgb_data["model"]
rf_model  = rf_data["model"]
scaler    = xgb_data["scaler"]
feature_names = xgb_data["feature_names"]

X = df[feature_names]
X_scaled = scaler.transform(X)
prices = df["close"].values

print("[INFO] Running hybrid classification on full dataset...")

# ===== STEP 1: XGBoost filter =====
xgb_pred = xgb_model.predict(X_scaled)           # 0=SELL, 1=NO_TRADE, 2=BUY
xgb_proba = xgb_model.predict_proba(X_scaled)    # [N, 3]

# ===== Prepare final signal array (in final domain: -1, 0, 1) =====
# -1 = SELL, 0 = NO_TRADE, 1 = BUY
hybrid_signal = np.zeros(len(xgb_pred), dtype=int)  # start semua NO_TRADE

for i in range(len(xgb_pred)):
    base_cls = int(xgb_pred[i])  # 0,1,2
    base_conf = float(xgb_proba[i, base_cls])

    # Filter 1: kalau XGB bilang NO_TRADE atau confidence rendah => tetap 0 (NO_TRADE)
    if base_cls == 1 or base_conf < CONF_XGB:
        continue  # hybrid_signal[i] tetap 0

    # Kalau lolos filter: market dianggap OK untuk trade
    # ===== STEP 2: RandomForest direction decision =====
    rf_cls = int(rf_model.predict(X_scaled[i:i+1])[0])   # 0,1,2

    # Kalau RF kasih NO_TRADE (1), fallback ke XGB direction
    if rf_cls == 1:
        final_cls = base_cls  # pakai 0 atau 2 dari XGB
    else:
        final_cls = rf_cls    # pakai arah dari RF

    # Map final_cls (0,1,2) → hybrid_signal final (-1,0,1)
    if final_cls == 0:
        hybrid_signal[i] = -1  # SELL
    elif final_cls == 2:
        hybrid_signal[i] = 1   # BUY
    else:
        hybrid_signal[i] = 0   # NO_TRADE (should be rare di sini)

print("[INFO] Hybrid signal distribution (final):")
unique, counts = np.unique(hybrid_signal, return_counts=True)
print(dict(zip(unique, counts)))

# ===== BACKTEST =====
results = []

for i in range(len(hybrid_signal) - max_bars_hold):
    signal = hybrid_signal[i]
    if signal == 0:   # NO_TRADE
        continue

    entry = prices[i]
    direction = "BUY" if signal == 1 else "SELL"

    if direction == "BUY":
        tp_level = entry + tp_pips
        sl_level = entry - sl_pips
    else:
        tp_level = entry - tp_pips
        sl_level = entry + sl_pips

    exit_price = entry
    hit = None

    for j in range(i+1, min(i + max_bars_hold, len(prices))):
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

    profit = tp_pips if hit == "TP" else -sl_pips if hit == "SL" else 0.0
    results.append([i, direction, entry, exit_price, profit])

results_df = pd.DataFrame(results, columns=["index","signal","entry","exit","profit_raw"])

total = len(results_df)
wins = len(results_df[results_df["profit_raw"] > 0])
losses = len(results_df[results_df["profit_raw"] < 0])
winrate = wins / total * 100 if total > 0 else 0.0
avg_p = results_df["profit_raw"].mean() if total > 0 else 0.0

print("\n===== HYBRID STRATEGY BACKTEST RESULT =====")
print(f"Total Trades: {total}")
print(f"Win Trades: {wins}")
print(f"Loss Trades: {losses}")
print(f"Winrate: {winrate:.2f}%")
print(f"Average Profit per trade: {avg_p:.3f}")

results_df.to_csv("ml_backtest_hybrid_signals.csv", index=False)
print("\n[SAVED] Hybrid trade results → ml_backtest_hybrid_signals.csv")
