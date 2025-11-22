"""
train_rf_direction_v4_balanced.py

Goal:
Train RandomForest model khusus untuk menentukan arah BUY / SELL
(lebih agresif, balanced, tanpa dominasi NO_TRADE).

✔ Event-based labeling (TP/SL realistic, bukan fixed-horizon)
✔ Drop NO_TRADE / sideways low movement
✔ Handle imbalance (class_weight + SMOTE)
✔ Save RF model (.pkl) for direct live use
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from imblearn.over_sampling import SMOTE

# ===== CONFIG =====
CSV_INPUT = "xau_M5_ml_dataset_ADV.csv"
TP_PIPS = 1.0
SL_PIPS = 0.5
MAX_HOLD = 10  # max bar after entry to monitor TP/SL
MODEL_OUTPUT = "rf_direction_model_v4_balanced.pkl"

# ===================================
# 1) Load Data
# ===================================
print("[INFO] Loading dataset...")
df = pd.read_csv(CSV_INPUT)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Ensure essential columns exist
required_cols = ["open", "high", "low", "close"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"[ERROR] Missing column: {col}")

# ===================================
# 2) Event-based Labeling (TP/SL logic)
# ===================================
labels = []
prices = df["close"].values
highs = df["high"].values
lows = df["low"].values

for i in range(len(df) - MAX_HOLD):
    entry_price = prices[i]
    tp_level_buy = entry_price + TP_PIPS
    sl_level_buy = entry_price - SL_PIPS
    tp_level_sell = entry_price - TP_PIPS
    sl_level_sell = entry_price + SL_PIPS

    hit_buy = None
    hit_sell = None

    for j in range(i+1, i+MAX_HOLD):
        if highs[j] >= tp_level_buy:
            hit_buy = "TP"; break
        if lows[j] <= sl_level_buy:
            hit_buy = "SL"; break

    for j in range(i+1, i+MAX_HOLD):
        if lows[j] <= tp_level_sell:
            hit_sell = "TP"; break
        if highs[j] >= sl_level_sell:
            hit_sell = "SL"; break

    # Label: BUY=2, SELL=0, NO_TRADE=1
    if hit_buy == "TP" and hit_sell == "SL":
        labels.append(2)     # Strong BUY
    elif hit_sell == "TP" and hit_buy == "SL":
        labels.append(0)     # Strong SELL
    else:
        labels.append(1)     # NO_TRADE
labels += [1]*MAX_HOLD  # padding

df["label"] = labels

# ===================================
# 3) Drop NO_TRADE (label=1)
# ===================================
df_filtered = df[df["label"] != 1].copy()
print(f"[INFO] Dataset size after dropping NO_TRADE: {len(df_filtered)}")

# ===================================
# 4) Prepare Features
# ===================================
drop_cols = ["time", "label"]
feature_cols = [c for c in df_filtered.columns if c not in drop_cols]

X = df_filtered[feature_cols]
y = df_filtered["label"]  # 0=SELL, 2=BUY

# ===================================
# 5) Balance BUY/SELL (SMOTE)
# ===================================
print("[INFO] Before SMOTE:", y.value_counts())
smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X, y)
print("[INFO] After SMOTE:", pd.Series(y_bal).value_counts())

# ===================================
# 6) Train/Test Split
# ===================================
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

# ===================================
# 7) Train RF Model
# ===================================
print("[INFO] Training RandomForest v4 Balanced...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ===================================
# 8) Evaluation
# ===================================
y_pred = rf_model.predict(X_test)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# Feature importance
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
print("\n=== TOP 10 IMPORTANT FEATURES ===")
print(importances.sort_values(ascending=False).head(10))

# ===================================
# 9) Save Model
# ===================================
joblib.dump({
    "model": rf_model,
    "feature_names": feature_cols
}, MODEL_OUTPUT)

print(f"\n[SAVED] Model → {MODEL_OUTPUT}")
