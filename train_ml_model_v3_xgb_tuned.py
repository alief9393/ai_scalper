"""
train_ml_model_v3_xgb_tuned.py

XGBoost tuning with advanced features + SMOTE balancing.

Goal:
- Improve recall for BUY (class 2) and SELL (class 0)
- Keep NO_TRADE class reasonable
"""

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# ===== 1. Load dataset =====
df = pd.read_csv("xau_M5_ml_dataset_ADV.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# Map labels [-1, 0, 1] -> [0, 1, 2]
label_mapping = {-1: 0, 0: 1, 1: 2}
df["label"] = df["label"].map(label_mapping)

X = df.drop(columns=["time", "label"])
y = df["label"]

print(f"[INFO] Dataset shape: {X.shape}, Features: {len(X.columns)}")

# ===== 2. Time-based split train/test (70/30) =====
split_idx = int(len(df) * 0.7)
X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"[INFO] Train_full: {X_train_full.shape}, Test: {X_test.shape}")

# ===== 3. Scale =====
scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ===== 4. SMOTE balancing on train_full =====
print("\n[INFO] BEFORE SMOTE (train_full) class distribution:")
print(y_train_full.value_counts())

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_full_scaled, y_train_full)

print("\n[INFO] AFTER SMOTE (train_bal) class distribution:")
print(pd.Series(y_train_bal).value_counts())

# ===== 5. Inner split for tuning (train_inner / val_inner) =====
# Pakai 80% train_bal untuk train_inner, 20% untuk val_inner
n_train = int(len(X_train_bal) * 0.8)
X_train_inner, X_val_inner = X_train_bal[:n_train], X_train_bal[n_train:]
y_train_inner, y_val_inner = y_train_bal[:n_train], y_train_bal[n_train:]

print(f"\n[INFO] Tuning split: train_inner={X_train_inner.shape}, val_inner={X_val_inner.shape}")

# ===== 6. Hyperparameter search space (manual grid) =====
param_grid = [
    {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "n_estimators": 500, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.8},
    {"max_depth": 8, "n_estimators": 400, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.9},
    {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.02, "subsample": 0.9, "colsample_bytree": 0.9},
    {"max_depth": 10, "n_estimators": 500, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8},
]

def evaluate_recall_0_2(y_true, y_pred):
    """Score = average recall of class 0 (SELL) and 2 (BUY)."""
    from sklearn.metrics import classification_report
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    r0 = rep.get("0", {}).get("recall", 0.0)
    r2 = rep.get("2", {}).get("recall", 0.0)
    return (r0 + r2) / 2.0, rep

best_score = -1.0
best_params = None
best_report = None

print("\n[SEARCH] Tuning XGBoost hyperparameters...\n")

for i, params in enumerate(param_grid, start=1):
    print(f"[TRY] Set {i}/{len(param_grid)}: {params}")
    xgb = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train_inner, y_train_inner)
    y_val_pred = xgb.predict(X_val_inner)

    score, rep = evaluate_recall_0_2(y_val_inner, y_val_pred)
    print(f"  -> Avg recall (0 & 2) on val: {score:.4f}")
    print(f"     Class 0 recall: {rep['0']['recall']:.4f}, Class 2 recall: {rep['2']['recall']:.4f}")
    print(f"     Overall accuracy val: {rep['accuracy']:.4f}\n")

    if score > best_score:
        best_score = score
        best_params = params
        best_report = rep

print("\n[BEST] Hyperparameters chosen:")
print(best_params)
print(f"[BEST] Avg recall (class 0 & 2) on val: {best_score:.4f}")
print(f"[BEST] Class 0 recall: {best_report['0']['recall']:.4f}, Class 2 recall: {best_report['2']['recall']:.4f}")
print(f"[BEST] Val accuracy: {best_report['accuracy']:.4f}")

# ===== 7. Train final model on full balanced train (X_train_bal) with best params =====
print("\n[FINAL TRAIN] Training final XGBoost with best params on full balanced train...")
xgb_final = XGBClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    learning_rate=best_params["learning_rate"],
    subsample=best_params["subsample"],
    colsample_bytree=best_params["colsample_bytree"],
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

xgb_final.fit(X_train_bal, y_train_bal)

# Evaluate on real test (X_test_scaled, y_test)
y_test_pred = xgb_final.predict(X_test_scaled)

print("\n[TEST] XGBoost Tuned - Classification Report:")
print(classification_report(y_test, y_test_pred, digits=4))

print("[TEST] Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ===== 8. Save tuned model =====
joblib.dump({
    "model": xgb_final,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "conf_threshold": 0.5
}, "xgb_scalping_model_v3_tuned.pkl")

print("\n[SAVED] xgb_scalping_model_v3_tuned.pkl")
