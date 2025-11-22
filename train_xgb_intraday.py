# train_xgb_intraday.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

FILE_INPUT = "XAUUSD_M15_FEATURES_LABELED.csv"

def main():
    # Load data
    df = pd.read_csv(FILE_INPUT, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Pisah fitur & target
    y = df["target"].astype(int)

    drop_cols = [
        "time",
        "date",
        "target",
        "future_ret",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Cek bentuk data
    print("[INFO] X shape:", X.shape)
    print("[INFO] y shape:", y.shape)

    # Time-based split: 70% train, 15% val, 15% test
    n = len(df)
    train_end = int(n * 0.7)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

    print("[INFO] Split:")
    print("  Train:", X_train.shape[0])
    print("  Val  :", X_val.shape[0])
    print("  Test :", X_test.shape[0])

    # Basic XGBoost classifier (nanti bisa di-tune)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",  # ganti "gpu_hist" kalau pakai GPU
        random_state=42,
    )

    # Train di train set, monitor di val set
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Evaluasi di test set (paling penting)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    print("\n[CONFUSION MATRIX] (Test)")
    print(confusion_matrix(y_test, y_pred))

    print("\n[CLASSIFICATION REPORT] (Test)")
    print(classification_report(y_test, y_pred, digits=4))

    # Sedikit lihat feature importance
    importances = model.feature_importances_
    feat_imp = (
        pd.DataFrame({"feature": X.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(25)
    )

    print("\n[TOP 25 FEATURES BY IMPORTANCE]")
    print(feat_imp.to_string(index=False))

    # Simpan model & feature list simpel kalau mau dipakai lagi
    model.save_model("xgb_intraday_model.json")
    feat_imp.to_csv("xgb_intraday_feature_importance.csv", index=False)
    print("\n[OK] Model saved -> xgb_intraday_model.json")
    print("[OK] Feature importance -> xgb_intraday_feature_importance.csv")


if __name__ == "__main__":
    main()
