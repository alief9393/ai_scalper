# train_xgb_intraday_xgb.py

import numpy as np
import xgboost as xgb

DATA_FILE = "xgb_intraday_dataset.npz"
MODEL_OUT = "xgb_intraday.model"
META_OUT  = "xgb_intraday_meta.npz"

def main():
    data = np.load(DATA_FILE, allow_pickle=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    feature_names = list(data["feature_names"])

    print("[INFO] Loaded dataset:")
    print("  X_train:", X_train.shape)
    print("  X_val  :", X_val.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",  # kalau CPU
    }

    evals = [(dtrain, "train"), (dval, "val")]

    print("[INFO] Training XGBoost...")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50,
    )

    print("[OK] Best iteration:", bst.best_iteration)
    print("[OK] Best score    :", bst.best_score)

    # Save model
    bst.save_model(MODEL_OUT)
    print("[OK] Saved model ->", MODEL_OUT)

    # Save meta (feature_names)
    np.savez_compressed(
        META_OUT,
        feature_names=np.array(feature_names),
        best_iteration=np.array([bst.best_iteration]),
    )
    print("[OK] Saved meta ->", META_OUT)


if __name__ == "__main__":
    main()
