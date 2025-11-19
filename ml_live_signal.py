"""
ml_live_signal.py
Real-time inference engine for ML scalping model
- Loads model & scaler
- Accepts latest candle(s) → output BUY / SELL / NO_TRADE
"""

import joblib
import pandas as pd
import numpy as np

MODEL_FILE = "rf_scalping_model_timesplit.pkl"


class MLSignalEngine:
    def __init__(self):
        print("[INFO] Loading ML model for live inference...")
        data = joblib.load(MODEL_FILE)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.conf_threshold = data["conf_threshold"]
        print("[INFO] Model loaded successfully.")

    def predict_signal(self, df_latest):
        """
        Input:
            df_latest => DataFrame with latest candle + full features
                         must include same feature columns used in training
        Output:
            Dict => {"signal": "BUY"/"SELL"/"NO_TRADE", "confidence": 0.82}
        """

        # Pastikan fitur lengkap dan urut
        X_live = df_latest[self.feature_names].tail(1)
        X_scaled = self.scaler.transform(X_live)

        # Predict probability & class
        proba = self.model.predict_proba(X_scaled)[0]
        pred_class = self.model.predict(X_scaled)[0]

        # Map confidence berdasarkan prediksi
        class_to_idx = {c: i for i, c in enumerate(self.model.classes_)}
        conf = proba[class_to_idx[pred_class]]

        # Apply confidence filter seperti training
        if pred_class == 1 and conf >= self.conf_threshold:
            return {"signal": "BUY", "confidence": float(conf)}

        elif pred_class == -1 and conf >= self.conf_threshold:
            return {"signal": "SELL", "confidence": float(conf)}

        else:
            return {"signal": "NO_TRADE", "confidence": float(conf)}


# ===== Quick Test (Manual) =====
if __name__ == "__main__":
    print("[TEST] Running demo...")

    engine = MLSignalEngine()

    # Contoh input: load 1 row terakhir dari dataset fitur kita
    test_df = pd.read_csv("xau_M5_ml_dataset.csv")

    result = engine.predict_signal(test_df)
    print("\n[RESULT SIGNAL]:", result)
