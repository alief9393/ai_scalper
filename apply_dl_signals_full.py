# apply_dl_signals_full.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_FILE   = "dl_intraday_dataset_seq64.npz"   # meta: feature_names, seq_len
MODEL_PATH  = "lstm_intraday_best.pt"
FEATURE_FILE = "XAUUSD_M15_FEATURES.csv"        # dari build_features_intraday.py
OUTPUT_FILE  = "XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv"

THR_LONG  = 0.70
THR_SHORT = 0.30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        logits = self.fc(h_last)
        return logits


def load_meta_and_model():
    data = np.load(DATA_FILE, allow_pickle=True)
    feature_names = list(data["feature_names"])
    seq_len = int(data["seq_len"][0])
    input_dim = len(feature_names)

    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Loaded meta/model:")
    print("  seq_len    :", seq_len)
    print("  n_features :", input_dim)
    print("  device     :", DEVICE)

    return feature_names, seq_len, model


def main():
    feature_names, seq_len, model = load_meta_and_model()

    df = pd.read_csv(FEATURE_FILE, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    print(f"[INFO] Loaded features: {n} rows")

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in FEATURES CSV: {missing}")

    proba_up_arr = np.full(n, np.nan, dtype=np.float32)
    signal_arr   = np.zeros(n, dtype=np.int8)

    for i in range(seq_len - 1, n):
        window = df.iloc[i - seq_len + 1 : i + 1]
        X = window[feature_names].to_numpy(dtype=np.float32)
        X = np.expand_dims(X, axis=0)  # (1, seq_len, n_features)

        X_t = torch.from_numpy(X).to(DEVICE)
        with torch.no_grad():
            logits = model(X_t)
            prob = torch.softmax(logits, dim=1)
            proba_up = float(prob[0, 1].cpu().item())

        if proba_up >= THR_LONG:
            sig = 1
        elif proba_up <= THR_SHORT:
            sig = -1
            # kalau mau FLAT-only, bisa di-set 0 di sini
        else:
            sig = 0

        proba_up_arr[i] = proba_up
        signal_arr[i]   = sig

    df["dl_proba_up"] = proba_up_arr
    df["dl_signal"]   = signal_arr

    # tandai semua sebagai test supaya backtest ga ngefilter aneh-aneh
    df["set"] = "test"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Saved DL signals -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
