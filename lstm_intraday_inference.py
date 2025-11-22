# lstm_intraday_inference.py
#
# Inference untuk DL Intraday v1 (LSTM) XAUUSD M15
# - Load model & meta (urutan fitur, seq_len) dari dl_intraday_dataset_seq64.npz
# - Baca CSV berisi data M15 ber-feature lengkap (39 fitur)
# - Ambil 64 bar terakhir -> (1, 64, 39)
# - Keluarkan:
#     - proba_up
#     - dl_signal (1/0/-1) berdasarkan threshold
#     - entry_price (default: last close, bisa override)
#     - SL & TP price (ATR-based: SL=1.2*ATR, TP=2*SL)

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ================= CONFIG ===================

DATA_META_FILE = "dl_intraday_dataset_seq64.npz"
MODEL_PATH     = "lstm_intraday_best.pt"

# Threshold final DL Intraday v1
THR_LONG  = 0.70
THR_SHORT = 0.30

# ATR-based SL/TP (konfigurasi hasil grid terbaik)
ATR_COL      = "atr_14_m15"
ATR_MULT_SL  = 1.2    # SL = 1.2 * ATR
RR_TP        = 2.0    # TP = 2.0 * SL = 2.4 * ATR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============== MODEL DEF ==================


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
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]        # (B, H)
        logits = self.fc(h_last)  # (B, 2)
        return logits


# ============= LOAD META & MODEL ============


def load_meta_and_model():
    """
    Load meta (seq_len, feature_names) dari NPZ dan model dari .pt
    """
    data = np.load(DATA_META_FILE, allow_pickle=True)
    feature_names = list(data["feature_names"])
    seq_len = int(data["seq_len"][0])

    input_dim = len(feature_names)
    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Loaded meta from", DATA_META_FILE)
    print("  seq_len      :", seq_len)
    print("  n_features   :", input_dim)
    print("[INFO] Loaded model from", MODEL_PATH, "on", DEVICE)

    return feature_names, seq_len, model


# ============= INFERENCE CORE ===============


def build_sequence_from_df(df, feature_names, seq_len):
    """
    df: DataFrame berisi history M15 dengan kolom fitur lengkap.
    Ambil seq_len bar terakhir -> (1, seq_len, n_features)
    """
    if len(df) < seq_len:
        raise ValueError(f"Data kurang: butuh minimal {seq_len} bar, sekarang cuma {len(df)}")

    df_seq = df.sort_values("time").iloc[-seq_len:]
    # pastikan semua feature ada
    missing = [f for f in feature_names if f not in df_seq.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df_seq[feature_names].to_numpy(dtype=np.float32)  # (seq_len, n_features)
    X = np.expand_dims(X, axis=0)  # (1, seq_len, n_features)
    return X, df_seq.iloc[-1]  # last_row


def infer_signal_and_levels(model, X, last_row, entry_price=None):
    """
    X: np.array shape (1, seq_len, n_features)
    last_row: row terakhir (pandas Series) untuk baca ATR & info price
    entry_price: kalau None, pakai last_row["close"]
    """
    X_t = torch.from_numpy(X).float().to(DEVICE)

    with torch.no_grad():
        logits = model(X_t)            # (1, 2)
        prob = torch.softmax(logits, dim=1)
        proba_up = float(prob[0, 1].cpu().item())

    # Tentukan signal
    if proba_up >= THR_LONG:
        signal = 1
    elif proba_up <= THR_SHORT:
        signal = -1
    else:
        signal = 0

    # Entry price (default: close bar terakhir)
    if entry_price is None:
        entry_price = float(last_row["close"])
    else:
        entry_price = float(entry_price)

    # ATR untuk SL/TP
    if ATR_COL not in last_row.index:
        raise ValueError(f"{ATR_COL} tidak ditemukan di data terakhir")

    atr_val = float(last_row[ATR_COL])
    sl_dist = ATR_MULT_SL * atr_val
    tp_dist = RR_TP * sl_dist

    if signal == 1:  # LONG
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
        direction_str = "LONG"
    elif signal == -1:  # SHORT
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
        direction_str = "SHORT"
    else:
        sl_price = None
        tp_price = None
        direction_str = "FLAT"

    result = {
        "proba_up": proba_up,
        "signal": signal,
        "direction": direction_str,
        "entry_price": entry_price,
        "atr_m15": atr_val,
        "sl_dist": sl_dist,
        "tp_dist": tp_dist,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "decision_time": last_row.get("time", None),
    }
    return result


# ================== CLI =====================


def main():
    parser = argparse.ArgumentParser(
        description="LSTM Intraday Inference (XAUUSD M15) - DL Intraday v1"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path ke CSV berisi history M15 ber-feature lengkap (minimal seq_len bar, ada kolom 'time').",
    )
    parser.add_argument(
        "--entry-price",
        type=float,
        default=None,
        help="Harga entry yang ingin digunakan (misal current Bid/Ask). Kalau None, pakai close bar terakhir.",
    )

    args = parser.parse_args()

    # Load meta & model
    feature_names, seq_len, model = load_meta_and_model()

    # Load CSV
    df = pd.read_csv(args.csv)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)
    print("[INFO] Loaded DF from", args.csv, "| rows:", len(df))

    # Build sequence dari 64 bar terakhir
    X, last_row = build_sequence_from_df(df, feature_names, seq_len)

    # Inference
    result = infer_signal_and_levels(
        model,
        X,
        last_row,
        entry_price=args.entry_price,
    )

    # Print hasil
    print("\n===== LSTM INTRADAY INFERENCE RESULT =====")
    print("Decision time :", result["decision_time"])
    print(f"proba_up      : {result['proba_up']:.4f}")
    print(f"signal        : {result['signal']} ({result['direction']})")
    print(f"entry_price   : {result['entry_price']:.2f}")
    print(f"ATR(M15)      : {result['atr_m15']:.2f}")
    print(f"SL distance   : {result['sl_dist']:.2f}")
    print(f"TP distance   : {result['tp_dist']:.2f}")
    if result["direction"] != "FLAT":
        print(f"SL price      : {result['sl_price']:.2f}")
        print(f"TP price      : {result['tp_price']:.2f}")
    else:
        print("No trade (FLAT) -> SL/TP tidak dihitung")
    print("==========================================\n")


if __name__ == "__main__":
    main()
