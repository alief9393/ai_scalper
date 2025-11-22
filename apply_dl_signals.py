# apply_dl_signals.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_FILE = "dl_intraday_dataset_seq64.npz"
MODEL_PATH = "lstm_intraday_best.pt"
INPUT_CSV = "XAUUSD_M15_WITH_SIGNALS.csv"
OUTPUT_CSV = "XAUUSD_M15_WITH_DL_SIGNALS.csv"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THR_LONG = 0.70
THR_SHORT = 1.0 - THR_LONG  # 0.30


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def load_dl_test():
    data = np.load(DATA_FILE)
    X_test = data["X_test"]
    y_test = data["y_test"]
    seq_len = int(data["seq_len"][0])
    print("[INFO] Loaded DL data:", X_test.shape, y_test.shape, "SEQ_LEN:", seq_len)
    return X_test, y_test, seq_len


def get_dl_probs(X_test):
    input_dim = X_test.shape[2]
    test_ds = SeqDataset(X_test, np.zeros(X_test.shape[0]))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH} on {DEVICE}")

    softmax = nn.Softmax(dim=1)

    all_probs = []

    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            prob = softmax(logits)
            prob_up = prob[:, 1]
            all_probs.append(prob_up.cpu().numpy())

    proba_up = np.concatenate(all_probs)
    print("[INFO] DL probs shape:", proba_up.shape)
    return proba_up


def main():
    # 1) Load DF asli (dengan OHLC, set, dsb)
    df = pd.read_csv(INPUT_CSV, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Ambil hanya TEST set
    df_test = df[df["set"] == "test"].copy().sort_values("time").reset_index(drop=True)
    print("[INFO] Rows in TEST set (DF):", len(df_test))

    # 2) Load DL test sequences & proba
    X_test, y_test, seq_len = load_dl_test()
    proba_up = get_dl_probs(X_test)

    # X_test length = len(df_test) - seq_len + 1
    expected = len(df_test) - seq_len
    assert proba_up.shape[0] == expected, f"Shape mismatch: proba {proba_up.shape[0]} vs expected {expected}"

    # Inisialisasi kolom DL di df_test
    df_test["dl_proba_up"] = np.nan
    df_test["dl_signal"] = 0

    # Map: sample j -> bar index = seq_len-1 + j
    for j in range(len(proba_up)):
        idx = seq_len - 1 + j
        p = proba_up[j]
        df_test.at[idx, "dl_proba_up"] = p

        if p >= THR_LONG:
            sig = 1
        elif p <= THR_SHORT:
            sig = -1
        else:
            sig = 0

        df_test.at[idx, "dl_signal"] = sig

    # Gabung kembali ke df full
    df_merged = df.copy()
    df_merged["dl_proba_up"] = np.nan
    df_merged["dl_signal"] = 0

    mask_test = df_merged["set"] == "test"
    df_merged.loc[mask_test, "dl_proba_up"] = df_test["dl_proba_up"].values
    df_merged.loc[mask_test, "dl_signal"] = df_test["dl_signal"].values

    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved -> {OUTPUT_CSV}")
    print("[INFO] dl_signal distribution (test set):")
    print(df_test["dl_signal"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
