# evaluate_dl_thresholds.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

DATA_FILE = "dl_intraday_dataset_seq64.npz"
MODEL_PATH = "lstm_intraday_best.pt"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Thresholds to evaluate (for prob of class 1 / UP)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


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


def load_data():
    data = np.load(DATA_FILE)
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("[INFO] Loaded data from", DATA_FILE)
    print("  X_test:", X_test.shape, "y_test:", y_test.shape)
    return X_test, y_test


def load_model(input_dim):
    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH} on {DEVICE}")
    return model


def get_test_probs(model, loader):
    all_probs = []
    all_labels = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            prob = softmax(logits)  # (B, 2)
            prob_up = prob[:, 1]    # kelas 1 = UP

            all_probs.append(prob_up.cpu().numpy())
            all_labels.append(y_batch.numpy())

    proba_up = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    return y_true, proba_up


def evaluate_thresholds(y_true, proba, thresholds):
    results = []

    for thr in thresholds:
        thr_long = thr
        thr_short = 1.0 - thr

        long_mask  = proba >= thr_long
        short_mask = proba <= thr_short
        trade_mask = long_mask | short_mask

        n_total  = len(y_true)
        n_trades = trade_mask.sum()
        n_longs  = long_mask.sum()
        n_shorts = short_mask.sum()

        if n_trades == 0:
            precision_all = np.nan
        else:
            correct_trades = (
                ((y_true == 1) & long_mask) |
                ((y_true == 0) & short_mask)
            ).sum()
            precision_all = correct_trades / n_trades

        if n_longs == 0:
            precision_long = np.nan
        else:
            precision_long = (y_true[long_mask] == 1).mean()

        if n_shorts == 0:
            precision_short = np.nan
        else:
            precision_short = (y_true[short_mask] == 0).mean()

        coverage = n_trades / n_total

        results.append({
            "thr_long": thr_long,
            "thr_short": thr_short,
            "n_trades": int(n_trades),
            "n_longs": int(n_longs),
            "n_shorts": int(n_shorts),
            "coverage": coverage,
            "precision_all": precision_all,
            "precision_long": precision_long,
            "precision_short": precision_short,
        })

    return pd.DataFrame(results)


def main():
    X_test, y_test = load_data()
    input_dim = X_test.shape[2]

    test_ds = SeqDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = load_model(input_dim)

    print("[INFO] Getting probabilities on TEST set...")
    y_true, proba_up = get_test_probs(model, test_loader)

    print("\n[INFO] Evaluating thresholds...")
    df_res = evaluate_thresholds(y_true, proba_up, THRESHOLDS)

    df_res["coverage_pct"] = df_res["coverage"] * 100.0
    for col in ["precision_all", "precision_long", "precision_short"]:
        df_res[col] = (df_res[col] * 100.0).round(2)
    df_res["coverage_pct"] = df_res["coverage_pct"].round(2)

    display_cols = [
        "thr_long", "thr_short",
        "n_trades", "n_longs", "n_shorts",
        "coverage_pct",
        "precision_all", "precision_long", "precision_short",
    ]

    print("\n[DL THRESHOLD EVALUATION RESULTS] (Test Set)")
    print(df_res[display_cols].to_string(index=False))

    df_res.to_csv("dl_threshold_evaluation_results.csv", index=False)
    print("\n[OK] Saved -> dl_threshold_evaluation_results.csv")


if __name__ == "__main__":
    main()
