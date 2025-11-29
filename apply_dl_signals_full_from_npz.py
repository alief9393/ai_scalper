# apply_dl_signals_full_from_npz.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_FILE = "dl_intraday_dataset_seq64.npz"
MODEL_PATH = "lstm_intraday_best.pt"
INPUT_CSV = "XAUUSD_M15_WITH_SIGNALS.csv"
OUTPUT_CSV = "XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

THR_LONG = 0.70
THR_SHORT = 1.0 - THR_LONG  # 0.30


class SeqDataset(Dataset):
    def __init__(self, X):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


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


def load_npz_info():
    data = np.load(DATA_FILE, allow_pickle=True)
    seq_len = int(data["seq_len"][0])

    feature_names = list(data["feature_names"])
    # kalau string-nya tipe numpy, ubah ke str biasa
    feature_names = [fn if isinstance(fn, str) else fn.item() for fn in feature_names]

    feat_mean = data["feat_mean"]  # array, urut sama kayak feature_names
    feat_std = data["feat_std"]

    print("[INFO] Loaded NPZ:", DATA_FILE)
    print("[INFO] seq_len:", seq_len)
    print("[INFO] num_features:", len(feature_names))

    return seq_len, feature_names, feat_mean, feat_std


def build_sequences_from_array(arr, seq_len):
    """
    arr: np.array shape (N, num_features) -> SUDAH di-normalize
    Return: X_seq shape (num_samples, seq_len, num_features)
    Logic sama persis dgn build_sequences() di prepare_dl_dataset.py
    """
    n, num_features = arr.shape
    if n <= seq_len:
        return np.empty((0, seq_len, num_features), dtype=np.float32)

    n_samples = n - seq_len
    X = np.zeros((n_samples, seq_len, num_features), dtype=np.float32)

    for i in range(n_samples):
        X[i] = arr[i : i + seq_len]

    return X


def get_dl_probs(X):
    input_dim = X.shape[2]
    ds = SeqDataset(X)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMClassifier(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print(f"[INFO] Loaded model from {MODEL_PATH} on {DEVICE}")

    softmax = nn.Softmax(dim=1)
    all_probs = []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            prob = softmax(logits)
            prob_up = prob[:, 1]
            all_probs.append(prob_up.cpu().numpy())

    proba_up = np.concatenate(all_probs)
    print("[INFO] DL probs shape:", proba_up.shape)
    return proba_up


def main():
    # ====== 1) Load info NPZ ======
    seq_len, feature_names, feat_mean, feat_std = load_npz_info()

    # ====== 2) Load DF full ======
    df = pd.read_csv(INPUT_CSV, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print("[INFO] Loaded DF:", len(df))

    # Cek semua feature_names ada di DF
    for col in feature_names:
        if col not in df.columns:
            raise ValueError(f"Missing feature column in DF: {col}")

    # Siapkan kolom output
    df["dl_proba_up"] = np.nan
    df["dl_signal"] = 0

    # Bikin Series mean/std supaya align ke nama kolom
    mean_s = pd.Series(feat_mean, index=feature_names)
    std_s = pd.Series(feat_std, index=feature_names)

    # ====== 3) Proses per set (train, val, test) ======
    set_names = ["train", "val", "test"]
    df_sets = {}

    for sname in set_names:
        df_set = df[df["set"] == sname].copy().sort_values("time").reset_index(drop=True)
        if df_set.empty:
            print(f"[WARN] No rows for set='{sname}', skip.")
            continue

        print(f"\n[INFO] Processing set='{sname}' with rows:", len(df_set))

        # ambil fitur & normalize pakai mean/std TRAIN dari NPZ
        X_df = df_set[feature_names].copy()
        X_norm = (X_df - mean_s) / std_s

        X_arr = X_norm.values.astype(np.float32)

        # build sequence pakai logic yg sama
        X_seq = build_sequences_from_array(X_arr, seq_len)
        print(f"[INFO] Built sequences for set='{sname}':", X_seq.shape)

        if X_seq.shape[0] == 0:
            print(f"[WARN] Not enough rows to build sequences for set='{sname}', skip.")
            continue

        # infer model
        proba_up = get_dl_probs(X_seq)

        expected = len(df_set) - seq_len
        assert proba_up.shape[0] == expected, \
            f"[{sname}] Shape mismatch: proba {proba_up.shape[0]} vs expected {expected}"

        # inisialisasi kolom di df_set
        df_set["dl_proba_up"] = np.nan
        df_set["dl_signal"] = 0

        # mapping sama: sample j -> bar index = seq_len - 1 + j
        for j in range(len(proba_up)):
            idx = seq_len - 1 + j
            p = proba_up[j]

            df_set.at[idx, "dl_proba_up"] = p

            if p >= THR_LONG:
                sig = 1
            elif p <= THR_SHORT:
                sig = -1
            else:
                sig = 0

            df_set.at[idx, "dl_signal"] = sig

        df_sets[sname] = df_set

    # ====== 4) Merge balik ke DF full ======
    for sname, df_set in df_sets.items():
        mask = df["set"] == sname
        # df[mask] sudah urut time, sama urutannya dengan df_set
        df.loc[mask, "dl_proba_up"] = df_set["dl_proba_up"].values
        df.loc[mask, "dl_signal"] = df_set["dl_signal"].values

    # ====== 5) Save ======
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Saved -> {OUTPUT_CSV}")
    print("[INFO] dl_signal distribution (non-NaN):")
    print(df["dl_signal"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
