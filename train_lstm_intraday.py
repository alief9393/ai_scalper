# train_lstm_intraday.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DATA_FILE = "dl_intraday_dataset_seq64.npz"
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, B, H)
        h_last = h_n[-1]  # (B, H)
        logits = self.fc(h_last)
        return logits


def load_data():
    data = np.load(DATA_FILE)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print("[INFO] Loaded data from", DATA_FILE)
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape,   "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape,  "y_test :", y_test.shape)
    print("  Num features:", len(feature_names))

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return avg_loss, acc, all_preds, all_labels


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    train_ds = SeqDataset(X_train, y_train)
    val_ds   = SeqDataset(X_val, y_val)
    test_ds  = SeqDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    input_dim = X_train.shape[2]
    model = LSTMClassifier(input_dim=input_dim).to(DEVICE)
    print(f"[INFO] Model on {DEVICE}")

    # Optional: compute class weights from train set
    cls0 = (y_train == 0).sum()
    cls1 = (y_train == 1).sum()
    w0 = len(y_train) / (2.0 * cls0)
    w1 = len(y_train) / (2.0 * cls1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE)
    print("[INFO] Class weights:", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "lstm_intraday_best.pt")
        print("[OK] Saved best model -> lstm_intraday_best.pt")

    # Evaluate di TEST set
    test_loss, test_acc, y_pred, y_true = eval_epoch(model, test_loader, criterion)
    print("\n===== LSTM TEST PERFORMANCE =====")
    print(f"Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")
    print("\n[CONFUSION MATRIX]")
    print(confusion_matrix(y_true, y_pred))
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()
