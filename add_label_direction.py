import pandas as pd

FILE_INPUT = "XAUUSD_M15_FEATURES.csv"
FILE_OUTPUT = "XAUUSD_M15_FEATURES_LABELED.csv"

HORIZON = 3  # 3 candle ahead (45 menit)
THRESH_PCT = 0.0003  # ambang profit (0.03%) ~ ±7 pip

def main():
    df = pd.read_csv(FILE_INPUT, parse_dates=['time'])
    df = df.sort_values("time").reset_index(drop=True)

    # Future return (forward shift)
    df["future_ret"] = df["close"].shift(-HORIZON) / df["close"] - 1

    # Klasifikasi label:
    def classify(x):
        if x > THRESH_PCT:
            return 1   # UP
        elif x < -THRESH_PCT:
            return 0   # DOWN
        else:
            return None  # Ignore / uncertain zone

    df["target"] = df["future_ret"].apply(classify)

    # Drop yang None (lebih aman buat training)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    print("[INFO] Final rows with target:", len(df))
    print("[INFO] Sample target distribution:")
    print(df["target"].value_counts(normalize=True))

    df.to_csv(FILE_OUTPUT, index=False)
    print(f"[OK] Saved labeled data -> {FILE_OUTPUT}")


if __name__ == "__main__":
    main()
