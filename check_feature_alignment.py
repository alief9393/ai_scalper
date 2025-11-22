import numpy as np
import pandas as pd

from features_intraday import add_m15_features, add_h1_context, add_daily_context

DATA_META_FILE = "dl_intraday_dataset_seq64.npz"

FILE_M15 = "XAUUSD_M15_ICM.csv"
FILE_H1  = "XAUUSD_H1_ICM.csv"
FILE_D1  = "XAUUSD_D1_ICM.csv"


def load_feature_names():
    data = np.load(DATA_META_FILE, allow_pickle=True)
    feature_names = list(data["feature_names"])
    seq_len = int(data["seq_len"][0])
    print("[INFO] Loaded meta from", DATA_META_FILE)
    print("  seq_len    :", seq_len)
    print("  n_features :", len(feature_names))
    return feature_names


def build_features_offline():
    df_m15 = pd.read_csv(FILE_M15, parse_dates=["time"])
    df_h1  = pd.read_csv(FILE_H1,  parse_dates=["time"])
    df_d1  = pd.read_csv(FILE_D1,  parse_dates=["time"])

    df_feat = add_m15_features(df_m15)
    df_feat = add_h1_context(df_feat, df_h1)
    df_feat = add_daily_context(df_feat, df_d1)

    df_feat = df_feat.sort_values("time").reset_index(drop=True)
    df_feat = df_feat.dropna().reset_index(drop=True)

    print("[INFO] Built feature DF offline")
    print("  rows :", len(df_feat))
    print("  from :", df_feat['time'].min(), "->", df_feat['time'].max())
    return df_feat


def main():
    feature_names = load_feature_names()
    df_feat = build_features_offline()

    cols = set(df_feat.columns)

    # fitur yang diminta model tapi tidak ada di DF
    missing = [f for f in feature_names if f not in cols]

    # kolom ekstra yang ada di DF tapi bukan fitur model (boleh aja, asalkan nggak wajib)
    extra = [c for c in df_feat.columns if c not in feature_names + ["time", "date", "target", "future_ret", "set", "dl_signal", "dl_proba_up"]]

    print("\n===== FEATURE ALIGNMENT CHECK =====")
    print("Total feature_names (model):", len(feature_names))
    print("Total DF columns           :", len(df_feat.columns))

    if missing:
        print("\n[WARN] Missing columns (dibutuhkan model, tapi nggak ada di DF):")
        for c in missing:
            print("  -", c)
    else:
        print("\n[OK] Tidak ada feature yang hilang (semua feature_names ada di DF).")

    if extra:
        print("\n[INFO] Extra columns (ada di DF, tapi bukan feature model):")
        for c in extra:
            print("  -", c)
    else:
        print("\n[INFO] Tidak ada kolom ekstra di DF (selain time/date/label).")

    print("==================================\n")


if __name__ == "__main__":
    main()
