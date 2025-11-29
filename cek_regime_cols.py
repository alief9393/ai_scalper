import pandas as pd

INPUT_FILE = "XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["time"])

print("=== Semua kolom yang mengandung 'adx' ===")
print([c for c in df.columns if "adx" in c.lower()])

print("\n=== Sample 5 bar test-set (kolom ADX + ATR) ===")
df_test = df[df["set"] == "test"].copy().sort_values("time").reset_index(drop=True)

# GANTI 'adx_14_m15' ke nama yang muncul di print atas
ADX_COL = "adx_14_m15"
ATR_COL = "atr_14_m15"

cols_to_show = ["time", "set"]
for c in [ADX_COL, ATR_COL]:
    if c in df_test.columns:
        cols_to_show.append(c)

print(df_test[cols_to_show].head(10))

if ADX_COL in df_test.columns:
    print("\n=== Deskripsi ADX di test-set ===")
    print(df_test[ADX_COL].describe())
else:
    print(f"\n[WARN] Kolom ADX '{ADX_COL}' tidak ditemukan di DF.")
