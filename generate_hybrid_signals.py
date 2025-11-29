import pandas as pd

# File input - sesuaikan dengan nama file lo
DL_FILE = "XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv"
XGB_FILE = "XAUUSD_M15_WITH_XGB_SIGNALS_FULL.csv"

OUTPUT_FILE = "XAUUSD_M15_WITH_HYBRID_SIGNALS_FULL.csv"

print("[INFO] Loading files...")
df_dl = pd.read_csv(DL_FILE, parse_dates=["time"])
df_xgb = pd.read_csv(XGB_FILE, parse_dates=["time"])

print("[INFO] Merging on 'time' column...")
df = pd.merge(
    df_dl,
    df_xgb[['time', 'xgb_proba_up', 'xgb_signal']],   # ambil kolom penting
    on='time',
    how='inner'  # pakai inner biar waktu cocok
)

# Optional: cek missing hybrid columns
if 'dl_proba_up' not in df.columns or 'xgb_proba_up' not in df.columns:
    raise ValueError("Kolom 'dl_proba_up' atau 'xgb_proba_up' tidak ditemukan!")

print("[INFO] Hybrid sample:")
print(df[['time', 'dl_proba_up', 'xgb_proba_up', 'dl_signal', 'xgb_signal']].head())

df.to_csv(OUTPUT_FILE, index=False)
print(f"[OK] Saved hybrid dataset -> {OUTPUT_FILE}")
