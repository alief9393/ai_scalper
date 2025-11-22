import pandas as pd

df = pd.read_csv("XAUUSD_M15_WITH_DL_SIGNALS.csv", parse_dates=["time"])

# Range yang tadi kamu backtest tapi 0 trade
mask = (df["time"] >= "2025-08-01") & (df["time"] <= "2025-09-15")
sub = df[mask]

print("Total rows di range ini:", len(sub))

print("\nSET value counts:")
print(sub["set"].value_counts(dropna=False))

print("\ndl_signal value counts:")
print(sub["dl_signal"].value_counts(dropna=False))

print("\nPersentase NaN dl_proba_up:", sub["dl_proba_up"].isna().mean() * 100, "%")
print("Persentase NaN ATR:", sub["atr_14_m15"].isna().mean() * 100, "%")

print("\nSample 10 baris:")
print(sub[["time","dl_signal","dl_proba_up","atr_14_m15","set"]].head(10))
