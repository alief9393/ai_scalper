import pandas as pd

old = pd.read_csv("XAUUSD_M15_WITH_DL_SIGNALS.csv", parse_dates=["time"])
new = pd.read_csv("XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv", parse_dates=["time"])

start = "2025-11-01"
end   = "2025-11-27"

old_r = old[(old["time"] >= start) & (old["time"] <= end)].copy()
new_r = new[(new["time"] >= start) & (new["time"] <= end)].copy()

merged = old_r.merge(new_r, on="time", suffixes=("_old", "_new"))

print("Rows overlapped:", len(merged))
print("Mean abs diff proba_up:",
      (merged["dl_proba_up_old"] - merged["dl_proba_up_new"]).abs().mean())
print("Signal match ratio:",
      (merged["dl_signal_old"] == merged["dl_signal_new"]).mean())
