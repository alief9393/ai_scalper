import pandas as pd

df = pd.read_csv("XAUUSD_M15_WITH_DL_SIGNALS_FULL.csv", parse_dates=["time"])
print(df.groupby("set")["time"].agg(["min","max","count"]))
