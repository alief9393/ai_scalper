import pandas as pd
from feature_engineering_advanced import add_advanced_features

df = pd.read_csv("xau_M5_ml_dataset.csv")  # dataset lama
df["time"] = pd.to_datetime(df["time"])

df_adv = add_advanced_features(df)

df_adv.to_csv("xau_M5_ml_dataset_ADV.csv", index=False)
print("[SAVED] xau_M5_ml_dataset_ADV.csv created with advanced features!")
