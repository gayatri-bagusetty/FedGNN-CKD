# merge UCI, Kaggle and synthetic datasets for centralized baseline training
import pandas as pd

uci = pd.read_csv("../data/processed/uci_clean.csv")
synthetic = pd.read_csv("../data/processed/synthetic_clean.csv")

merged = pd.concat([uci, synthetic], axis=0, ignore_index=True)

merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

print("Before removing duplicates:", merged.shape)

merged = merged.drop_duplicates()

print("After removing duplicates:", merged.shape)

merged.to_csv("centralized_ckd_dataset.csv", index=False)