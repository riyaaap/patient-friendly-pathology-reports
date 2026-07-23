# scripts/noteaid_recon.py
import pandas as pd

exp_good = pd.read_csv("data/raw/readme_exp_good.csv")
syn_good = pd.read_csv("data/raw/readme_syn_good.csv")

print("exp_good columns:", exp_good.columns.tolist())
print("exp_good shape:", exp_good.shape)
print(exp_good.head(3))

print("\nsyn_good columns:", syn_good.columns.tolist())
print("syn_good shape:", syn_good.shape)
print(syn_good.head(3))


