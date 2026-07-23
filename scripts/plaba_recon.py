# scripts/plaba_recon.py

"""
quick download of PLABA sample dataset, check row count + schema fit before integrating to Phase B setup.
"""

import json

from datasets import load_dataset
ds = load_dataset("plaba")

print(f"Total rows: {len(ds['train'])}")
print("Sample 5 rows:")
for i, row in enumerate(ds["train"].select(range(5))):
    print(f"--- row {i} ---")
    print(row) 

# check schema fit vs. target: input text, target text, task type
print("\nAvailable fields:", ds["train"].column_names)


