# scripts/noteaid_parse.py

import pandas as pd
import json

exp_good = pd.read_csv("data/raw/readme_exp_good.csv")
syn_good = pd.read_csv("data/raw/readme_syn_good.csv")

rows= []
for _, r in exp_good.iterrows():
    rows.append({
        "input_text": r["ann_text"],
        "target_text": r["split_print"],
        "task_type": "jargon_definition",
        "source": "NoteAid_exp_good",
    })

for _, r in syn_good.iterrows():
    rows.append({
        "input_text": r["ann_text"],
        "target_text": r["gen_def"],
        "task_type": "jargon_definition",
        "source": "NoteAid_syn_good",
    })

with open("data/processed/noteaid_flat.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"Wrote {len(rows)} rows")


