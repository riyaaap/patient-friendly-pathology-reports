import json
from collections import defaultdict

term_defs = defaultdict(set)
with open("data/processed/noteaid_flat.jsonl") as f:
    for line in f:
        r = json.loads(line)
        key = r["input_text"].strip().lower()
        term_defs[key].add(r["target_text"].strip())

unique_terms = len(term_defs)
avg_defs_per_term = sum(len(v) for v in term_defs.values()) / unique_terms
print(f"Unique terms: {unique_terms}")
print(f"Avg distinct definitions per term: {avg_defs_per_term:.2f}")
