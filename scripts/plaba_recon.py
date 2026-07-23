# scripts/plaba_recon.py

"""
quick download of PLABA sample dataset, check row count + schema fit before integrating to Phase B setup.
"""

import json
import os

def flatten_plaba(data, adaptation_priority=None):
    """
    adaptation_priority: ex: if only want to use ONE variant, like ["adaptation2"] per article.
    Otherwise, None - to produce one row per available variant (data augmentation)
    """
    rows, skipped = [], []

    for article_id, pmid_dict in data.items():
        if not isinstance(pmid_dict, dict):
            continue

        for pmid, entry in pmid_dict.items():
            if not isinstance(entry, dict):
                continue

            abstract = entry.get("abstract", {})
            adaptations = entry.get("adaptations", {})
            if not abstract or not adaptations:
                skipped.append(pmid)
                continue

            try:
                input_text = " ".join(abstract[k] for k in sorted(abstract, key=int))
            except (ValueError, TypeError, KeyError):
                skipped.append(pmid)
                continue

            keys = [k for k in (adaptation_priority or adaptations) if k in adaptations]
            for adapt_key in keys:
                sents = adaptations[adapt_key]
                try:
                    target_text = " ".join(sents[k] for k in sorted(sents, key=int))
                except (ValueError, TypeError, KeyError):
                    continue
                rows.append({
                    "input_text": input_text,
                    "target_text": target_text,
                    "task_type": "simplification",
                    "source": "PLABA",
                    "pmid": pmid,
                    "adaptation_variant": adapt_key,
                })
    print(f"Flattened {len(rows)} rows, skipped {len(skipped)} (missing/malformed)")
    return rows

if __name__ == "__main__":
    with open("data/raw/plaba.json") as f:
        data = json.load(f)

    print("Top-level keys count:", len(data.keys()))
    print("Dictionary keys:", list(data.keys()))
    print("Type of data:", type(data))

    rows = flatten_plaba(data) # pass adaptation_priority=["adaptation2"] to restrict, etc.

    with open("data/processed/plaba_flat.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(rows)} items to data/processed/plaba_flat.jsonl")
    print("\nSample rows:")
    for r in rows[:3]:
        print(json.dumps(r, indent=2))

