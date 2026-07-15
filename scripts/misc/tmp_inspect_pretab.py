import json

targets = ["24-1616", "IN-A6RO"]

with open("data/processed/silver_labels_dryrun.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        rec_str = json.dumps(rec)
        for t in targets:
            if t in rec_str:
                text = rec.get("silver_target", rec.get("output", ""))
                print("\n========== ORIGINAL (PRE-VALIDATION) RECORD MATCHING '" + t + "' ==========\n")
                print(text)
                print("\n========== END RECORD ==========\n")
