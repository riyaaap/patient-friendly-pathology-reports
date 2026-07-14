import json
with open("data/processed/silver_labels_dryrun_validated.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if "24-1616" in json.dumps(rec):
            text = rec.get("silver_target", rec.get("output", ""))
            start = text.find("### 1. Your Diagnosis")
            end = text.find("### 2. What This Means")
            print(text[start:end])
