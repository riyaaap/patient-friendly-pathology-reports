# quick lookup before running the full script
import json

lengths = []
with open("data/processed/silver_labels_final.jsonl") as f:
    for line in f:
        r = json.loads(line)
        lengths.append((len(r["raw_report"]), r["patient_filename"]))

lengths.sort()
print("Shortest 50 reports:")
for length, fname in lengths[:50]:
    print(length, fname)

