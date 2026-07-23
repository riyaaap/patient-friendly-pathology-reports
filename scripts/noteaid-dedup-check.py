import json
seen = set()
dupes = 0
with open("data/processed/noteaid_flat.jsonl") as f:
    for line in f:
        r = json.loads(line)
        key = r["input_text"].strip().lower()
        if key in seen:
            dupes += 1
        seen.add(key)
print(f"Duplicate input_text terms: {dupes} / {sum(1 for _ in open('data/processed/noteaid_flat.jsonl'))}")
