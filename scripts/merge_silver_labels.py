import json
import pandas as pd

authoritative_df = pd.read_csv("data/processed/tcga_reports_cleaned_filtered.csv")
authoritative_filenames = set(authoritative_df["patient_filename"])

EXCLUDED = {
    "TCGA-E2-A109.089E0FB9-6B45-4E3E-843E-EC98D39863F6",
    "TCGA-CN-6022.f7b8c864-6fcd-4b64-bcee-74bc3dcf6c39",
    "TCGA-CN-A49B.3B5F227F-B65D-483A-8BCE-A8C079326510",
    "TCGA-CN-A642.79C16F29-FA94-44CD-A971-558B23508CE3",
    "TCGA-W8-A86G.A5433CA5-1DCC-4711-9B3B-6478AE7B5FAA",
    "TCGA-KM-A7QK.192A2806-61A7-49C9-91D6-B345DDFB7331",
    "TCGA-E2-A158.1CEBF20D-767A-4A19-AA41-B9EAFCF05176",
}

INPUT_FILES = [
    "data/processed/silver_labels_full.jsonl",
    "data/processed/silver_labels_retry2.jsonl",
    "data/processed/silver_labels_timeout_retry.jsonl",
]

OUTPUT_FILE = "data/processed/silver_labels_merged.jsonl"

seen = {}
skipped_excluded = 0
skipped_invalid = 0
dupes_overwritten = 0
skipped_out_of_scope = 0

for path in INPUT_FILES:
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            fname = row["patient_filename"]
            if fname in EXCLUDED:
                skipped_excluded += 1
                continue
            if fname not in authoritative_filenames:
                skipped_out_of_scope += 1
                continue
            if row["error"] is not None:
                skipped_invalid += 1
                continue
            if fname in seen:
                dupes_overwritten += 1
            seen[fname] = row

with open(OUTPUT_FILE, "w") as f:
    for row in seen.values():
        f.write(json.dumps(row) + chr(10))

print("Merged valid rows:", len(seen))
print("Skipped (excluded long-reports):", skipped_excluded)
print("Skipped (invalid/error rows):", skipped_invalid)
print("Skipped (out-of-scope, not in authoritative CSV):", skipped_out_of_scope)
print("Duplicate filenames overwritten:", dupes_overwritten)
print("Output written to:", OUTPUT_FILE)
