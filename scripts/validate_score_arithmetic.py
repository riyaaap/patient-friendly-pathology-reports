"""
Tier 3 generic validator: flags internal arithmetic inconsistencies in
generated silver labels, independent of any specific named scoring system.
E.g. catches "Gleason 7 (4 + 4)" where 4+4=8, not 7.

Usage:
    python scripts/validate_score_arithmetic.py \
        --input_jsonl data/processed/silver_labels_dryrun_validated.jsonl \
        --report_csv data/processed/arithmetic_validation_report.csv
"""

import argparse
import json
import re
import pandas as pd

# Matches patterns like: "9 (4 + 5)", "score of 7 (3+4)", "Gleason 8 (4 + 4)"
SUM_PATTERN = re.compile(r"(\d+)\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)")


def check_text(text: str):
    issues = []
    for match in SUM_PATTERN.finditer(text):
        stated_total, a, b = match.groups()
        stated_total, a, b = int(stated_total), int(a), int(b)
        actual_sum = a + b
        if actual_sum != stated_total:
            issues.append({
                "matched_text": match.group(),
                "stated_total": stated_total,
                "components": f"{a} + {b}",
                "actual_sum": actual_sum,
                "flag": "ARITHMETIC_MISMATCH",
            })
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--report_csv", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.input_jsonl) as f:
        for line in f:
            record = json.loads(line)
            patient_id = record.get("patient_filename", "UNKNOWN")
            issues = check_text(record.get("silver_target", ""))
            for issue in issues:
                rows.append({"patient_filename": patient_id, **issue})

    report_df = pd.DataFrame(rows)
    report_df.to_csv(args.report_csv, index=False)

    if len(report_df) == 0:
        print("No arithmetic mismatches found.")
    else:
        print(f"Found {len(report_df)} arithmetic mismatch(es). See {args.report_csv}")
        print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
