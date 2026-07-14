"""
Scan cleaned/filtered pathology reports for mentions of named clinical
scoring/grading systems, to prioritize which systems get hardcoded
reference tables (Tier 1 registry) in the teacher llm output validation system.

Usage:
    python scripts/scan_scoring_systems.py \
        --input_csv data/processed/tcga_reports_cleaned_filtered.csv \
        --text_col cleaned_text \
        --output_csv data/processed/scoring_system_frequency.csv \
        --context_chars 80
"""

import argparse
import re
import pandas as pd

# Keyword -> canonical system name. Add more here as you discover new ones
# via the "other_hits" fallback pass below.
SCORING_SYSTEMS = {
    r"\bgleason\b": "Gleason (prostate)",
    r"\bnottingham\b": "Nottingham (breast)",
    r"\bbloom[\s-]?richardson\b": "Bloom-Richardson (breast, legacy)",
    r"\bfuhrman\b": "Fuhrman (renal)",
    r"\bwho[\s/-]?isup\b": "WHO/ISUP (renal)",
    r"\bfigo\b": "FIGO (gynecologic)",
    r"\bann\s?arbor\b": "Ann Arbor (lymphoma)",
    r"\btnm\b": "TNM staging (general)",
    r"\bwho\s?cns\b|\bwho\s?grade\b": "WHO CNS Grade (brain/CNS)",
    r"\bbreslow\b": "Breslow thickness (melanoma)",
    r"\bclark\'?s?\s?level\b": "Clark Level (melanoma, legacy)",
    r"\bki[\s-]?67\b": "Ki-67 index (proliferation marker, not a staging system but flagged)",
    r"\bedmondson[\s-]?steiner\b": "Edmondson-Steiner (hepatocellular)",
    r"\bisup\s?grade\b": "ISUP Grade Group (prostate)",
    r"\bmasaoka\b": "Masaoka (thymoma)",
    r"\bfrench[\s-]?american[\s-]?british\b|\bfab\b": "FAB Classification (hematologic)",
    r"\bamerican joint committee\b": "AJCC (general staging body)",
}


def scan(df: pd.DataFrame, text_col: str, context_chars: int):
    rows = []
    for name, pattern in SCORING_SYSTEMS.items():
        pass  # placeholder to keep structure readable; real loop below

    for keyword_pattern, system_name in SCORING_SYSTEMS.items():
        regex = re.compile(keyword_pattern, flags=re.IGNORECASE)
        count = 0
        example_row_id = None
        example_excerpt = None

        for idx, row in df.iterrows():
            text = str(row[text_col])
            match = regex.search(text)
            if match:
                count += 1
                if example_excerpt is None:
                    start = max(0, match.start() - context_chars)
                    end = min(len(text), match.end() + context_chars)
                    example_excerpt = text[start:end].replace("\n", " ")
                    example_row_id = row.get("patient_filename", idx)

        rows.append({
            "scoring_system": system_name,
            "keyword_pattern": keyword_pattern,
            "num_reports_matched": count,
            "pct_of_total": round(100 * count / len(df), 2) if len(df) else 0,
            "example_patient_filename": example_row_id,
            "example_excerpt": example_excerpt,
        })

    result_df = pd.DataFrame(rows).sort_values("num_reports_matched", ascending=False)
    return result_df


def main():
    parser = argparse.ArgumentParser(description="Scan reports for named scoring/grading systems.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--text_col", default="cleaned_text")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--context_chars", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found. Available: {list(df.columns)}")

    result_df = scan(df, args.text_col, args.context_chars)
    result_df.to_csv(args.output_csv, index=False)

    print(f"Scanned {len(df)} reports for {len(SCORING_SYSTEMS)} known scoring systems.")
    print(f"Results written to: {args.output_csv}\n")
    print(result_df[["scoring_system", "num_reports_matched", "pct_of_total"]].to_string(index=False))


if __name__ == "__main__":
    main()

