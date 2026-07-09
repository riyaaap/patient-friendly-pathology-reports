"""
inspect_short_reports.py

Prints the N shortest reports (filename, word_count, and a text preview) from the cleaned CSV
Usage:
    python inspect_short_reports.py --input_csv data/processed/tcga_reports_cleaned_filtered.csv --n 30
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--n", type=int, default=30, help="Number of shortest reports to display")
    parser.add_argument("--preview_chars", type=int, default=200, help="How many characters of text to preview")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    shortest = df.sort_values("word_count").head(args.n)

    for _, row in shortest.iterrows():
        preview = row["cleaned_text"][: args.preview_chars].replace("\n", " ")
        print(f"[{row['word_count']} words] {row['patient_filename']}")
        print(f"    {preview}{'...' if len(row['cleaned_text']) > args.preview_chars else ''}")
        print()

    # Also surface a quick breakdown by TCGA project code, if filenames follow the
    # standard TCGA-XX-XXXX-... naming convention, to check whether short reports
    # cluster in specific cancer types (which would argue for per-type thresholds
    # rather than one global cutoff)
    try:
        df["project_code"] = df["patient_filename"].str.extract(r"(TCGA-[A-Z]+)")
        print("--- Word count distribution by project code (if extractable) ---")
        print(df.groupby("project_code")["word_count"].describe()[["count", "mean", "50%", "min"]])
    except Exception as e:
        print(f"Could not extract project codes from filenames: {e}")


if __name__ == "__main__":
    main()
