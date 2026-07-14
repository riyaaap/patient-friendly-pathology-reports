"""
clean_and_filter_reports.py

Loads the Kefeli et al. TCGA-Reports Mendeley CSV (patient_filename, text),
cleans the raw report text, plots a histogram of report word counts, filters out
the bottom 5% shortest reports, and writes a cleaned CSV ready for tokenization/training.

Usage:
    python clean_and_filter_reports.py \
        --input_csv /path/to/tcga_reports.csv \
        --filename_col patient_filename \
        --text_col text \
        --output_dir ./data_out
"""

import argparse
import os
import re
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def clean_report_text(text: str) -> str:
    """
    Cleans a single raw pathology report string.

    Handles common artifacts seen in OCR'd / plain-text-extracted pathology reports:
    - Normalizes line endings and collapses excessive whitespace/newlines
    - Strips form-feed / page-break characters
    - Removes repeated redaction placeholders (e.g. "[Patient Name]", "XXXX") -- keep the
      placeholder itself, just normalize spacing around it
    - Fixes hyphenation artifacts from line-wrapped OCR text ("carcin-\noma" -> "carcinoma")
    - Strips leading/trailing whitespace
    - Normalizes multiple spaces to a single space, but preserves paragraph breaks (double newline)
    """
    if not isinstance(text, str):
        return ""

    t = text

    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # Remove form-feed / page-break control characters
    t = t.replace("\x0c", "\n")

    # Fix hyphenated line-wrap artifacts: "word-\nword" -> "wordword"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Collapse 3+ newlines down to a double newline (paragraph break)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # Collapse runs of spaces/tabs (but not newlines) into a single space
    t = re.sub(r"[ \t]{2,}", " ", t)

    # Strip trailing whitespace on each line
    t = "\n".join(line.strip() for line in t.split("\n"))

    # Remove stray non-printable / control characters except newline
    t = re.sub(r"[^\x20-\x7E\n]", "", t)

    # Final strip
    t = t.strip()

    return t


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(description="Clean and filter TCGA pathology report CSV")
    parser.add_argument("--input_csv", required=True, help="Path to raw Mendeley CSV")
    parser.add_argument("--filename_col", default="patient_filename", help="Column name for patient/filename identifier")
    parser.add_argument("--text_col", default="text", help="Column name for raw report text")
    parser.add_argument("--output_dir", default="./data_out", help="Directory to write cleaned CSV + histogram")
    parser.add_argument("--filter_percentile", type=float, default=5.0, help="Bottom percentile of word counts to drop")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_csv):
        print(f"ERROR: input CSV not found at {args.input_csv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)

    missing_cols = [c for c in (args.filename_col, args.text_col) if c not in df.columns]
    if missing_cols:
        print(f"ERROR: expected column(s) not found in CSV: {missing_cols}", file=sys.stderr)
        print(f"Columns present: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    n_start = len(df)
    print(f"Loaded {n_start} rows.")

    # --- Basic hygiene before cleaning ---
    # Drop exact duplicate rows
    df = df.drop_duplicates(subset=[args.filename_col, args.text_col])
    n_after_dedupe = len(df)
    if n_after_dedupe != n_start:
        print(f"Dropped {n_start - n_after_dedupe} exact duplicate rows.")

    # Drop rows with null/empty filename or text
    df = df.dropna(subset=[args.filename_col, args.text_col])
    df = df[df[args.text_col].astype(str).str.strip() != ""]
    n_after_na = len(df)
    if n_after_na != n_after_dedupe:
        print(f"Dropped {n_after_dedupe - n_after_na} rows with missing filename/text.")

    # --- Clean text ---
    print("Cleaning report text...")
    df["cleaned_text"] = df[args.text_col].apply(clean_report_text)

    # Drop any rows that became empty after cleaning
    df = df[df["cleaned_text"].str.strip() != ""]

    # --- Word counts + histogram ---
    df["word_count"] = df["cleaned_text"].apply(word_count)

    print(df["word_count"].describe())

    hist_path = os.path.join(args.output_dir, "report_word_count_histogram.png")
    plt.figure(figsize=(10, 6))
    plt.hist(df["word_count"], bins=60, edgecolor="black")
    plt.axvline(
        df["word_count"].quantile(args.filter_percentile / 100.0),
        color="red",
        linestyle="--",
        label=f"{args.filter_percentile:.0f}th percentile cutoff",
    )
    plt.xlabel("Word count per report")
    plt.ylabel("Number of reports")
    plt.title("Distribution of TCGA Pathology Report Word Counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    print(f"Histogram saved to {hist_path}")

    # --- Filter bottom N% shortest reports ---
    cutoff = df["word_count"].quantile(args.filter_percentile / 100.0)
    n_before_filter = len(df)
    df_filtered = df[df["word_count"] >= cutoff].copy()
    n_after_filter = len(df_filtered)

    print(
        f"Filtering out reports below the {args.filter_percentile:.0f}th percentile "
        f"(< {cutoff:.0f} words): removed {n_before_filter - n_after_filter} rows, "
        f"{n_after_filter} rows remain."
    )

    # --- Save cleaned + filtered dataset ---
    out_csv = os.path.join(args.output_dir, "tcga_reports_cleaned_filtered.csv")
    out_cols = [args.filename_col, "cleaned_text", "word_count"]
    df_filtered[out_cols].to_csv(out_csv, index=False)
    print(f"Cleaned + filtered CSV written to {out_csv}")

    # Quick summary printout
    print("\n--- Summary ---")
    print(f"Starting rows:          {n_start}")
    print(f"After dedupe/NA drop:   {n_after_na}")
    print(f"After text cleaning:    {len(df)}")
    print(f"After {args.filter_percentile:.0f}% length filter: {n_after_filter}")


if __name__ == "__main__":
    main()
