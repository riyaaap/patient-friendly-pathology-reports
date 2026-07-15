# scripts/split_dataset.py
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def get_tss_code(patient_filename):
    return patient_filename.split(".")[0].split("-")[1]

def main():
    parser = argparse.ArgumentParser(description="Stratified 90/5/5 train/val/test split by cancer type (TSS-derived)")
    parser.add_argument("--input_csv", required=True, help="Cleaned/filtered or merged silver-label CSV/JSONL source")
    parser.add_argument("--tss_lookup_csv", default="data/reference/tcga_tss_codes.csv")
    parser.add_argument("--output_dir", default="data/processed/splits")
    parser.add_argument("--min_class_size", type=int, default=20,
                         help="Classes smaller than this go entirely to train (can't be safely stratified)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    tss_df = pd.read_csv(args.tss_lookup_csv, keep_default_na=False, na_values=[""])
    tss_map = dict(zip(tss_df["tss_code"], tss_df["study_name"]))

    df["tss_code"] = df["patient_filename"].apply(get_tss_code)
    df["study_name"] = df["tss_code"].map(tss_map)

    unmapped = df[df["study_name"].isna()]
    if len(unmapped) > 0:
        print(f"WARNING: {len(unmapped)} rows have unmapped TSS codes, sample:")
        print(unmapped[["patient_filename", "tss_code"]].head(10))
        print("These will be grouped into an 'UNKNOWN' stratum and sent entirely to train.")
        df["study_name"] = df["study_name"].fillna("UNKNOWN")

    class_counts = df["study_name"].value_counts()
    small_classes = class_counts[class_counts < args.min_class_size].index.tolist()
    if small_classes:
        print(f"NOTE: {len(small_classes)} study_name classes below min_class_size={args.min_class_size} "
              f"will be sent entirely to train (cannot be safely stratified):")
        print(class_counts[class_counts < args.min_class_size])

    df_small = df[df["study_name"].isin(small_classes)].copy()
    df_large = df[~df["study_name"].isin(small_classes)].copy()

    train_large, rest = train_test_split(
        df_large, test_size=0.10, stratify=df_large["study_name"], random_state=args.seed
    )
    val, test = train_test_split(
        rest, test_size=0.50, stratify=rest["study_name"], random_state=args.seed
    )

    train = pd.concat([train_large, df_small], ignore_index=True)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    import os
    os.makedirs(args.output_dir, exist_ok=True)
    train.to_csv(f"{args.output_dir}/train.csv", index=False)
    val.to_csv(f"{args.output_dir}/val.csv", index=False)
    test.to_csv(f"{args.output_dir}/test.csv", index=False)

    print(f"\nTotal rows: {len(df)}")
    print(f"Train: {len(train)} ({len(train)/len(df):.1%})")
    print(f"Val:   {len(val)} ({len(val)/len(df):.1%})")
    print(f"Test:  {len(test)} ({len(test)/len(df):.1%})")
    print(f"\nSmall/unstratifiable classes routed to train only: {len(small_classes)} classes, {len(df_small)} rows")

if __name__ == "__main__":
    main()


