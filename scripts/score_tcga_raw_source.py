"""
Scores raw TCGA pathology report text directly against the reference
(silver_target) — a floor baseline: how far is raw clinical text from a
patient-friendly explanation with zero processing?
"""
import json, os
import pandas as pd
# TODO: adjust these imports to your actual function names in eval_checkpoint.py
from eval_checkpoint import compute_rouge, compute_bleu, compute_bertscore, compute_readability

VAL_PATH = "data/processed/tcga_splits/val.csv"
TEST_PATH = "data/processed/tcga_splits/test.csv"
SILVER_LABELS_PATH = "data/processed/silver_labels_final.jsonl"

combined = pd.concat([pd.read_csv(VAL_PATH), pd.read_csv(TEST_PATH)], ignore_index=True)
silver = pd.read_json(SILVER_LABELS_PATH, lines=True)
merged = combined.merge(silver, on="patient_filename", how="inner")

n_total = len(merged)
merged = merged[merged["error"].isna()]
n_excluded = n_total - len(merged)

predictions = merged["raw_report"].tolist()
references = merged["silver_target"].tolist()

rouge = compute_rouge(predictions, references)
bleu_r = compute_bleu(predictions, references)
bert = compute_bertscore(predictions, references)
read = compute_readability(predictions)

metrics = {
    "checkpoint": "tcga_raw_source",
    "n_samples": len(merged),
    "n_excluded": n_excluded,
    "rouge_l_mean": rouge["rouge_l_mean"],
    "bleu": bleu_r["bleu"],
    "bleu_precisions": bleu_r["bleu_precisions"],
    "bertscore_f1_mean": bert["bertscore_f1_mean"],
    "flesch_kincaid_mean": read["flesch_kincaid_mean"],
    "gunning_fog_mean": read["gunning_fog_mean"],
}
os.makedirs("eval_results/tcga_raw_source", exist_ok=True)
with open("eval_results/tcga_raw_source/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
