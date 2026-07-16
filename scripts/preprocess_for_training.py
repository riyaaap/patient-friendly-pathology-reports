import json
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

MODEL_PATH = "meta-llama/Llama-3.1-8B"
SILVER_LABELS_PATH = "data/processed/silver_labels_final.jsonl"
SPLITS_DIR = "data/processed/splits"
OUTPUT_DIR = "data/processed/tokenized"
MAX_LENGTH = 4096

PROMPT_TEMPLATE = (
    "### Pathology Report:" + chr(10) +
    "{raw_report}" + chr(10) + chr(10) +
    "### Patient-Friendly Explanation:" + chr(10)
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

silver = {}
with open(SILVER_LABELS_PATH) as f:
    for line in f:
        r = json.loads(line)
        silver[r["patient_filename"]] = r["silver_target"]

def build_example(raw_report, target_text):
    prompt = PROMPT_TEMPLATE.format(raw_report=raw_report)
    full_text = prompt + target_text + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    labels = full_ids.copy()
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

for split_name in ["train", "val", "test"]:
    split_df = pd.read_csv(f"{SPLITS_DIR}/{split_name}.csv")
    examples = []
    missing = 0
    for _, row in split_df.iterrows():
        fname = row["patient_filename"]
        if fname not in silver:
            missing += 1
            continue
        target_text = silver[fname]
        raw_report = row["cleaned_text"] if "cleaned_text" in split_df.columns else row.get("raw_report", "")
        examples.append(build_example(raw_report, target_text))

    ds = Dataset.from_list(examples)
    ds.save_to_disk(f"{OUTPUT_DIR}/{split_name}")
    print(split_name, "- saved:", len(examples), "| missing from silver labels:", missing)


