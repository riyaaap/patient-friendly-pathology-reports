# scripts/gen_qualitative_example.py

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd

MODEL_PATH = "models/llama-3.1-8b-base"
ADAPTER_PATH = "checkpoints/phase_a_sft/final_adapter"
SPLITS_DIR = "data/processed/tcga_splits"
SILVER_LABELS_PATH = "data/processed/silver_labels_final.jsonl"

PROMPT_TEMPLATE = (
    "### Pathology Report:" + chr(10) +
    "{raw_report}" + chr(10) + chr(10) +
    "### Patient-Friendly Explanation:" + chr(10)
)


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

val_split = pd.read_csv(f"{SPLITS_DIR}/val.csv")
sample_filename = val_split.iloc[0]["patient_filename"]  # swap index to pick a different example

silver = {}
with open(SILVER_LABELS_PATH) as f:
    for line in f:
        r = json.loads(line)
        silver[r["patient_filename"]] = r

row = silver[sample_filename]
raw_report = row["raw_report"] if "raw_report" in row else row.get("cleaned_text")
reference_target = row["silver_target"]

prompt = PROMPT_TEMPLATE.format(raw_report=raw_report)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096, add_special_tokens=False).to("cuda:0")


output = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,   # deterministic, matches eval's greedy setup
    pad_token_id=tokenizer.eos_token_id,
)
generated_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("PATIENT_FILENAME:", sample_filename)
print("\n=== RAW REPORT (truncated to 500 chars) ===\n", raw_report[:500], "...")
print("\n=== REFERENCE (silver target) ===\n", reference_target)
print("\n=== MODEL GENERATED ===\n", generated_text)

