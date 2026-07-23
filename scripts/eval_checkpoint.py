import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from peft import PeftModel
import textstat
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import pandas as pd
from evaluate import load as load_metric

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "device(s)")
assert torch.cuda.device_count() == 1, "This eval script is single-GPU only."

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_name", required=True, help="e.g. checkpoint_0_base or phase_a_sft")
parser.add_argument("--model_path", required=True, help="path to base model or adapter dir")
parser.add_argument("--adapter_path", default=None, help="optional LoRA adapter path, omit for base model")
parser.add_argument("--n_samples", type=int, default=20, help="number of val rows to eval on (small for quick test)")
args = parser.parse_args()

VAL_PATH = "data/processed/tcga_splits/val.csv"
OUT_DIR = f"eval_results/{args.checkpoint_name}"
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_TEMPLATE = (
    "### Pathology Report:" + chr(10) +
    "{raw_report}" + chr(10) + chr(10) +
    "### Patient-Friendly Explanation:" + chr(10)
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map={"": 0})
if args.adapter_path:
    model = PeftModel.from_pretrained(model, args.adapter_path)
model.eval()

SILVER_LABELS_PATH = "data/processed/silver_labels_final.jsonl"
# Load silver labels into a lookup dict keyed by patient_filename
silver_lookup = {}
with open(SILVER_LABELS_PATH) as f:
    for line in f:
        row = json.loads(line)
        silver_lookup[row["patient_filename"]] = row

val_meta = pd.read_csv(VAL_PATH).head(args.n_samples)

val_df = []
for _, meta_row in val_meta.iterrows():
    fname = meta_row["patient_filename"]
    if fname not in silver_lookup:
        print(f"WARNING: {fname} not found in silver labels, skipping")
        continue
    entry = silver_lookup[fname]
    val_df.append({"raw_report": entry["raw_report"], "silver_target": entry["silver_target"]})

val_df = pd.DataFrame(val_df)

generations, references, sources = [], [], []
for _, row in val_df.iterrows():
    prompt = PROMPT_TEMPLATE.format(raw_report=row["raw_report"])   # fixed: wrap in template
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    generations.append(gen_text)
    references.append(row["silver_target"])
    sources.append(prompt)

# Save raw generations
with open(f"{OUT_DIR}/generations.jsonl", "w") as f:
    for s, g, r in zip(sources, generations, references):
        f.write(json.dumps({"source": s, "generated": g, "reference": r}) + "\n")

# Metrics
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l_scores = [scorer.score(r, g)["rougeL"].fmeasure for r, g in zip(references, generations)]

bleu_metric = load_metric("bleu")
def compute_bleu(predictions, references):
    """
    predictions: list[str], model-generated patient-friendly reports
    references: list[str], teacher-label target ground truth
    """
    results = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],  # BLEU expects list-of-lists for refs
    )
    return {
        "bleu": results["bleu"],
        "bleu_precisions": results["precisions"],
    }

bleu_results = compute_bleu(generations, references)

_, _, bert_f1 = bertscore(generations, references, lang="en", verbose=False)

fk_scores = [textstat.flesch_kincaid_grade(g) for g in generations]
gf_scores = [textstat.gunning_fog(g) for g in generations]

metrics = {
    "checkpoint": args.checkpoint_name,
    "n_samples": len(generations),
    "rouge_l_mean": sum(rouge_l_scores) / len(rouge_l_scores),
    "bleu": bleu_results["bleu"], # scalar for heatmap
    "bleu_precisions": bleu_results["bleu_precisions"], # not for heatmap
    "bertscore_f1_mean": bert_f1.mean().item(),
    "flesch_kincaid_mean": sum(fk_scores) / len(fk_scores),
    "gunning_fog_mean": sum(gf_scores) / len(gf_scores),
}

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print(f"Saved to {OUT_DIR}/metrics.json")
