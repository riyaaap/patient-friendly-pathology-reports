import json, glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#CHECKPOINT_ORDER = ["checkpoint_0_base", "phase_a_sft", "phase_b_sft", "phase_c_dpo"]
CHECKPOINT_ORDER = ["checkpoint_0_base", "phase_a_sft"]  # abstract-submission subset; restore full list later

rows = []
for ckpt in CHECKPOINT_ORDER:
    path = f"eval_results/{ckpt}/metrics.json"
    with open(path) as f:
        metrics = json.load(f)
    metrics["checkpoint"] = ckpt
    rows.append(metrics)

df = pd.DataFrame(rows).set_index("checkpoint")

# after loading metrics.json into `df` (one row per checkpoint)
#heatmap_cols = [
#    "rouge_l_mean", "bleu", "bertscore_f1_mean",
#    "flesch_kincaid_mean", "gunning_fog_mean",
#]
#df = df[heatmap_cols]
df = df.select_dtypes(exclude=["object"])  # drops list/str columns automatically

# normalize per-metric (min-max) since scales differ (loss vs. BERTScore vs. ROUGE)
df_norm = (df - df.min()) / (df.max() - df.min())

plt.figure(figsize=(10, 6))
sns.heatmap(df_norm.T, annot=df.T, fmt=".3f", cmap="RdYlGn", cbar_kws={"label": "normalized score"})
plt.title("Evaluation Metrics Across Training Phases")
plt.tight_layout()
plt.savefig("eval_results/heatmap_all_phases.png", dpi=300)
