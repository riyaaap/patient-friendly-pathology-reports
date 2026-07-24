"""
generate_heatmap.py
Normalized metric-comparison heatmap across pipeline stages:
TCGA Raw Source -> Teacher LLM (self-consistency) -> checkpoint_0_base -> phase_a_sft

Reads eval_results/<name>/metrics.json for each entry in CHECKPOINT_ORDER.
Any stage whose metrics.json doesn't exist yet is skipped (so columns/rows
can be filled in incrementally without breaking the script).
"""

import json, glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

EVAL_RESULTS_DIR = "eval_results"

# Explicit row order — controls top-to-bottom order in the heatmap.
CHECKPOINT_ORDER = [
    "tcga_raw_source",     # raw pathology report scored directly vs reference (floor baseline)
    "teacher_llm",          # Qwen2.5-14B fresh generation vs its own frozen silver_target (self-consistency ceiling)
    "checkpoint_0_base",    # Llama-3.1-8B, no fine-tuning
    "phase_a_sft",          # Llama-3.1-8B + Phase A LoRA
]

DISPLAY_NAMES = {
    "tcga_raw_source": "TCGA Raw Report (ground truth)",
    "teacher_llm": "Teacher LLM (Qwen2.5-14B, 'summary' ground truth)",
    "checkpoint_0_base": "Student Base (Llama-3.1-8B, no FT)",
    "phase_a_sft": "Student Phase A (LoRA SFT)",
}


HEATMAP_COLS = ["rouge_l_mean", "bleu", "bertscore_f1_mean",
                "flesch_kincaid_mean", "gunning_fog_mean"]

COL_LABELS = {
    "rouge_l_mean": "ROUGE-L",
    "bleu": "BLEU",
    "bertscore_f1_mean": "BERTScore (F1)",
    "flesch_kincaid_mean": "Flesch-Kincaid Grade",
    "gunning_fog_mean": "Gunning Fog Index",
}

# Metrics where a LOWER raw value is the better outcome
LOWER_IS_BETTER = {"flesch_kincaid_mean", "gunning_fog_mean"}

# ---- Load metrics ---- #

rows, sample_info, missing = {}, {}, []

for name in CHECKPOINT_ORDER:
    path = os.path.join(EVAL_RESULTS_DIR, name, "metrics.json")
    if not os.path.exists(path):
        missing.append(name)
        continue
    with open(path) as f:
        m = json.load(f)
    rows[name] = {col: m.get(col) for col in HEATMAP_COLS}
    sample_info[name] = {
        "n_samples": m.get("n_samples", "N/A"),
        "n_excluded": m.get("n_excluded", "N/A"),
    }

if missing:
    print(f"[generate_heatmap] Skipping (metrics.json not found yet): {missing}")
if not rows:
    raise SystemExit("No metrics.json files found for any checkpoint in CHECKPOINT_ORDER.")

df = pd.DataFrame.from_dict(rows, orient="index")
df = df.reindex([n for n in CHECKPOINT_ORDER if n in df.index])  # preserve explicit order
df_display = df.rename(index=DISPLAY_NAMES, columns=COL_LABELS)

# ---- Normalize per column, inverting "lower is better" columns ----
df_norm = pd.DataFrame(index=df_display.index, columns=df_display.columns, dtype=float)
for raw_col in HEATMAP_COLS:
    disp_col = COL_LABELS[raw_col]
    col_min, col_max = df_display[disp_col].min(), df_display[disp_col].max()
    if col_max == col_min:
        df_norm[disp_col] = 0.5  # no variation across rows -> neutral color
    else:
        norm = (df_display[disp_col] - col_min) / (col_max - col_min)
        if raw_col in LOWER_IS_BETTER:
            norm = 1 - norm
        df_norm[disp_col] = norm

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 0.9 * len(df_display) + 2))
sns.heatmap(
    df_norm, annot=df_display.round(3), fmt="", cmap="RdYlGn",
    vmin=0, vmax=1, linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Normalized score (0=worst, 1=best, per column)"},
    ax=ax,
)
ax.set_title("Pipeline Stage Comparison — Automatic Metrics", fontsize=13, pad=12)
ax.set_xlabel("")
ax.set_ylabel("")

# ---- Legend/footnote with sample sizes ----
footnote_lines = [
    "Legend: color = relative performance per metric across rows shown (green=better, red=worse); "
    "annotated numbers = raw metric values. For Flesch-Kincaid / Gunning Fog, LOWER raw "
    "values are better (simpler reading level) and are colored accordingly."
]
for name in CHECKPOINT_ORDER:
    if name in sample_info:
        disp = DISPLAY_NAMES.get(name, name)
        info = sample_info[name]
        footnote_lines.append(f"{disp}: n={info['n_samples']}, excluded={info['n_excluded']}")

fig.text(0.01, -0.02, "\n".join(footnote_lines), ha="left", va="top", fontsize=7, wrap=True)

plt.tight_layout()
plt.savefig("eval_results/checkpoint_comparison_heatmap.png", dpi=300, bbox_inches="tight")
print("Saved: eval_results/checkpoint_comparison_heatmap.png")
