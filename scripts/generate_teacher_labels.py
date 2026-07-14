"""
generate_teacher_labels.py

Generates patient-friendly "silver label" pathology report rewrites using a
locally-hosted teacher LLM (via vLLM's OpenAI-compatible chat completions API).
Teacher LLM model used is Qwen2.5-14B
"""

"""
To run this file in terminal to view sample generated output for a few reports, do:
python -c "
import json

with open('data/processed/silver_labels_dryrun.jsonl') as f:
    lines = [json.loads(l) for l in f]

for entry in lines:
    if 'EB-A82B' in entry['patient_filename']:
        print('=== RAW REPORT ===')
        print(entry['raw_report'][:500])
        print()
        print('=== SILVER TARGET ===')
        print(entry['silver_target'])
        print('='*80)
"

... where 'E8-A82B' can be replaced with the specific TCGA report ID wanted to be viewed
"""

import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests


SYSTEM_PROMPT = """You are a medical writing assistant that rewrites raw pathology reports into patient-friendly reports for a lay audience (target reading level: 8th-10th grade, Flesch-Kincaid Grade Level).

You MUST follow this exact structure, using these exact section headers in this exact order:

{{PATIENT_HEADER}}

## About This Report
Briefly explain why biopsies are done (what they screen for, what kind of test was used), in plain language.

## Your Biopsy Results

### 1. Your Diagnosis
Include a small side-by-side summary table with these rows ONLY:
- Biopsy result and type of biopsy
- Diagnosis
- Grade
- Amount (number of samples with cancer)
Do NOT include a "Risk Category" row unless the source report explicitly states a formal named risk category (e.g. "low risk," "intermediate risk," "high risk," NCCN risk group). If no such explicit statement exists in the source report, omit the row entirely.

Briefly explain how aggressiveness/grade is determined in plain language.

### 2. What This Means
Explain what the diagnosis means for the patient in plain language, grounded ONLY in what the report states.

### 3. Key Markers & Numbers Explained
Explain any specific markers, scores, or measurements mentioned in the report in plain language.

If and only if a Gleason score (or a similar score for a different type of cancer,
in which case you would explain the same relevant details in a similar format for interpreting that score)
is present in the source report, include this exact subsection:

#### What is a Gleason Score?
(a) Explain what a Gleason score is.
(b) Explain how it is reported: two numbers added together.
(c) Explain how the resulting Gleason sum maps to the Grade Group (1-5) system.
(d) Include a small table correlating Gleason sum / Grade Group to standard published risk tiers, clearly labeled as general reference information.
Do NOT include this subsection if no Gleason score appears in the source report.

### 4. Inline Term Definitions
Formatting rule: throughout sections 1-3, define medical terms inline the first time they appear.

### 5. Glossary Recap
A short bulleted recap of key terms, only included if there are 5+ terms to list.

### 6. Next Steps & Talking to Your Doctor
Explain the diagnosis only. NEVER recommend a specific treatment. Redirect to "talk to your doctor about treatment options."
Include: {{OFFICE_CONTACT_INFO}}

### 7. Additional Notes
{{PHYSICIAN_NOTES}}

IMPORTANT RULES:
- Never fabricate patient names, dates, or contact info. Leave {{PATIENT_HEADER}}, {{OFFICE_CONTACT_INFO}}, {{PHYSICIAN_NOTES}} untouched as literal blank placeholders.
- Never state a Risk Category unless explicitly present in the source report.
- Never recommend a specific treatment.
- Ground every statement in the source report content only.
"""

def build_messages(raw_report: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the raw pathology report:\n\n{raw_report}\n\nRewrite it following the required structure exactly.",
        },
    ]

def call_teacher_model(vllm_url: str, model_name: str, raw_report: str,
                        max_retries: int = 3, timeout: int = 120) -> dict:
    payload = {
        "model": model_name,
        "messages": build_messages(raw_report),
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(vllm_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {"silver_target": content, "error": None}
        except Exception as e:
            last_error = str(e)
            time.sleep(2 ** attempt)

    return {"silver_target": None, "error": last_error}


def process_row(row: dict, vllm_url: str, model_name: str) -> dict:
    result = call_teacher_model(vllm_url, model_name, row["cleaned_text"])
    return {
        "patient_filename": row["patient_filename"],
        "raw_report": row["cleaned_text"],
        "silver_target": result["silver_target"],
        "error": result["error"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--vllm_url", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.limit:
        df = df.head(args.limit)

    rows = df.to_dict(orient="records")
    print(f"Processing {len(rows)} reports with {args.max_workers} workers...")

    write_lock = threading.Lock()
    completed = 0
    errors = 0

    with open(args.output_jsonl, "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(process_row, row, args.vllm_url, args.model_name): row
                for row in rows
            }
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                completed += 1
                if result["error"]:
                    errors += 1
                    print(f"[{completed}/{len(rows)}] ERROR on {result['patient_filename']}: {result['error']}")
                else:
                    print(f"[{completed}/{len(rows)}] OK: {result['patient_filename']}")

    print(f"\nDone. {completed} processed, {errors} errors.")
    print(f"Output written to: {args.output_jsonl}")


if __name__ == "__main__":
    main()



