"""
Post-process silver-label output: enforce hardcoded, source-cited reference
tables for known scoring systems (Tier 1), and strip any freely-generated
tables for named-but-unregistered systems (Tier 2), rather than trusting the
teacher LLM to reproduce clinical lookup tables from parametric memory.

Usage:
    python scripts/apply_reference_table_registry.py \
        --input_jsonl data/processed/silver_labels_dryrun.jsonl \
        --output_jsonl data/processed/silver_labels_dryrun_validated.jsonl \
        --registry_dir configs/clinical_reference_tables \
        --report_csv data/processed/registry_application_report.csv
"""

import argparse
import json
import re
import yaml
import pandas as pd
from pathlib import Path

# Systems we know are present in the data but have deliberately NOT built a
# registry entry for yet (site-dependent staging, unsafe to auto-generate).
# Any table the teacher produces under these names gets stripped, not trusted.
DEFERRED_UNREGISTERED_SYSTEMS = {
    "TNM": [r"\btnm\b"],
    "FIGO": [r"\bfigo\b"],
    "AJCC": [r"\bamerican joint committee\b", r"\bajcc\b"],
    "Masaoka": [r"\bmasaoka\b"],
    "Clark Level": [r"\bclark'?s?\s?level\b"],
    "Edmondson-Steiner": [r"\bedmondson[\s-]?steiner\b"],
    "WHO/ISUP (renal)": [r"\bwho[\s/-]?isup\b"],
    "FAB Classification": [r"\bfrench[\s-]?american[\s-]?british\b|\bfab\b"],
}


def load_registry(registry_dir: str):
    registry = []
    for path in Path(registry_dir).glob("*.yaml"):
        with open(path) as f:
            entry = yaml.safe_load(f)
            entry["_source_file"] = path.name
            registry.append(entry)
    return registry


def detect_matches(text: str, patterns: list) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def find_markdown_tables(text: str):
    """Return list of (start_idx, end_idx, table_text) for markdown tables."""
    pattern = re.compile(r"(\|.+\|\n\|[-:\| ]+\|\n(?:\|.*\|\n?)+)")
    return [(m.start(), m.end(), m.group()) for m in pattern.finditer(text)]


def get_table_header(table_text: str) -> str:
    first_line = table_text.strip().split(chr(10))[0]
    return first_line.lower()


def replace_or_insert_table(silver_text: str, canonical_table_md: str,
                             explanation: str, heading_hint: str,
                             header_signature: list):
    """
    Identify the correct table to replace by matching column header
    signatures (e.g. "Grade Group" only appears in the Gleason reference
    table, never in the Section 1 diagnosis table). This avoids the failure
    mode where keyword-proximity or table-count heuristics get confused by
    unrelated tables that happen to mention the same term in passing.
    """
    tables = find_markdown_tables(silver_text)
    if not tables:
        return silver_text, "NO_TABLE_FOUND_FLAG_FOR_MANUAL_INSERT"

    if not header_signature:
        # No signature defined for this system yet -- fall back to the
        # old conservative behavior rather than guessing blindly.
        if len(tables) == 1:
            start, end, _ = tables[0]
            new_text = silver_text[:start] + canonical_table_md + silver_text[end:]
            return new_text, "REPLACED_OK_NO_SIGNATURE_SINGLE_TABLE"
        return silver_text, "MULTIPLE_TABLES_NO_SIGNATURE_MANUAL_REVIEW_NEEDED"

    matches = []
    for start, end, table_text in tables:
        header = get_table_header(table_text)
        if any(sig.lower() in header for sig in header_signature):
            matches.append((start, end))

    if len(matches) == 0:
        return silver_text, "NO_MATCHING_HEADER_FOUND_FLAG_FOR_MANUAL_INSERT"
    if len(matches) > 1:
        return silver_text, "MULTIPLE_MATCHING_HEADERS_MANUAL_REVIEW_NEEDED"

    start, end = matches[0]
    new_text = silver_text[:start] + canonical_table_md + silver_text[end:]
    return new_text, "REPLACED_OK"


SECTION1_TABLE_HEADER_SIGNATURE = [
    "Biopsy Result and Type of Biopsy",
    "Diagnosis",
    "Grade",
    "Amount",
]


def get_table_header_columns(table_text: str):
    first_line = table_text.strip().splitlines()[0]
    return [c.strip() for c in first_line.strip("|").split("|")]


def is_protected_section1_table(table_text: str) -> bool:
    cols = get_table_header_columns(table_text)
    return all(sig in cols for sig in SECTION1_TABLE_HEADER_SIGNATURE)

def strip_table_for_deferred_system(silver_text: str, system_label: str):
    tables = find_markdown_tables(silver_text)
    if not tables:
        return silver_text, "NO_ACTION_NO_TABLE_PRESENT"

    protected = [t for t in tables if is_protected_section1_table(t[2])]
    candidates = [t for t in tables if not is_protected_section1_table(t[2])]

    note_after_note = "Note: " + system_label + " staging is mentioned in the source report. A general reference table is not shown here because this system's interpretation depends on cancer type or site. Please discuss the specific staging with your doctor."
    note_replace_text = "Note: " + system_label + " staging is present in the source report. A general reference table is not shown here because this system's interpretation depends on cancer type or site. Please discuss the specific staging with your doctor."

    if len(candidates) == 0:
        if protected:
            start, end, _ = protected[0]
            replacement_note = chr(10) + chr(10) + "_" + note_after_note + "_" + chr(10)
            new_text = silver_text[:end] + replacement_note + silver_text[end:]
            return new_text, "NOTE_APPENDED_AFTER_SECTION1_TABLE"
        return silver_text, "NO_ACTION_NO_TABLE_PRESENT"

    if len(candidates) > 1:
        return silver_text, "MULTIPLE_NON_PROTECTED_TABLES_MANUAL_REVIEW_NEEDED"

    start, end, _ = candidates[0]
    replacement_note = chr(10) + "_" + note_replace_text + "_" + chr(10)
    new_text = silver_text[:start] + replacement_note + silver_text[end:]
    return new_text, "STRIPPED_DEFERRED_SYSTEM"

def process(input_jsonl, output_jsonl, registry_dir, report_csv):
    registry = load_registry(registry_dir)
    report_rows = []

    with open(input_jsonl) as f_in, open(output_jsonl, "w") as f_out:
        for line in f_in:
            record = json.loads(line)
            raw_text = record.get("raw_report", "")
            silver_text = record.get("silver_target", "")
            patient_id = record.get("patient_filename", "UNKNOWN")

            action_log = []

            # --- Tier 1: registered systems ---
            for entry in registry:
                patterns = entry.get("detection_patterns", [])
                if detect_matches(raw_text, patterns):
                    # Ki-67 lightweight handling: only force a table if the
                    # conditional NET trigger also matches; otherwise just
                    # ensure explanation text is present (not enforced here
                    # automatically — left as a lighter-touch flag).
                    if entry.get("is_lightweight"):
                        cond = entry.get("conditional_table_trigger", {})
                        cond_patterns = cond.get("detection_patterns", [])
                        if cond_patterns and detect_matches(raw_text, cond_patterns):
                            silver_text, status = replace_or_insert_table(
                                silver_text, cond["table_markdown"],
                                entry["explanation_text"], entry["system_name"],
                                entry.get("table_header_signature", [])
                            )
                            action_log.append(f"{entry['system_name']} (NET-conditional): {status}")
                        else:
                            action_log.append(f"{entry['system_name']}: lightweight, no table enforced")
                        continue

                    silver_text, status = replace_or_insert_table(
                        silver_text, entry["table_markdown"],
                        entry["explanation_text"], entry["system_name"],
                        entry.get("table_header_signature", [])
                    )
                    action_log.append(f"{entry['system_name']}: {status}")

            # --- Tier 2: deferred/unregistered named systems ---
            for system_label, patterns in DEFERRED_UNREGISTERED_SYSTEMS.items():
                if detect_matches(raw_text, patterns):
                    silver_text, status = strip_table_for_deferred_system(silver_text, system_label)
                    action_log.append(f"DEFERRED[{system_label}]: {status}")

            record["silver_target"] = silver_text
            record["registry_actions"] = action_log
            f_out.write(json.dumps(record) + "\n")

            report_rows.append({
                "patient_filename": patient_id,
                "actions": "; ".join(action_log) if action_log else "none",
            })

    pd.DataFrame(report_rows).to_csv(report_csv, index=False)
    print(f"Processed {len(report_rows)} records.")
    print(f"Updated silver labels written to: {output_jsonl}")
    print(f"Action report written to: {report_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--registry_dir", default="configs/clinical_reference_tables")
    parser.add_argument("--report_csv", required=True)
    args = parser.parse_args()
    process(args.input_jsonl, args.output_jsonl, args.registry_dir, args.report_csv)


if __name__ == "__main__":
    main()
