import re

with open('scripts/apply_reference_table_registry.py', 'r') as f:
    content = f.read()

new_block = '''SECTION1_TABLE_HEADER_SIGNATURE = [
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

    if len(candidates) == 0:
        if protected:
            _, end, _ = protected[0]
            replacement_note = (
                f"\\n\\n_Note: {system_label} staging is mentioned in the source "
                f"report. A general reference table is not shown here because "
                f"this system's interpretation depends on cancer type/site — "
                f"please discuss the specific staging with your doctor._\\n"
            )
            new_text = silver_text[:end] + replacement_note + silver_text[end:]
            return new_text, "NOTE_APPENDED_AFTER_SECTION1_TABLE"
        return silver_text, "NO_ACTION_NO_TABLE_PRESENT"

    if len(candidates) > 1:
        return silver_text, "MULTIPLE_NON_PROTECTED_TABLES_MANUAL_REVIEW_NEEDED"

    start, end, _ = candidates[0]
    replacement_note = (
        f"\\n_Note: {system_label} staging is present in the source report. "
        f"A general reference table is not shown here because this system's "
        f"interpretation depends on cancer type/site — please discuss the "
        f"specific staging with your doctor._\\n"
    )
    new_text = silver_text[:start] + replacement_note + silver_text[end:]
    return new_text, "STRIPPED_DEFERRED_SYSTEM"
'''

pattern = re.compile(
    r"def strip_table_for_deferred_system\(.*?\n(?=def |class |\Z)",
    re.DOTALL
)

new_content, n = pattern.subn(new_block + "\n\n", content, count=1)

if n == 0:
    print("NO MATCH — do not proceed, inspect manually")
else:
    with open('scripts/apply_reference_table_registry.py', 'w') as f:
        f.write(new_content)
    print(f"Patched successfully, {n} replacement(s) made")
