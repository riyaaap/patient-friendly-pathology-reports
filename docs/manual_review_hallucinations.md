# Manual Review: Initial Hallucinated Arithmetic Errors during Teacher label generation 

31 reports were excluded from the reports to be used as the initial Phase A SFT training set after the
arithmetic-validation pass flagged inconsistencies between the teacher
model's generated silver labels (e.g., reference tables mapping Gleason
scores to ISUP Grade Groups containing miscomputed entries such as
"Gleason 3+3 → total score 7" — the correct sum is 6) and raw reports.

**Root cause confirmed**: spot-checked against raw source reports; the
errors are NOT present in the original pathology text. These are teacher
model (Qwen2.5-14B-AWQ) hallucinations introduced during silver-label
generation, caught by `validate_score_arithmetic.py` before entering the
training set.

**Status**: Routed to `data/processed/silver_labels_manual_review.jsonl`.
Excluded from Phase A/B training data. Available for later manual
correction/reuse if desired, or as a hallucination-rate case study for
Phase D evaluation (Med-HALT) reporting.

**Count**: 31 rows (0.34% of the 9050-row merged dataset).
**Final clean training dataset size**: 9019 rows.

# ALSO: just a note for documentation on contents of files at this stage (pre-entering phase A SFT training..)
File      	Rows	Purpose
silver_labels_final.jsonl	9019	Clean training data, hallucination + long-report rows excluded
silver_labels_manual_review.jsonl	31	Documented teacher hallucinations, deferred
docs/excluded_long_reports.md	7	Token-overflow exclusions, documented
docs/manual_review_hallucinations.md	31	Hallucination exclusions, documented
data/processed/splits/{train,val,test}.csv	8117/451/451	Final stratified 90/5/5 split
