# Excluded Long Reports (Token-Overflow during teacher label generation, Deferred for now, may use later on)

These 7 reports exceed the teacher server's `--max-model-len 8192` token budget,
causing `400 Bad Request` errors during teacher silver-label generation.
They are NOT truncated (risk of losing clinically important info) and are
excluded from the teacher-generated silver-label training set.

**Status:** Deferred. Options under consideration: chunked/parallel teacher
generation, or student-only exposure (no silver label, seen only during
raw fine-tuning if included later). Not yet decided. Low priority (~0.08% of data).

| patient_filename | char_length |
|---|---|
| TCGA-E2-A109.089E0FB9-6B45-4E3E-843E-EC98D39863F6 | 24091 |
| TCGA-CN-6022.f7b8c864-6fcd-4b64-bcee-74bc3dcf6c39 | 18207 |
| TCGA-CN-A49B.3B5F227F-B65D-483A-8BCE-A8C079326510 | 17531 |
| TCGA-CN-A642.79C16F29-FA94-44CD-A971-558B23508CE3 | 16653 |
| TCGA-W8-A86G.A5433CA5-1DCC-4711-9B3B-6478AE7B5FAA | 24189 |
| TCGA-KM-A7QK.192A2806-61A7-49C9-91D6-B345DDFB7331 | 25640 |
| TCGA-E2-A158.1CEBF20D-767A-4A19-AA41-B9EAFCF05176 | 26161 |

**Total excluded:** 7 rows
**Corrected final dataset rows target:** 9057 − 7 = **9050** rows
