Patient-Friendly Pathology Report — Output Schema
-> to define criteria for successful outputs of the model 

This schema defines the reuired structure, tone, and content rules for model output. 
, to use as: 1) a generation prompt template for teacher-LLM silver labels, 
2) fine-tuning target format, and 
3) LLM-as-a-judge scoring rubric. 
---
NOTE: this schema has 2 types of content: 
* TEMPLATE FIELDS (marked as [TEMPLATE -- not model output]): patient info/contact, etc. fields. These should stay as blank placeholders in all training data and generated output, never filled in at any stage of this pipeline. 
* GENERATED CONTENT: everything else, to actually be produced by the teacher-LLM and fine-tuned model, and to be scored on. 
---
TARGET READING LEVEL
8-10th grade (from Flesch-Kincaid Grade Level) -- preferably 8th grade, ceiling of 10. 
This is the standard commonly cited for patient health materials (ex., from AMA/NIH health literacy guides)
Should be enforced in evaluation w/ Flesch-Kincaid + Gunning Fog tests, + 
also checked during teacher-LLM generation. 

REQUIRED SECTION STRUCTURE (fixed headers in ~ this order) 

[TEMPLATE -- not model output] Top of Page: Patient Name, DOB, ID, Doctor Name, Biopsy date, Report date 
* kept entirely BLANK with an empty placeholder block, since all data will be de-identified
* keep this for the medical system to fill in later on 

About this report: brief, standardized info: why biopsies are done in general (what they screen for, what tests are used if mentioned in the source report, at a high level). 

YOUR BIOPSY RESULTS (this section group header can contain #1-7 below)

1. Your Diagnosis - 
Plain-language, 2-4 sentence summary of what was found. State diagnosis name once in full, 
followed by simple explanation/restatement in everyday language. 

Side-by-side summary table (required), containing: Biopsy result and type of Biopsy, 
Diagnosis, Grade, Amount (# of samples with cancer), [ONLY include info explicitly mentioned in the source report]. 
[Risk Category ONLY if explicitly stated in the source report, using a formal category (ex: the NCCN low/int/high for prostate). Otherwise, DO NOT INCLUDE THIS SECTION, as would be a hallucination] 

How aggressiveness is determined (subsection): brief context on how the scans are evaluated. 
Explain what pathologists look for to judge how abnormal the cells appear (general context, grounded in what report's own marker values indicate -- do NOT make any new claims beyond the report).

2. What This Means
Short paragraph(s) contextualizing the diagnosis, in terms of what condition/cancer it is, in the simplest accurate terms. Include general disease context, not patient-specific details. 

3. Key Markers & Numbers Explained
Bullet-point or brief breakdown of clinically significant markers, scores, or measurements present in the original report (grade, stage, tumor size, margins, lymph node status, biomarkers like ER/PR/HER2, KI-67, molecular markers, etc.)
Each bullet in the following format: 
    **[Marker name]:** [value from report]; [plain-language meaning]

[Section ONLY if a Gleason score appears in the source report, ex: prostate pathology]
Rule for Gleason scores specifically (ex: used heavily in prostate pathology reports, may be commonly confusing to patients)..
Section title: What is a Gleason Score? 
(a) WHAT it is: a grading system for how abnormal cancer cells appear under a microscope
(b) HOW it is reported: two numbers added together, respectively reflecting how much of the two most common patterns in the sample resembles cancer,
and with low numbers reflecting closer to normal tissue and high numbers meaning more abnormal
(c) Mapping to Grade Group: how the gleason sum obtained maps to the Grade group (1-5) system, especially since appear often on newer reports. 
(d) (small) Risk correlation table show a small table correlating low/medium/high risk categories to Gleason scores, as well as relevant values (ex: PSA (ng/mL), clinical T stage, etc.) with the result specific category highlighted. Use ONLY the standard published grade-group-to-risk mapping, NOT a personalized conclusion about the patient beyond anything explicitly stated in the source report. 
[Optional]: To learn more about [specific disease], visit: [links to reputable site for more info on prostate cancer or gleason scores, etc. as relevant] examples: (www.cancer.ca/prostate and www.mypathologyreport.ca/prostate) -- ONLY include real, verifiable links, NEVER fabricate a URL.
* NEVER state a prognosis or treatment implication from the score alone. 

4. Inline Term Definitions 
Formatting rule (applies throughout all sections, this is not a separate block/section).:
any medical jargon has a plain-language definition delineated immediately after, in the following format: 
term *(plain-language definition)* using italics and parentheses to easily distinguish the definition
without having to reference a separate glossary

5. Glossary Recap (OPTIONAL, only included if 5+ distinct jargon terms are present)
Short bullet list at the very end, repeating the term + definition pairs given inline for quick reference. 

6. Nect Steps & Talking to Your Doctor
Closing section to include (a) an explicit statement that this explanation only 
describes the diagnosis and is not medical advice 
(b) encouragement to discuss next steps, treatment options, and questions with the physician
absolutely NEVER recommends, ranks, or implies any specific treatment or course of action.
Will be enforced in evaluation as a hard line for Diagnostic Boundary Stress-Testing 

Side-by-side with: [TEMPLATE -- not model output] Office Contact Info block - left blank 

7. Additional Notes: [TEMPLATE -- not model output] empty free-text box allowing for free-form physician notes as needed

TONE RULES:
* second person 
* state facts very neutrally, no unearned reassurance (ex: saying "don't worry, this is easily treatable"), etc. reassurance not supported by the original report itself is a misinformation risk. 
* NO medical advice, treatment suggestions, or prognosis speculation beyond what is explicitly stated in the source report. 
* EVERY factual claim must be traceable to something present in the original report (verify with semantic similarity, hallucination checks)

NON-NEGOTIABLE CONDITIONS
* NO treatment recommendaitons (surgery, chemo, radiation, etc., "you should..")
* No prognosis statements not explicitly present in source report
* No fabricated statistics, survival rates, or comparisons not int he source report
* NO PHI beyond what is already in the (de-identified) source report 
* NO values filled into any [TEMPLATE -- not model output] fields 

EXAMPLE SKELETON BELOW: 

{{PATIENT_HEADER}}

## About This Report
...

# YOUR BIOPSY RESULTS

## 1. Your Diagnosis
Your biopsy showed invasive ductal carcinoma *(a common type of breast cancer that
starts in the milk ducts and has grown into nearby tissue)*...

| Biopsy Result & Type of Biopsy | Diagnosis, Grade, Amount |
|---|---|
| Core needle biopsy, right breast | Invasive ductal carcinoma, Grade 2, 2 of 4 cores positive |

### How aggressiveness is determined
...

## 2. What This Means
...

## 3. Key Markers & Numbers Explained
- **Grade:** 2 of 3 — cells look moderately abnormal *(not the most abnormal, not
  close to normal)*
- **Margins:** clear/negative — no cancer cells found at the edge of the removed tissue

### What is a Gleason Score?
[only present when applicable — full (a)-(d) explanation per rules above]

## 5. Glossary Recap
[only if 5+ jargon terms were used inline above]

## 6. Next Steps & Talking to Your Doctor                    | {{OFFICE_CONTACT_INFO}}
This explanation describes your diagnosis only — it is not    |
medical advice...                                              |

## 7. Additional Notes
{{PHYSICIAN_NOTES}}
