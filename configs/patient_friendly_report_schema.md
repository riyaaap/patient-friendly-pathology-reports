Patient-Friendly Pathology Report — Output Schema
-> to define criteria for successful outputs of the model 

This schema defines the reuired structure, tone, and content rules for model output. 
, to use as: 1) a generation prompt template for teacher-LLM silver labels, 
2) fine-tuning target format, and 
3) LLM-as-a-judge scoring rubric. 

TARGET READING LEVEL
8-10th grade (from Flesch-Kincaid Grade Level) -- preferably 8th grade. 
This is the standard commonly cited for patient health materials (ex., from AMA/NIH health literacy guides)
Should be enforced in evaluation w/ Flesch-Kincaid + Gunning Fog tests, + 
also checked during teacher-LLM generation. 

REQUIRED SECTION STRUCTURE (fixed headers in ~ this order) 

Top of Page: Patient Name, DOB, ID, Doctor Name, Biopsy date, Report date 
* for now, keep this BLANK/with only placeholders, since all data will be de-identified
* keep this for the medical system to fill in later on 

About this report: Why we do biopsies (screening for what, and using which tests, help decide whether to have more tests, and degree of influence on diagnosing specific disease). what it contains, what tissue was removed/studied, what it was being checked for. 

YOUR BIOPSY RESULTS

1. Your Diagnosis - 
Plain-language, 2-4 sentence summary of what was found. State diagnosis name once in full, 
followed by simple explanation/restatement in everyday language. 

Small side by side table, containing: Biopsy result and type of Biopsy 
Diagnosis, Grade, Amount (# of samples with cancer), 
[Risk Category ONLY if explicitly stated in the source report, using a formal category (ex: the NCCN low/int/high for prostate). Otherwise, DO NOT INCLUDE, as would be a hallucination] 
Later on, can include image of the scan labeled, with legend at the side. 

How aggressiveness is determined: provide context on how the scans are evaluated. 

2. What This Means
Short paragraph(s) contextualizing the diagnosis, in terms of what condition/cancer it is, in the simplest accurate terms. Include general disease context, not patient-specific details. 

3. Key Markers & Numbers Explained
Bullet-point or brief breakdown of clinically significant markers, scores, or measurements present in the original report (grade, stage, tumor size, margins, lymph node status, biomarkers like ER/PR/HER2, KI-67, molecular markers, etc.)
Each bullet can be in the following format: 
    **[Marker name]:** [value from report]; [plain-language meaning]

Rule for Gleason scores specifically (ex: used heavily in prostate pathology reports, may be commonly confusing to patients)..
Section title: What is a Gleason Score? 
(a) explain WHAT a Gleason score is: grading system for how abnormal cancer cells appear under microscope
(b) HOW it is reported: two numbers added together, respectively reflecting how much of the two most common patterns in the sample resembles cancer,
and with low numbers reflecting closer to normal tissue and high numbers meaning more abnormal
(c) how the gleason sum obtained maps to the Grade group (1-5) system, especially since appear often on newer reports. 
(d) perhaps show a small table correlating low/medium/high risk categories to Gleason scores, as well as relevant values (ex: PSA (ng/mL), clinical T stage, etc.) with the result specific category highlighted. 
[Optional]: To learn more about [specific disease], visit: [links to sites for more info on prostate cancer or gleason scores, etc. as relevant] examples: (www.cancer.ca/prostate and www.mypathologyreport.ca/prostate)

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
And, Office Contact info side-by-side with the Next Steps 

7. Additional Notes: empty box allowing for free-form physician notes as needed

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


EXAMPLE SKELETON BELOW: 

## Your Diagnosis
Your biopsy showed invasive ductal carcinoma *(a common type of breast cancer that
starts in the milk ducts and has grown into nearby tissue)*...

## What This Means
...

## Key Markers & Numbers Explained
- **Grade:** 2 of 3 — cells look moderately abnormal *(not the most abnormal, not
  close to normal)*
- **Gleason Score:** 3+4=7 — [full explanation per standing rule above]
- **Margins:** clear/negative — no cancer cells found at the edge of the removed tissue

## Talking to Your Doctor
This explanation describes your diagnosis only — it is not medical advice... 
