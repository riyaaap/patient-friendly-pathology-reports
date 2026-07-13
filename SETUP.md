Documentation on Environment Setups + design choices for future reference / reproducibility... 

This project design currently uses *two separate* conda environments: 
* pathrep-train -- for fine-tuning (transformers + PEFT/LoRA training)
* pathrep-serve -- for model hosting/inference via vLLM 

why using 2 instead of 1: had some issues with vLLM requiring very specific versions of torch, transformers, and numpy. 
installing training-related packages (like autoawq, peft, etc) in the same environment as vLLM caused silent version drift of certain packages multiple times, that broke the serving pipeline + the other way around too --> so chose to keep them isolated such that dependency changes on one side don't affect the other side 

Hardware & GPU Allocation
* server using NVIDIA L4 GPUs, 24GB each 
* This project is using 2 physical GPUs, numbered 1 and 2 on server
* have a hard cap on 90% GPU memory utilization for any single GPU at all times... being enforced via --gpu-memory-utilization in vLLM and monitored manually during training

Required environment variables (both envs)
	export CUDA_DEVICE_ORDER=PCI_BUS_ID 
	export CUDA_VISIBLE_DEVICES=1,2

note: CUDA_DEVICE_ORDER=PCI_BUS_ID is important since nvidia-smi index ordering and CUDA's internal numbering may differ, o/w would risk accidental use of wrong physical GPU 

these are persisted via conda activation hooks in both environments: 
* $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh (sets vars)
* $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh (unsets vars)

For any INDIVIDUAL process, if want to further restrict to a single physical GPU.. 
	CUDA_VISIBLE_DEVICES=1 python train_lora.py # physical GPU 1 only 
	CUDA_VISIBLE_DEVICES=2 vllm serve ... # physical GPU 2 only 

Code should always reference logical cuda:0, never hardcode physical indices — CUDA_VISIBLE_DEVICES handles the remapping

Environment: pathrep-serve (for vLLM hosting)
Package Version Notes

Python 3.10.20

torch 2.4.0+cu121 Must stay pinned — multiple installs attempted to silently upgrade this

vllm 0.6.3.post1 0.6.2 had a broken rope_scaling config parser

for quantized Qwen2 models; upgraded to 0.6.3.post1 to fix

transformers 4.45.2 Installing vllm==0.6.3.post1 initially pulled in transformers 5.13.1 (breaking change); force-reinstalled down to 4.45.2

numpy 1.26.4 Must stay under 2.0 — outlines (a vLLM dependency) breaks on numpy 2.x's reorganized internals

autoawq 0.2.9 Deprecated upstream (functionality moving into vLLM/llm-compressor) but still functional; used for AWQ-quantized teacher model

pyairports 0.0.1 (broken stub — see below) **Manual fix required**, see next section

*Required attention backend setting*: export VLLM_ATTENTION_BACKEND=XFORMERS
notes:
* Do NOT use TORCH_SDPA: this vLLM build asserts SDPA is CPU-only and crashes with AssertionError: Torch SDPA backend is only used for the CPU device.
* GPU.XFORMERS is precompiled (no JIT step) and avoids FlashInfer JIT-compilation issues from before 
* If XFORMERS ever causes issues, fallback is --enforce-eager appended to the vllm serve command (disables CUDA graph capture, slower but avoids backend/compile issues entirely)

other known issue: broken `pyairports` package requires manual stub
	outlines (is hard dependency of vLLM, used for guided/structured decoding) unconditionally imports pyairports.airports.AIRPORT_LIST at import time — even though this project never uses airport-code-constrained generation. 
	The published pyairports==0.0.1 package on PyPI is a broken/mislabeled stub (its top_level.txt lists sample/tests, not pyairports — it never contained the actual module).

to fix: (may need to be reapplied if building this env from scratch): 
	mkdir -p $CONDA_PREFIX/lib/python3.10/site-packages/pyairports 
	echo 'AIRPORT_LIST = []' > $CONDA_PREFIX/lib/python3.10/site-packages/pyairports/airports.py 
	touch $CONDA_PREFIX/lib/python3.10/site-packages/pyairports/__init__.py

an empty AIRPORT_LIST is safe.. this project never uses

Install steps (from scratch): 

conda create -n pathrep-serve python=3.10 -y conda activate pathrep-serve pip install --no-cache-dir --index-url https://pypi.org/simple vllm==0.6.3.post1 pip install --no-cache-dir --force-reinstall transformers==4.45.2 pip install --no-cache-dir "numpy<2.0" pip install --no-cache-dir autoawq # Manual pyairports stub fix (see above) — required every rebuild mkdir -p $CONDA_PREFIX/lib/python3.10/site-packages/pyairports echo 'AIRPORT_LIST = []' > $CONDA_PREFIX/lib/python3.10/site-packages/pyairports/airports.py touch $CONDA_PREFIX/lib/python3.10/site-packages/pyairports/__init__.py # Set env vars (see GPU Allocation section + attention backend above)

Verification checklist


python -c " import torch, transformers, vllm, numpy print('torch:', torch.__version__) # expect 2.4.0+cu121 print('transformers:', transformers.__version__) # expect 4.45.2 print('vllm:', vllm.__version__) # expect 0.6.3.post1 (prints oddly but is correct) print('numpy:', numpy.__version__) # expect 1.26.4

print('CUDA available:', torch.cuda.is_available()) print('Visible GPUs:', torch.cuda.device_count()) "

LAUNCHING THE TEACHER MODEL SERVER: 
CUDA_VISIBLE_DEVICES=2 vllm serve ./models/qwen2.5-14b-teacher \ --quantization awq \ --gpu-memory-utilization 0.85 \ --max-model-len 8192 \ --port 8001

Test with:

curl http://localhost:8001/v1/chat/completions \ -H "Content-Type: application/json" \ -d '{"model": "./models/qwen2.5-14b-teacher", "messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 30}'

Environment: `pathrep-train` (LoRA fine-tuning)

Package versions


Package Version

Python 3.10

torch 2.4.0 (cu121)

transformers 4.44.2

accelerate 0.34.2

peft 0.12.0

datasets, bitsandbytes, huggingface_hub,

sentencepiece latest at install time

pandas, matplotlib, textstat, rouge-score,

bert-score, sentence-transformers, evaluate latest at install time


Install steps (from scratch)


conda create -n pathrep-train python=3.10 -y conda activate pathrep-train pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 pip install transformers==4.44.2 accelerate==0.34.2 peft==0.12.0 pip install datasets bitsandbytes pip install huggingface_hub sentencepiece pip install pandas matplotlib textstat rouge-score bert-score sentence-transformers evaluate huggingface-cli login # required for gated meta-llama/Llama-3.1-8B repo


Same CUDA_DEVICE_ORDER / CUDA_VISIBLE_DEVICES=1,2 env vars applied via conda activation hooks as described in the GPU Allocation section above.


Models Used


Role Model Notes

Base/student model `meta-llama/Llama-3.1-8B` Non-instruct base model; gated repo, requires HF license acceptance + token auth

Teacher model (silver-label

generation) `Qwen/Qwen2.5-14B-Instruct-AWQ` 4-bit AWQ quantized to fit comfortably on a single L4 under the 90% memory cap


RECOMMENDATION:
if upgrading to latest versions of all later, do in a new separate env and valiate separately instead of upgrading in place bc have many breaking changes


Files in This Repo
· environment-serve.yml, requirements-serve-full.txt — full exported state of pathrep-serve

· environment-train.yml, requirements-train-full.txt — full exported state of pathrep-train

· requirements-serve.txt, requirements-train.txt — minimal pinned core dependencies

· scripts/launch_vllm_server.sh — vLLM launch script with correct backend/memory flags baked in

· scripts/clean_and_filter_reports.py — data cleaning pipeline (TCGA-Reports / Mendeley source CSV)

· scripts/inspect_short_reports.py — diagnostic tool for word-count distribution review

· scripts/generate_silver_labels.py — teacher-LLM silver-label generation script

· configs/patient_friendly_output_schema.md — locked output schema/rubric for patient-friendly report generation

