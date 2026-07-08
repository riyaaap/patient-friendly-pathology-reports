#!/bin/bash
# launch_vllm_server.sh
#
# Starts a vLLM OpenAI-compatible server for local Llama-3.1-8B (base) or a
# LoRA-adapted checkpoint, on a single L4 GPU, with:
#   - PyTorch-native (SDPA) attention backend to avoid FlashInfer JIT failures
#   - GPU memory utilization capped below your 90% hard limit
#
# Usage:
#   ./launch_vllm_server.sh                     # serve base model
#   ./launch_vllm_server.sh /path/to/lora_adapter  # serve base model + LoRA adapter

set -e

# --- Force PyTorch-native attention backend (avoids FlashInfer JIT compile errors) ---
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# --- Model config ---
BASE_MODEL="meta-llama/Llama-3.1-8B"
LORA_PATH="$1"   # optional first arg: path to a trained LoRA adapter directory

# --- Memory config ---
# vLLM's --gpu-memory-utilization is a FRACTION OF TOTAL GPU MEMORY (not free memory),
# and vLLM's own estimate can run slightly over in practice. Setting 0.85 here leaves
# real headroom under your 90% hard cap (24GB L4 -> ~20.4GB target vs ~21.6GB hard cap).
GPU_MEM_UTIL=0.85

# --- Context length ---
# Llama-3.1 supports up to 128k via RoPE scaling, but you don't need that much for
# pathology reports and it costs KV-cache memory. Cap it to something realistic for
# your longest cleaned reports (adjust after you've seen your word_count histogram --
# roughly 1.3 tokens per word as a rule of thumb, plus room for the generated report).
MAX_MODEL_LEN=8192

echo "Attention backend: $VLLM_ATTENTION_BACKEND"
echo "GPU memory utilization target: $GPU_MEM_UTIL"
echo "Max model length: $MAX_MODEL_LEN"

if [ -n "$LORA_PATH" ]; then
    echo "Serving $BASE_MODEL with LoRA adapter at $LORA_PATH"
    vllm serve "$BASE_MODEL" \
        --enable-lora \
        --lora-modules pathrep-lora="$LORA_PATH" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype float16 \
        --port 8000
else
    echo "Serving base model $BASE_MODEL (no LoRA)"
    vllm serve "$BASE_MODEL" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --dtype float16 \
        --port 8000
fi
