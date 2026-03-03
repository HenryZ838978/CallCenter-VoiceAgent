#!/bin/bash
# Start vLLM server for MiniCPM4.1-8B-GPTQ on a specific GPU
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/MiniCPM4.1-8B-GPTQ"
GPU_ID="${LLM_GPU:-1}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "Starting vLLM server on GPU ${GPU_ID}..."
echo "Model: ${MODEL_DIR}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --served-model-name "MiniCPM4.1-8B-GPTQ" \
    --trust-remote-code \
    --dtype auto \
    --quantization gptq_marlin \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
