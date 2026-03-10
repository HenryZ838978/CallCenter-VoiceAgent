#!/bin/bash
# Start all LiveKit services for VoxLabs Voice Agent
# Usage: bash start_livekit.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"
export PATH="/cache/zhangjing/miniconda3/envs/voiceagent/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE="$SCRIPT_DIR/models"
export HF_HUB_OFFLINE=1
export FUNASR_DISABLE_UPDATE=1

export LIVEKIT_API_KEY="devkey"
export LIVEKIT_API_SECRET="secret"
export LIVEKIT_URL="ws://localhost:7880"

echo "=== VoxLabs LiveKit Voice Agent ==="

# 1. Check vLLM
if lsof -ti:8100 >/dev/null 2>&1; then
    echo "[OK] vLLM already running on :8100"
else
    echo "[STARTING] vLLM on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$SCRIPT_DIR/models/MiniCPM4.1-8B-GPTQ" \
        --served-model-name MiniCPM4.1-8B-GPTQ \
        --trust-remote-code --dtype auto --quantization gptq_marlin \
        --gpu-memory-utilization 0.45 --max-model-len 2048 --max-num-seqs 8 \
        --host 0.0.0.0 --port 8100 \
        > /tmp/vllm_lk.log 2>&1 &
    echo "  vLLM PID: $!"
    echo "  Waiting 60s for model load..."
    sleep 60
fi

# 2. Start LiveKit Server
if lsof -ti:7880 >/dev/null 2>&1; then
    echo "[OK] LiveKit Server already running on :7880"
else
    echo "[STARTING] LiveKit Server..."
    "$SCRIPT_DIR/livekit-server" --dev --bind 0.0.0.0 \
        > /tmp/livekit_server.log 2>&1 &
    echo "  LiveKit PID: $!"
    sleep 3
fi

# 3. Start Agent Worker
echo "[STARTING] Agent Worker on GPU 2..."
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 \
    TTS_GPU_UTIL=0.55 \
    $PYTHON "$SCRIPT_DIR/livekit_agent/run.py" dev \
    > /tmp/lk_agent.log 2>&1 &
echo "  Agent PID: $!"

# 4. Generate token
echo ""
echo "=== Connection Info ==="
echo "LiveKit URL: ws://$(hostname -I | awk '{print $1}'):7880"
echo ""
echo "Token:"
$PYTHON "$SCRIPT_DIR/livekit_agent/token_gen.py"
echo ""
echo "Frontend: file://$SCRIPT_DIR/livekit_agent/playground.html"
echo "Or serve via: python -m http.server 8080 -d $SCRIPT_DIR/livekit_agent"
echo ""
echo "=== All services started ==="
