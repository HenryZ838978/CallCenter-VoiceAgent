#!/bin/bash
# VoxLabs Voice Agent — One-click startup
# Usage: bash start_all.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"

echo "=========================================="
echo "  VoxLabs Voice Agent — Starting"
echo "=========================================="

# 1. Check vLLM
echo "[1/3] Checking vLLM server on :8100..."
if curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
    echo "  vLLM already running"
else
    echo "  Starting vLLM on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$SCRIPT_DIR/models/MiniCPM4.1-8B-GPTQ" \
        --served-model-name MiniCPM4.1-8B-GPTQ \
        --trust-remote-code --dtype auto --quantization gptq_marlin \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --host 0.0.0.0 --port 8100 &
    echo "  Waiting for vLLM..."
    for i in $(seq 1 90); do
        if curl -s http://localhost:8100/v1/models > /dev/null 2>&1; then
            echo "  vLLM ready"
            break
        fi
        sleep 2
    done
fi

# 2. Start WebSocket server (GPU 2: ASR + TTS + RAG)
echo "[2/3] Starting Voice Agent server on :3000 (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 RAG_DEVICE=cuda:0 \
    $PYTHON "$SCRIPT_DIR/ws_server.py" &
WS_PID=$!

echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -sk https://localhost:3000/api/info > /dev/null 2>&1; then
        echo "  Server ready"
        break
    fi
    sleep 2
done

# 3. Start Cloudflare tunnel
echo "[3/3] Starting Cloudflare tunnel..."
"$SCRIPT_DIR/cloudflared" tunnel --url https://localhost:3000 --no-tls-verify 2>&1 &
TUNNEL_PID=$!

sleep 8
TUNNEL_URL=$(grep -o 'https://[^ ]*trycloudflare.com' /proc/$TUNNEL_PID/fd/2 2>/dev/null || echo "check logs")

echo "=========================================="
echo "  VoxLabs Voice Agent — READY"
echo ""
echo "  Local:   https://localhost:3000"
echo "  Public:  (check cloudflared output for URL)"
echo ""
echo "  APIs:"
echo "    GET  /api/info     — model info"
echo "    GET  /api/metrics  — latency stats"
echo "    GET  /api/rag/docs — knowledge base"
echo "    WS   /ws/voice     — duplex voice"
echo ""
echo "  PIDs: Server=$WS_PID Tunnel=$TUNNEL_PID"
echo "=========================================="

wait $WS_PID
