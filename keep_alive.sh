#!/bin/bash
# keep_alive.sh — 持久化守护脚本，确保 Voice Agent + vLLM + SSH 隧道始终在线
# 用法: nohup bash keep_alive.sh > /tmp/keep_alive.log 2>&1 &
# 客户演示前启动，Ctrl+C 或 kill 停止

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"
export PATH="/cache/zhangjing/miniconda3/envs/voiceagent/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE="$SCRIPT_DIR/models"
export HF_HUB_OFFLINE=1
export FUNASR_DISABLE_UPDATE=1

VPS_IP="82.156.207.59"
CHECK_INTERVAL=30

log() { echo "$(date '+%H:%M:%S') [KEEPALIVE] $*"; }

start_vllm() {
    log "Starting vLLM on GPU 1..."
    setsid bash -c "CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
      --model $SCRIPT_DIR/models/MiniCPM4.1-8B-GPTQ \
      --served-model-name MiniCPM4.1-8B-GPTQ \
      --trust-remote-code --dtype auto --quantization gptq_marlin \
      --gpu-memory-utilization 0.40 --max-model-len 2048 --max-num-seqs 4 \
      --enforce-eager --host 0.0.0.0 --port 8100 > /tmp/vllm_prod.log 2>&1" &
    sleep 70
}

start_voice() {
    log "Starting Voice Agent on GPU 2..."
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES=2 ASR_DEVICE=cuda:0 TTS_DEVICE=cuda:0 RAG_DEVICE=cpu \
      USE_FIRERED_ASR=1 USE_MOONSHINE_ASR=1 USE_SMART_TURN=1 \
      TTS_GPU_UTIL=0.45 \
      $PYTHON ws_server.py > /tmp/voiceagent_prod.log 2>&1 &
    sleep 110
}

start_tunnel() {
    log "Starting SSH tunnel to VPS..."
    ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -R 3000:localhost:3000 root@$VPS_IP -N -f 2>/dev/null
    sleep 2
}

# Initial start
log "=== Keep-alive daemon starting ==="

while true; do
    # Check Voice Agent
    if ! lsof -ti:3000 >/dev/null 2>&1; then
        log "Voice Agent down! Restarting..."
        # Kill orphan GPU processes first
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | \
          while read pid; do
            lsof -ti:8100 2>/dev/null | grep -q "^${pid}$" || [ "$pid" = "1092885" ] || kill -9 $pid 2>/dev/null
          done
        sleep 3
        start_voice
    fi

    # Check vLLM
    if ! lsof -ti:8100 >/dev/null 2>&1; then
        log "vLLM down! Restarting..."
        start_vllm
    fi

    # Check SSH tunnel
    if ! ps aux | grep "ssh.*${VPS_IP}.*3000" | grep -v grep >/dev/null 2>&1; then
        log "SSH tunnel down! Reconnecting..."
        start_tunnel
    fi

    sleep $CHECK_INTERVAL
done
