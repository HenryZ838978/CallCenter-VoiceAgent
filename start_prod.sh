#!/bin/bash
# start_prod.sh — 生产启动脚本 (3卡分离 + 持久化守护 + SSH 隧道)
#
# GPU 布局:
#   GPU 1: vLLM (MiniCPM4.1-8B-GPTQ)
#   GPU 2: TTS (VoxCPM 1.5 nanovllm) — 独占，无 VRAM 竞争
#   GPU 3: ASR (FireRedASR2-AED FP16) — 独占
#
# 用法: bash start_prod.sh
# 停止: bash start_prod.sh stop

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"
export PATH="/cache/zhangjing/miniconda3/envs/voiceagent/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE="$SCRIPT_DIR/models"
export HF_HUB_OFFLINE=1
export FUNASR_DISABLE_UPDATE=1

VPS_IP="82.156.207.59"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [PROD] $*"; }

if [ "$1" = "stop" ]; then
    log "Stopping all services..."
    kill $(lsof -ti:3000) 2>/dev/null
    kill $(lsof -ti:8100) 2>/dev/null
    ps aux | grep "keep_prod\|ssh.*${VPS_IP}" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null
    log "All stopped"
    exit 0
fi

log "=== Starting Production Services (3-GPU) ==="

# 1. Voice Agent (GPU 2=TTS, GPU 3=ASR)
log "Starting Voice Agent (GPU 2+3)..."
CUDA_VISIBLE_DEVICES=2,3 ASR_DEVICE=cuda:1 TTS_DEVICE=cuda:0 RAG_DEVICE=cpu \
  USE_FIRERED_ASR=1 USE_MOONSHINE_ASR=1 USE_SMART_TURN=1 \
  TTS_GPU_UTIL=0.55 \
  $PYTHON "$SCRIPT_DIR/ws_server.py" > /tmp/voiceagent_prod.log 2>&1 &
VOICE_PID=$!
log "Voice Agent PID: $VOICE_PID (waiting for model load...)"

# Wait for voice agent to finish loading (TTS is the slowest)
for i in $(seq 1 60); do
    if curl -s http://localhost:3000/api/info > /dev/null 2>&1; then
        log "Voice Agent ready!"
        break
    fi
    sleep 3
done

# 2. vLLM (GPU 1) — start AFTER voice agent to avoid CUDA conflict
log "Starting vLLM (GPU 1)..."
setsid bash -c "CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
  --model $SCRIPT_DIR/models/MiniCPM4.1-8B-GPTQ \
  --served-model-name MiniCPM4.1-8B-GPTQ \
  --trust-remote-code --dtype auto --quantization gptq_marlin \
  --gpu-memory-utilization 0.40 --max-model-len 2048 --max-num-seqs 4 \
  --enforce-eager --host 0.0.0.0 --port 8100 > /tmp/vllm_prod.log 2>&1" &
log "vLLM starting (waiting 70s for model load...)"
sleep 70

if lsof -ti:8100 > /dev/null 2>&1; then
    log "vLLM ready!"
else
    log "WARNING: vLLM may still be loading"
fi

# 3. SSH tunnel to VPS
log "Establishing SSH tunnel to VPS..."
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 3000:localhost:3000 root@$VPS_IP -N -f 2>/dev/null
log "SSH tunnel established"

# 4. Start keepalive daemon
log "Starting keepalive daemon..."
nohup bash -c "
while true; do
    # Check tunnel
    if ! ps aux | grep 'ssh.*${VPS_IP}' | grep -v grep > /dev/null 2>&1; then
        echo \"\$(date) Tunnel down, reconnecting...\"
        ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 3000:localhost:3000 root@${VPS_IP} -N -f 2>/dev/null
    fi
    # Check vLLM
    if ! lsof -ti:8100 > /dev/null 2>&1; then
        echo \"\$(date) vLLM down, restarting...\"
        setsid bash -c 'CUDA_VISIBLE_DEVICES=1 $PYTHON -m vllm.entrypoints.openai.api_server \
          --model $SCRIPT_DIR/models/MiniCPM4.1-8B-GPTQ --served-model-name MiniCPM4.1-8B-GPTQ \
          --trust-remote-code --dtype auto --quantization gptq_marlin \
          --gpu-memory-utilization 0.40 --max-model-len 2048 --max-num-seqs 4 \
          --enforce-eager --host 0.0.0.0 --port 8100 > /tmp/vllm_prod.log 2>&1' &
    fi
    sleep 15
done" > /tmp/keep_prod.log 2>&1 &
log "Keepalive PID: $!"

log ""
log "=== All Services Started ==="
log "Voice Agent: http://localhost:3000 (GPU 2+3)"
log "vLLM:        http://localhost:8100 (GPU 1)"
log "Public:      https://hzai.online"
log ""
log "GPU Layout:"
log "  GPU 1: vLLM MiniCPM4.1-8B-GPTQ (~10GB)"
log "  GPU 2: VoxCPM TTS nanovllm (~13.5GB)"
log "  GPU 3: FireRedASR2-AED FP16 (~2.2GB)"
log ""
log "To stop: bash $0 stop"
