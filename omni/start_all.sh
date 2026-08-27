#!/bin/bash
# Omni Voice Agent — one-click start (server + tunnel)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_PYTHON="/cache/zhangjing/miniconda3/envs/omni_agent/bin/python"
GPU="${OMNI_GPU:-2}"
PORT="${PORT:-3001}"

echo "=== Omni Voice Agent — MiniCPM-o 4.5 ==="
echo "GPU:  $GPU"
echo "Port: $PORT"
echo ""

# Kill any existing processes on this port
lsof -ti:$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

# 1. Start WebSocket server
echo "[1/2] Starting WebSocket server (GPU $GPU, port $PORT)..."
CUDA_VISIBLE_DEVICES=$GPU OMNI_DEVICE=cuda:0 LOAD_IN_4BIT=0 PORT=$PORT \
  nohup $CONDA_PYTHON ws_server.py > /tmp/omni_agent.log 2>&1 &
SERVER_PID=$!
echo "  Server PID: $SERVER_PID (log: /tmp/omni_agent.log)"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 120); do
    if curl -sk https://localhost:$PORT/api/info > /dev/null 2>&1; then
        echo "  Server ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "  ERROR: Server crashed. Check /tmp/omni_agent.log"
        tail -20 /tmp/omni_agent.log
        exit 1
    fi
    sleep 2
done

# 2. Start Cloudflare tunnel
echo "[2/2] Starting Cloudflare tunnel..."
nohup ./cloudflared tunnel --url https://localhost:$PORT --no-tls-verify > /tmp/omni_tunnel.log 2>&1 &
TUNNEL_PID=$!
echo "  Tunnel PID: $TUNNEL_PID"

sleep 5
TUNNEL_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/omni_tunnel.log | head -1)

echo ""
echo "============================================="
echo "  Omni Voice Agent is LIVE!"
echo ""
echo "  Local:  https://localhost:$PORT"
echo "  Public: $TUNNEL_URL"
echo ""
echo "  Server PID: $SERVER_PID"
echo "  Tunnel PID: $TUNNEL_PID"
echo "  Log: tail -f /tmp/omni_agent.log"
echo "============================================="
