#!/bin/bash
cd /cache/zhangjing/omni_agent
export CUDA_VISIBLE_DEVICES=1
export TTS_GPU_MEM=0.50
export TTS_DEVICE=cuda:0
export PORT=3001
nohup /cache/zhangjing/miniconda3/envs/voiceagent/bin/python ws_server_hybrid.py > /tmp/hybrid_v3.log 2>&1 &
echo "PID=$!"
