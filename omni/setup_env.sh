#!/bin/bash
set -e
ENV_NAME="omni_agent"
CONDA_BASE="/cache/zhangjing/miniconda3"
MODEL_ID="openbmb/MiniCPM-o-4_5"
MODEL_DIR="/cache/zhangjing/omni_agent/models/MiniCPM-o-4_5"

echo "=== Omni Agent Environment Setup ==="

if [ ! -d "$CONDA_BASE/envs/$ENV_NAME" ]; then
    echo "[1/4] Creating conda env..."
    $CONDA_BASE/bin/conda create -n $ENV_NAME python=3.10 -y
else
    echo "[1/4] Conda env exists, skipping."
fi

PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

echo "[2/4] Installing PyTorch..."
$PIP install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "[3/4] Installing dependencies..."
$PIP install "transformers==4.51.0" "accelerate>=0.26.0" "minicpmo-utils[all]>=1.0.5"
$PIP install bitsandbytes librosa soundfile
$PIP install fastapi "uvicorn[standard]" websockets numpy

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "[4/4] Downloading MiniCPM-o-4.5..."
    $PIP install huggingface_hub
    $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')
print('Download complete.')
"
else
    echo "[4/4] Model already exists."
fi

echo "=== Setup Complete ==="
