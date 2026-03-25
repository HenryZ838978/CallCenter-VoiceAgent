#!/bin/bash
# MiniCPM4.1-8B LoRA 外呼坐席 SFT 训练启动脚本
#
# 用法:
#   bash run_train.sh          # 2卡默认参数
#   bash run_train.sh --epochs 5 --lr 2e-4  # 自定义参数
#
# 显存占用 (每卡): ~18-20GB
# 预估时间: 312条数据 x 3 epochs ≈ 10-15 min

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/cache/zhangjing/miniconda3/envs/voiceagent/bin/python"
ACCELERATE="/cache/zhangjing/miniconda3/envs/voiceagent/bin/accelerate"

export CUDA_VISIBLE_DEVICES=5,6

$ACCELERATE launch \
    --config_file "$SCRIPT_DIR/accelerate_config.yaml" \
    "$SCRIPT_DIR/finetune_lora.py" \
    --model_path "$SCRIPT_DIR/../models/MiniCPM4.1-8B" \
    --data_path "$SCRIPT_DIR/outbound_sft_data.json" \
    --output_dir "$SCRIPT_DIR/output" \
    --epochs 3 \
    --lr 1e-4 \
    --batch_size 2 \
    --grad_accum 8 \
    --max_length 2048 \
    --lora_r 16 \
    --lora_alpha 32 \
    "$@"

echo ""
echo "Training complete! LoRA adapter saved to: $SCRIPT_DIR/output/final"
echo ""
echo "Next steps:"
echo "  1. Merge & quantize: python merge_and_quantize.py --adapter_path output/final --output_dir merged_gptq --quantize"
echo "  2. Test: replace models/MiniCPM4.1-8B-GPTQ with merged_gptq, restart vLLM, re-run test_outbound_deviation.py"
