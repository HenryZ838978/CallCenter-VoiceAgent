"""用 gptqmodel 将 SFT 合并模型量化为 GPTQ 4-bit.

使用与官方 MiniCPM4.1-8B-GPTQ 完全相同的配置:
  bits=4, group_size=128, desc_act=True, sym=True, true_sequential=True

校准数据使用 SFT 训练数据，最大程度保留微调效果。
"""
import json
import os
import shutil

import torch
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

MERGED_PATH = os.path.join(os.path.dirname(__file__), "output", "merged")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "gptq_4bit")
SFT_DATA_PATH = os.path.join(os.path.dirname(__file__), "outbound_sft_data.json")


def build_calibration_data(tokenizer, n_samples=128):
    """从 SFT 训练数据构建校准集 — 让量化过程看到微调时的分布."""
    with open(SFT_DATA_PATH, "r", encoding="utf-8") as f:
        sft_data = json.load(f)

    texts = []
    for sample in sft_data[:n_samples]:
        turns = sample["conversations"]
        formatted = tokenizer.apply_chat_template(turns, tokenize=False)
        texts.append(formatted)

    return texts


def main():
    print(f"Loading tokenizer from {MERGED_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)

    print("Building calibration data from SFT training set...")
    calib_data = build_calibration_data(tokenizer, n_samples=128)
    print(f"  {len(calib_data)} calibration samples")

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        true_sequential=True,
        damp_percent=0.01,
    )

    print(f"Loading merged BF16 model from {MERGED_PATH}")
    model = GPTQModel.load(
        MERGED_PATH,
        quant_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("Quantizing to GPTQ 4-bit (this takes ~15-30 min)...")
    model.quantize(calib_data)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"Saving quantized model to {OUTPUT_PATH}")
    model.save(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    for f in ["configuration_minicpm.py", "modeling_minicpm.py"]:
        src = os.path.join(MERGED_PATH, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_PATH, f))

    total_size = sum(
        os.path.getsize(os.path.join(OUTPUT_PATH, f))
        for f in os.listdir(OUTPUT_PATH)
    ) / 1e9
    print(f"\nDone! GPTQ model size: {total_size:.1f} GB")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
