"""将 LoRA adapter 合并回底座并导出, 可选 GPTQ 量化.

用法:
  # 仅合并 (输出 BF16 全精度)
  python merge_and_quantize.py --adapter_path output/final --output_dir merged_model

  # 合并 + GPTQ 4bit 量化 (替换现有推理模型)
  python merge_and_quantize.py --adapter_path output/final --output_dir merged_gptq --quantize
"""
import os
import sys
import argparse
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(BASE_DIR, "..", "models", "MiniCPM4.1-8B")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=DEFAULT_BASE)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--quantize", action="store_true", help="Apply GPTQ 4-bit quantization")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group_size", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quantize:
        print(f"Quantizing to GPTQ {args.bits}-bit (group_size={args.group_size})...")
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            print("ERROR: auto-gptq not installed. Run: pip install auto-gptq")
            print("Saving unquantized model instead.")
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            return

        quant_config = BaseQuantizeConfig(
            bits=args.bits,
            group_size=args.group_size,
            desc_act=True,
            sym=True,
        )

        merged_tmp = args.output_dir + "_tmp_merged"
        model.save_pretrained(merged_tmp, safe_serialization=True)
        tokenizer.save_pretrained(merged_tmp)

        quant_model = AutoGPTQForCausalLM.from_pretrained(
            merged_tmp, quant_config, trust_remote_code=True,
        )

        calib_data = [
            "你好，请问是张先生吗？我这边是面壁智能的李明。",
            "是这样的，我们做了一套AI智能客服系统，能帮企业降低客服成本60%左右。",
            "好的，不好意思打扰您了。后续有需要可以搜面壁智能找到我们。",
            "费用跟您的坐席数量和通话量有关，一般是人工的三分之一到五分之一。",
            "没问题，我先给您发一份案例资料和报价方案，方便加个微信吗？",
        ]
        calib_encoded = [tokenizer(t, return_tensors="pt") for t in calib_data]
        quant_model.quantize(calib_encoded)
        quant_model.save_quantized(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)

        import shutil
        shutil.rmtree(merged_tmp, ignore_errors=True)
        print(f"GPTQ model saved to {args.output_dir}")
    else:
        print("Saving merged BF16 model...")
        model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
