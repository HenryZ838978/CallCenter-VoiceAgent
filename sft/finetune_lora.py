"""MiniCPM4.1-8B LoRA SFT 微调脚本.

用法:
  # 单卡
  python finetune_lora.py

  # 多卡 (DeepSpeed ZeRO-2)
  torchrun --nproc_per_node=2 finetune_lora.py

  # 自定义参数
  python finetune_lora.py --epochs 5 --lr 2e-4 --lora_r 16
"""
import os
import sys
import json
import argparse
import torch
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "..", "models", "MiniCPM4.1-8B")
DEFAULT_DATA = os.path.join(BASE_DIR, "outbound_sft_data.json")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=50)
    return p.parse_args()


def load_data(path, tokenizer, max_length):
    """Load conversations JSON and tokenize for causal LM training."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    input_ids_list = []
    labels_list = []

    for item in raw:
        convs = item["conversations"]
        text = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False,
        )
        encoded = tokenizer(
            text, truncation=True, max_length=max_length,
            padding=False, return_tensors=None,
        )
        input_ids = encoded["input_ids"]

        system_user_text = tokenizer.apply_chat_template(
            [c for c in convs if c["role"] != "assistant"][:1],
            tokenize=False, add_generation_prompt=True,
        )

        labels = list(input_ids)
        prompt_tokens = tokenizer(
            system_user_text, truncation=True, max_length=max_length,
            padding=False, return_tensors=None,
        )["input_ids"]

        for i in range(min(len(prompt_tokens) - 1, len(labels))):
            labels[i] = -100

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
    })
    return ds


def find_target_modules(model):
    """Auto-detect linear layers for LoRA."""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            last = name.split(".")[-1]
            if last not in ("lm_head",):
                targets.add(last)
    targets = sorted(targets)
    print(f"LoRA target modules: {targets}")
    return targets


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}")
    load_kwargs = dict(
        trust_remote_code=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=torch.bfloat16, **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, **load_kwargs,
        )
    model.enable_input_require_grads()

    target_modules = find_target_modules(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data from {args.data_path}")
    dataset = load_data(args.data_path, tokenizer, args.max_length)
    print(f"Dataset size: {len(dataset)} samples")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        optim="adamw_torch",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, max_length=args.max_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}/final")
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()
