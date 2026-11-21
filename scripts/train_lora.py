"""Fine-tune GPT-2 with a small supervised travel-guide dataset using PEFT LoRA.

Run from repository root:
  python scripts/train_lora.py --epochs 8
This downloads GPT-2 on first run. CPU is supported but slow; a CUDA GPU is recommended.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "travel_examples.jsonl"

class TravelDataset(Dataset):
    def __init__(self, tokenizer, max_length=256):
        self.rows = []
        for line in DATA.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = f"Destination: Aurelia Bay\nInterests: {row['tags']}\nMini-guide recommendation:"
            full = prompt + " " + row["recommendation"] + " Why it fits: " + row["why_it_fits"] + " Local tip: " + row["tip"] + tokenizer.eos_token
            enc = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
            labels = enc["input_ids"].copy()
            prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"])
            labels[:prompt_len] = [-100] * min(prompt_len, len(labels))
            labels = labels[:max_length]
            self.rows.append({"input_ids": torch.tensor(enc["input_ids"]),
                              "attention_mask": torch.tensor(enc["attention_mask"]),
                              "labels": torch.tensor(labels)})
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "lora-adapter"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.eos_token_id
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
                        target_modules=["c_attn"], fan_in_fan_out=True)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    training = TrainingArguments(
        output_dir=str(ROOT / "artifacts" / "training-runs"),
        num_train_epochs=args.epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=2, learning_rate=2e-4,
        logging_steps=2, save_strategy="no", report_to="none",
        fp16=torch.cuda.is_available(), use_cpu=not torch.cuda.is_available()
    )
    trainer = Trainer(model=model, args=training, train_dataset=TravelDataset(tokenizer),
                      tokenizer=tokenizer)
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")

if __name__ == "__main__": main()
