from __future__ import annotations
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_DIR", "artifacts/merged-model"))

class LocalGuideModel:
    """Loads the LoRA-merged causal LM for local inference."""
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Merged model not found at {self.model_dir}. Run scripts/train_lora.py "
                "then scripts/merge_adapter.py, or use the transparent baseline mode."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.float32
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, destination: str, interests: list[str], days: int = 2) -> str:
        prompt = (
            "You are a thoughtful local travel guide. Give safe, specific, respectful "
            "recommendations and practical tips. Do not invent opening hours or prices.\n"
            f"Destination: {destination}\nInterests: {', '.join(interests)}\n"
            f"Trip length: {days} days\nMini-guide:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = self.model.generate(
            **inputs, max_new_tokens=240, do_sample=True, temperature=0.75,
            top_p=0.9, repetition_penalty=1.12,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip() or decoded
