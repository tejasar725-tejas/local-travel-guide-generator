"""Merge trained LoRA weights into the GPT-2 base model for inference."""
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE = "gpt2"
ADAPTER = ROOT / "artifacts" / "lora-adapter"
OUTPUT = ROOT / "artifacts" / "merged-model"

def main():
    if not (ADAPTER / "adapter_config.json").exists():
        raise SystemExit("Adapter not found. Run: python scripts/train_lora.py")
    base = AutoModelForCausalLM.from_pretrained(BASE)
    tuned = PeftModel.from_pretrained(base, ADAPTER)
    merged = tuned.merge_and_unload()
    merged.save_pretrained(OUTPUT, safe_serialization=True)
    AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(OUTPUT)
    print(f"Merged model saved to {OUTPUT}")

if __name__ == "__main__": main()
