from __future__ import annotations
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "travel_examples.jsonl"

def load_examples(path: Path = DATA_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
