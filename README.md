# Local Travel Guide Generator

A Streamlit application that creates interest-aware mini travel guides for a fictional city, **Luma Bay**. It includes a structured training dataset, GPT-2 LoRA fine-tuning, adapter merging, and merged-model inference.

> Educational prototype: Luma Bay is fictional. Recommendations are sample content, not verified real-world businesses, opening hours, prices, or safety advice.

## Features

- Structured JSONL dataset mapping destination, interests, duration, budget and local recommendations
- Fine-tuning of `gpt2` with Hugging Face Transformers and PEFT LoRA adapters
- LoRA adapter saving and merging into the base model
- Streamlit UI for destination, interests, trip length, budget and guide generation
- Feedback rating saved locally to a CSV file
- Tests for data loading, prompt construction and output formatting

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train LoRA and merge

Training downloads GPT-2 model files from Hugging Face and may take time. A GPU is recommended; CPU training is possible but slow.

```powershell
python -m app.core train --epochs 3
```

Optional arguments:

```powershell
python -m app.core train --model-name gpt2 --epochs 3 --batch-size 2 --max-length 256
```

The command saves:
- `artifacts/lora_adapter/` — trained LoRA adapter
- `artifacts/merged_model/` — merged model ready for inference

The training script uses the local JSONL examples and a causal language-modeling objective. It does not claim the small sample dataset will produce production-quality travel advice.

## Run the Streamlit app

```powershell
streamlit run app/main.py
```

Choose Luma Bay, select interests, trip duration and budget, then generate a guide. The app loads `artifacts/merged_model` if it exists. If you have not trained yet, the UI offers a clearly labelled dataset-based preview so you can inspect the app without model downloads.

## Dataset

Edit `data/travel_examples.jsonl`. Each line is a JSON object with:
- `destination`
- `interests` (list)
- `duration`
- `budget`
- `recommendations` (list)
- `guide` (training target text)

Use content you have permission to train on. For a new destination, add examples in the same schema.

## Model design

1. `prepare_training_texts` formats each JSONL example as an instruction and target guide.
2. GPT-2 is loaded with a causal language-modeling head.
3. PEFT injects trainable low-rank matrices into attention projection layers (`c_attn`).
4. The trainer optimizes adapter parameters while the base model remains frozen.
5. `merge_and_unload()` merges LoRA weights into the base model and saves a standalone inference checkpoint.
6. The Streamlit app loads the merged model and generates a guide from the user's selections.

## Evaluate

Run:

```powershell
pytest -v
```

Review generated guides for:
- Match between stated interests and recommended activities
- Practical variety and clear sequencing
- Budget and duration consistency
- Unsupported claims (especially hours, costs, transport and safety)

The built-in feedback control stores ratings and optional comments in `outputs/feedback.csv`.

## Limitations

- The included fictional dataset is deliberately small and is for demonstrating the workflow.
- GPT-2 may produce repetitive or inaccurate text, especially with a small fine-tuning set.
- Increase dataset size and manually validate all travel facts before using this for real customers.
- The demo preview is a deterministic retrieval/template preview, not an LLM generation result.
