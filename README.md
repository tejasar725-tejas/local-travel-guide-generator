# Local Travel Guide Generator

<<<<<<< HEAD
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
=======
A Streamlit prototype for travel agencies. It personalizes a mini-guide from a destination, trip length and traveler interests. It includes (1) a transparent interest-ranked baseline that works immediately, and (2) a GPT-2 causal language model fine-tuned with PEFT LoRA and merged for local inference.

> **Honest scope:** the included destination, Aurelia Bay, and all 12 recommendations are fictional. The small dataset proves the workflow, not real-world travel accuracy. A trained model checkpoint is not committed because it is several hundred MB; train it locally using the steps below. Until then, use Baseline mode in the UI.

## Features
- Structured JSONL dataset with interest tags, recommendation, fit rationale and local tip.
- LoRA injection on GPT-2 attention projections (`c_attn`), supervised fine-tuning, adapter checkpointing.
- `merge_and_unload()` merges adapter weights into the base model.
- Streamlit UI with destination, interests, duration, Markdown download and a simple interest-coverage metric.
- Baseline mode works without downloading a language model.
- Pytest coverage and GitHub Actions CI.

## Requirements
- Python 3.12 and Git.
- Internet for the first GPT-2 download and Python package installation.
- For practical training speed, use a CUDA-capable GPU. CPU training is supported but slow and memory-intensive.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
pytest
streamlit run app/main.py
```

## Setup (macOS / Linux / WSL Ubuntu)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
pytest
streamlit run app/main.py
```
Open the local URL Streamlit prints (usually `http://localhost:8501`). Choose **Baseline (always available)** to try the app immediately.

## Train and merge the LoRA model
Run these commands from the repository root after installing requirements:
```bash
python scripts/train_lora.py --epochs 8
python scripts/merge_adapter.py
```
The first command downloads `gpt2` from Hugging Face, injects rank-8 LoRA adapters into GPT-2's attention projection, and trains on `data/travel_examples.jsonl`. The adapter is saved under `artifacts/lora-adapter/`. The second command loads the base GPT-2 plus adapter and merges the weights into `artifacts/merged-model/`.

Then select **Merged LoRA model** in the Streamlit sidebar. The model directory is ignored by Git to avoid committing large model binaries. `MODEL_DIR` in `.env` can point to a different merged checkpoint.

## Evaluation and sample output
```bash
python scripts/evaluate.py
pytest
```
The baseline evaluator checks whether requested interest words appear in the generated guide. It is a transparent lexical coverage proxy, not a human-quality or factuality score. Review generated suggestions manually before sharing with customers.

Example input: destination `Aurelia Bay`; interests `food`, `history`; duration `2 days`.

Example output excerpt:
```text
# Aurelia Bay: your 2-day local mini-guide
A practical starting point based on your interests: food, history.

## 1. Old Lantern Quarter — Lantern Lane
Wander a compact district of preserved merchant houses, tiled courtyards and small artisan studios...
Why it fits: You like heritage streets and handmade design.
Local tip: Ask a studio before photographing makers at work.

## 2. Spice Wharf Breakfast — Harbour Market
Try a tasting walk through family-run stalls serving saffron buns, coconut stew and seasonal fruit...
```
The complete deterministic sample is in `outputs/sample_guide.md`; evaluation cases and coverage are in `outputs/evaluation.txt`.

## Dataset provenance and limitations
The dataset is hand-authored for this project; it is not scraped and does not describe real establishments. Twelve records are intentionally small enough for a teaching-scale LoRA drill. Such a dataset can overfit, and GPT-2 may produce fluent but unsupported claims. Verify real venues, hours, costs, safety, transport, dietary and accessibility details from trusted local sources before publication. For production, replace the fictional dataset with licensed, verified destination content and add source citations, freshness dates, moderation and human review.

## Project layout
```text
app/                 Streamlit UI, generation and data helpers
data/                Fictional structured fine-tuning examples
scripts/             LoRA training, merging and evaluation
tests/               Unit and smoke tests
outputs/             Sample generated guide and evaluation run
.github/workflows/   CI test workflow
```

## GitHub submission checklist
Create an empty **public** repository named `local-travel-guide-generator` (do not initialize it with a README or .gitignore), then from this project folder:
```bash
git init -b main
git add .
git commit -m "Start local-travel-guide-generator"
git remote add origin https://github.com/YOUR-USERNAME/local-travel-guide-generator.git
git push -u origin main
```
Commit each completed milestone separately:
```bash
git add -A && git commit -m "Milestone 1: add fictional travel dataset" && git push
git add -A && git commit -m "Milestone 2: implement LoRA training" && git push
git add -A && git commit -m "Milestone 3: build Streamlit guide generator" && git push
git add -A && git commit -m "Milestone 4: evaluate recommendation relevance" && git push
git add -A && git commit -m "Milestone 5: merge LoRA adapter for inference" && git push
```
Never commit `.env` or model artifacts. Confirm `.env` is absent from the GitHub file list. GitHub Actions runs the tests on push and pull requests.
>>>>>>> d0166ed (Milestone 5: merge LoRA adapter for local inference and document setup)
