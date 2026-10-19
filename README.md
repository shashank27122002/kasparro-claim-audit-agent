# KASPARRO Catalogue Claim Audit Agent

Python agent for discovering factual claim sentences in long speaker-product answers, retrieving catalogue facts via a model-requested tool call, and classifying claims as `supported`, `contradicted`, or `unknown`.

## What is implemented
- OpenAI Responses API with live model-requested `get_brand_facts` function tool.
- Exact source quotes and zero-based character offsets; invalid spans are dropped by local validation.
- Baseline `v1` and candidate `v2` prompt versions. The one targeted v2 change is a two-pass claim-completeness/span audit, explicitly covering negation, scope, historical prices, arithmetic, and unknowns.
- Deterministic exact-span extraction evaluation. Missed spans count as false negatives; extra/invented spans count as false positives. Classification metrics are calculated only on exact-span matches and include macro F1.
- Async execution bounded by a semaphore, exponential backoff with jitter, and selective retries on errors.
- FastAPI `/health` and `/run`; optional PostgreSQL persistence for runs, answers, and predictions.
- CSV/span/fact lookup tests.

## Input data
The three supplied CSVs are under `data/`. `answers.csv` has 40 development answers and 10 final-check answers. `development_gold.csv` is used only for development evaluation. Do not inspect/use private check labels; no check gold is supplied.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY. Set DATABASE_URL if PostgreSQL is available.
```
Never commit `.env` or API keys.

## Required experiment sequence
Run from repository root, with your own API key:
```powershell
$env:VERSION='v1'; $env:SPLIT='development'; python run_batch.py
```
Inspect the generated JSON and errors, then run candidate:
```powershell
$env:VERSION='v2'; $env:SPLIT='development'; python run_batch.py
```
Freeze both prompt versions/code before final-check predictions:
```powershell
$env:VERSION='v1'; $env:SPLIT='check'; python run_batch.py
$env:VERSION='v2'; $env:SPLIT='check'; python run_batch.py
```
This creates four actual live-run JSON artifacts. Commit those artifacts and console logs only after the calls complete. Check predictions are not scored locally because their labels are private.

## API
```powershell
uvicorn agent.main:app --reload
```
- `GET /health`
- `POST /run` with `{"version":"v1","split":"development"}` or `{"version":"v2","split":"check"}`

## PostgreSQL
Set `DATABASE_URL=postgresql+psycopg://...`. Tables `runs`, `answers`, `predictions` are created on API startup. The batch runner also attempts initialization and persists if configured.

## Langfuse
The prediction method is decorated with Langfuse `observe`; set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` to emit traces. Include trace links/screenshots after actual runs. Do not claim Langfuse evidence exists unless you configure and run it.

## Evaluation definitions
- Exact-span precision/recall/F1 use `(answer_id, start_char, end_char, quote)` identity.
- Classification accuracy and macro F1 use only exact-span matched claims, so extraction misses/extras cannot be hidden by classification performance.
- The final-check set is run only after v1/v2 are frozen; no check gold labels are included or expected.

## Known limitations / honest status
- This repository is a runnable implementation scaffold, not evidence of completed live runs. Four artifacts must be generated with a working API key before submission.
- Langfuse tracing is wired through the observation decorator but needs valid credentials and actual runs; no traces are fabricated.
- Model-requested facts tool is implemented. For long answers, tool filtering is broad and may return many facts; consider token budgeting if time permits.
- v2 is a prompt-only targeted improvement; do not claim it improves scores until measured on the identical development set.

## AI assistance disclosure
AI assistant used to scaffold files, implementation, and tests under time pressure. I reviewed the data schema, added local exact-span validation and deterministic evaluation tests, and must run the tests and inspect the four live outputs myself before submission.
