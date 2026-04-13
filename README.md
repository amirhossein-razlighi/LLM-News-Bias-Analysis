# NLP_Project

Local source-selection experiment pipeline with Ollama, analytics API, and Streamlit dashboard.

## Run And See Results

If you want the fastest path from setup to visible results, do this.

### 1) Install dependencies

```bash
uv venv --python 3.10
uv sync
uv add streamlit plotly
```

### 2) Pull the Ollama models you want to compare

Example:

```bash
ollama pull qwen2.5:7b
ollama pull qwen3:8b
ollama pull gemma3:4b
```

If you are using the current default manifest, also make sure every model in [configs/models.example.yaml](/Users/amirhossein/Documents/University/NLP/NLP_Project/configs/models.example.yaml) is pulled locally.

### 3) Start Ollama

In terminal 1:

```bash
ollama serve
```

### 4) Start the analytics API

In terminal 2:

```bash
uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Start the Streamlit dashboard

In terminal 3:

```bash
uv run streamlit run dashboard.py
```

Then open:
- http://localhost:8501

### 6) Prepare real data

In terminal 4:

```bash
uv run python -m app.cli.prepare_real_incidents \
  --json-dir data/jsons \
  --split-file data/splits/random/valid.tsv \
  --output data/real_incidents_random_valid.jsonl \
  --min-per-leaning 3 \
  --max-articles-per-leaning 8
```

### 7) Run the experiment

```bash
uv run python -m app.cli.run_experiments \
  --input data/real_incidents_random_valid.jsonl \
  --models-manifest configs/models.example.yaml \
  --output-dir outputs \
  --conditions headlines_only headlines_with_sources sources_only headlines_with_manipulated_sources \
  --max-combinations 3 \
  --seed 42
```

This creates a new folder like:

```text
outputs/run_YYYYMMDD_HHMMSS/
```

with:
- `experiment_requests.jsonl`
- `model_decisions.jsonl`
- `raw_outputs.jsonl`

### 8) Ingest the run and view the results

Use either option:

- In the Streamlit sidebar, click `Ingest all runs in outputs`
- Or call the API directly:

```bash
curl -X POST http://127.0.0.1:8000/ingest/runs \
  -H "Content-Type: application/json" \
  -d '{"outputs_dir":"outputs"}'
```

Then go to the `Analytics` tab in Streamlit to compare models, conditions, latency, parse success, outlet preferences, and leaning-selection behavior.

## Setup

### 1) Install uv (if needed)

```bash
brew install uv
```

### 2) Create env and install dependencies

```bash
uv venv --python 3.10
uv sync
```

### 3) Install dashboard dependencies (one-time)

```bash
uv add streamlit plotly
```

### 4) Run tests

```bash
uv run pytest
```

## Ollama Prerequisites

Start Ollama in a separate terminal:

```bash
ollama serve
```

Pull models you plan to use (example):

```bash
ollama pull qwen2.5:7b
```
Pull all the models inside `configs/models.example.yaml` if you want to use the current default manifest.

List local models:

```bash
uv run python -m app.cli.list_models
```

## Fast End-to-End Path

1. Start Ollama: `ollama serve`
2. Start API: `uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000 --reload`
3. Start dashboard: `uv run streamlit run dashboard.py`
4. In dashboard sidebar, click **Ingest all runs in outputs**
5. View charts in **Analytics** tab
6. Run new jobs in **Run Models** tab

## CLI Workflow

Probe one model quickly:

```bash
uv run python -m app.cli.probe_model \
	--model qwen2.5:7b \
	--prompt "Return strict JSON with selected_article_id and reason."
```

Run a full experiment:

```bash
uv run python -m app.cli.prepare_real_incidents \
	--json-dir data/jsons \
	--split-file data/splits/random/train.tsv \
	--output data/real_incidents_random_train.jsonl \
	--min-per-leaning 3 \
	--max-articles-per-leaning 8

uv run python -m app.cli.run_experiments \
	--input data/real_incidents_random_train.jsonl \
	--models-manifest configs/models.example.yaml \
	--output-dir outputs \
	--conditions headlines_only headlines_with_sources sources_only headlines_with_manipulated_sources \
	--max-combinations 3 \
	--seed 42
```

Notes for real data:
- `app.cli.prepare_real_incidents` groups `data/jsons/*.json` by topic and keeps topics that have left/center/right coverage.
- Use `data/splits/random/*.tsv` or `data/splits/media/*.tsv` with `--split-file` if you want to stay inside one dataset split.
- `--max-articles-per-leaning` keeps each incident bounded so the runner does not build an enormous search space for large topics like `elections`.

Candidate order randomization:
- Enabled by default for position-bias control.
- Disable with `--no-shuffle-candidates`.

Benchmark latency across models:

```bash
uv run python -m app.cli.benchmark_models \
	--models qwen2.5:7b \
	--prompt "Return strict JSON only." \
	--rounds 3
```

## Output Files

Each run writes:
- `outputs/run_YYYYMMDD_HHMMSS/experiment_requests.jsonl`
- `outputs/run_YYYYMMDD_HHMMSS/model_decisions.jsonl`
- `outputs/run_YYYYMMDD_HHMMSS/raw_outputs.jsonl`

`experiment_requests.jsonl` includes `candidate_order` for position-bias analysis.

## Analytics API

API module is now under `app/api/engine_analytics.py`.

Run API server:

```bash
uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000 --reload
```

Browser docs:
- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

### API Endpoints

- `POST /ingest`
	- Ingests already-normalized analytics rows (`ExperimentResult` schema).

- `POST /ingest/run`
	- Ingests one run directory containing:
	- `model_decisions.jsonl`
	- `experiment_requests.jsonl`
	- Automatically maps pipeline fields to analytics fields.

- `POST /ingest/runs`
	- Bulk-ingests all `run_*` folders under an outputs directory.
	- Default `outputs_dir` is `outputs`.

- `GET /metrics/summary`
	- Returns aggregate metrics over ingested data.
	- Optional query parameter: `model`.

- `GET /metrics/inter-model`
	- Returns metrics split by model.

- `GET /metrics/runs`
	- Lists ingested run IDs.

- `GET /export/csv`
	- Returns path to the analytics CSV database.

## Streamlit Dashboard

Run dashboard:

```bash
uv run streamlit run dashboard.py
```

Open in browser:
- http://localhost:8501

### Dashboard Features

- **Analytics tab**
	- Summary cards and distributions.
	- Inter-model comparison table and charts.
	- Run ID listing.

- **Run Models tab**
	- Probe Model: single prompt call using project Ollama client.
	- Batch Experiment: full run using existing experiment builders, parser, and Pydantic schemas.
	- Optional auto-ingest after batch completion.

- **Sidebar ingest tools**
	- Bulk ingest all runs from `outputs` (default first-try path).
	- Single-run ingest by explicit run directory.

## Data Contracts

Core schemas are in `app/schemas/models.py`:
- `PreparedIncident`
- `ExperimentRequest`
- `ModelDecision`

These are the integration boundary between experiment execution, storage, and analytics.
