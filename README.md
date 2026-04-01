# NLP_Project

Local source-selection experiment pipeline with Ollama, analytics API, and Streamlit dashboard.

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

List local models:

```bash
uv run python -m app.cli.list_models
```

## CLI Workflow

Probe one model quickly:

```bash
uv run python -m app.cli.probe_model \
	--model qwen2.5:7b \
	--prompt "Return strict JSON with selected_article_id and reason."
```

Run a full experiment:

```bash
uv run python -m app.cli.run_experiments \
	--input data/mock_incidents.jsonl \
	--models-manifest config/models.example.yaml \
	--output-dir outputs \
	--conditions headlines_only headlines_with_sources sources_only headlines_with_manipulated_sources \
	--max-combinations 3 \
	--seed 42
```

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

## Fast End-to-End Path

1. Start Ollama: `ollama serve`
2. Start API: `uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000 --reload`
3. Start dashboard: `uv run streamlit run dashboard.py`
4. In dashboard sidebar, click **Ingest all runs in outputs**
5. View charts in **Analytics** tab
6. Run new jobs in **Run Models** tab

## Data Contracts

Core schemas are in `app/schemas/models.py`:
- `PreparedIncident`
- `ExperimentRequest`
- `ModelDecision`

These are the integration boundary between experiment execution, storage, and analytics.