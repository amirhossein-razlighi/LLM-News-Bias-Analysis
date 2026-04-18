# Quickstart

## Prerequisites

- Python 3.10
- uv package manager
- Ollama installed locally

## Setup

```bash
uv venv --python 3.10
uv sync
```

## Start Services

```bash
ollama serve
uv run streamlit run dashboard.py
```

Optional analytics API:

```bash
uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000 --reload
```

## Minimal Workflow

1. Prepare incidents:

```bash
uv run python -m app.cli.prepare_real_incidents --json-dir data/jsons --output data/real_incidents_all.jsonl
```

1. Run experiments:

```bash
uv run python -m app.cli.run_experiments --input data/real_incidents_all.jsonl --models-manifest configs/models.example.yaml --output-dir outputs
```

1. Generate report assets:

```bash
uv run python -m app.cli.generate_report_assets --outputs-dir outputs --assets-dir docs/figures
```

1. Generate LLM dashboard summary:

```bash
uv run python -m app.cli.generate_llm_dashboard_summary --outputs-dir outputs --model gemma4:latest --summary-json outputs/llm_dashboard_summary.json
```
