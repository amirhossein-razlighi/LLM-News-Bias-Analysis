# NLP_Project

## Setup (uv + Python 3.10)

1. Install `uv` if needed:

```bash
brew install uv
```

2. Create and sync a Python 3.10 environment:

```bash
uv venv --python 3.10
uv sync
```

3. Run tests:

```bash
uv run pytest
```

## Quick Commands

List local Ollama models:

```bash
uv run python -m app.cli.list_models
```

Probe one model with one prompt:

```bash
uv run python -m app.cli.probe_model \
	--model llama3.1:8b \
	--prompt "Return JSON with selected_article_id and reason."
```

Run the full batch experiment:

```bash
uv run python -m app.cli.run_experiments \
	--input data/mock_incidents.jsonl \
	--models-manifest config/models.example.yaml \
	--output-dir outputs \
	--conditions headlines_only headlines_with_sources sources_only headlines_with_manipulated_sources \
	--max-combinations 3 \
	--seed 42
```

Benchmark latency across models:

```bash
uv run python -m app.cli.benchmark_models \
	--models llama3.1:8b qwen2.5:7b gemma3:4b \
	--prompt "Return strict JSON only." \
	--rounds 3
```

## Data Contracts

- `PreparedIncident`: canonical incident input
- `ExperimentRequest`: per-condition request payload and prompt
- `ModelDecision`: parsed model output + parse status + raw response

These are defined in `app/schemas/models.py` and are the integration boundary between execution and storage/API/metrics. For a sample experiment run, see `outputs/*`.