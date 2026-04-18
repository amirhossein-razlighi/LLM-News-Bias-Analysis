# API Reference Overview

This section is generated from source modules using mkdocstrings.

Coverage includes:

- FastAPI analytics engine and ingestion utilities.
- Data schemas and enums.
- Experiment condition/prompt builders.
- Parsing logic.
- Ollama client integration.
- CLI entrypoint modules.

Tip: Use source links and signatures to navigate implementation quickly.

## Quick Usage Snippets

### Parse a model response

```python
from app.parsing.response_parser import parse_model_response

result = parse_model_response(
	'{"selected_article_id":"a2","reason":"balanced"}',
	allowed_article_ids={"a1", "a2", "a3"},
)
print(result.status, result.selected_article_id)
```

### Generate with Ollama client

```python
from app.models.ollama_client import OllamaClient

client = OllamaClient()
generation = client.generate(model="gemma4:latest", prompt="Say hello in JSON")
print(generation.latency_ms)
```

### Compute analytics metrics

```python
import pandas as pd
from app.api.engine_analytics import calculate_all_metrics

df = pd.DataFrame([...])
metrics = calculate_all_metrics(df)
print(metrics["parse_success_rate"])
```
