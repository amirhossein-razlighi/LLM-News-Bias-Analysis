# Examples

This page contains practical examples for core classes and functions used throughout Sourcerers.

## 1. Build Condition Bundles

```python
from app.experiment.condition_builder import build_condition_bundles
from app.schemas.models import ConditionName, PreparedIncident

incident = PreparedIncident.model_validate({
    "incident_id": "topic_example",
    "topic": "example",
    "neutral_summary": "Example summary",
    "articles": [
        {"article_id": "a1", "headline": "H1", "outlet_name": "OutletA", "leaning": "left"},
        {"article_id": "a2", "headline": "H2", "outlet_name": "OutletB", "leaning": "center"},
        {"article_id": "a3", "headline": "H3", "outlet_name": "OutletC", "leaning": "right"},
    ],
})

bundles = build_condition_bundles(
    incident=incident,
    conditions=[
        ConditionName.HEADLINES_ONLY,
        ConditionName.HEADLINES_WITH_SOURCES,
    ],
    max_combinations=1,
    seed=42,
)
```

## 2. Build Selection Prompt

```python
from app.experiment.prompt_builder import build_selection_prompt
from app.schemas.models import ConditionName, PresentedArticle

candidates = [
    PresentedArticle(article_id="a1", headline="H1", outlet_name="OutletA", leaning="left"),
    PresentedArticle(article_id="a2", headline="H2", outlet_name="OutletB", leaning="center"),
    PresentedArticle(article_id="a3", headline="H3", outlet_name="OutletC", leaning="right"),
]

prompt = build_selection_prompt(
    incident=incident,
    candidates=candidates,
    condition=ConditionName.HEADLINES_WITH_SOURCES,
)
print(prompt[:300])
```

## 3. Parse Model Response

```python
from app.parsing.response_parser import parse_model_response

allowed_ids = {"a1", "a2", "a3"}
raw_text = '{"selected_article_id": "a2", "reason": "Balanced framing"}'

parsed = parse_model_response(raw_text, allowed_article_ids=allowed_ids)
print(parsed.status, parsed.selected_article_id, parsed.reason)
```

## 4. Ollama Client Usage

```python
from app.models.ollama_client import OllamaClient

client = OllamaClient(base_url="http://localhost:11434")
models = client.list_models()

result = client.generate(
    model=models[0],
    prompt='Return JSON: {"selected_article_id":"a1","reason":"example"}',
    temperature=0.0,
    max_tokens=200,
    timeout_seconds=60,
)
print(result.latency_ms)
print(result.text)
```

## 5. Compute Metrics From DataFrame

```python
import pandas as pd
from app.api.engine_analytics import calculate_all_metrics

df = pd.DataFrame([
    {
        "incident_id": "topic_example",
        "model_name": "gemma4:latest",
        "condition": "headlines_only",
        "selected_article_id": "a2",
        "selected_bucket": "center",
        "parsed_successfully": True,
        "parse_status": "success",
        "latency_ms": 1200,
        "candidate_signature": "a1|a2|a3",
    }
])

metrics = calculate_all_metrics(df)
print(metrics["parse_success_rate"], metrics["center_preference_index"])
```

## 6. Ingest Existing Runs Into Analytics DB

```python
from pathlib import Path
from app.api.engine_analytics import sync_outputs_to_db

resp = sync_outputs_to_db(
    outputs_dir=Path("outputs"),
)
print(resp["runs_ingested"], resp["records_added"])
```

## 7. Create/Validate Model Manifest

```python
from app.schemas.models import ModelManifest

manifest = ModelManifest.model_validate({
    "models": [
        {
            "name": "gemma4:latest",
            "temperature": 0.0,
            "max_tokens": 300,
            "timeout_seconds": 60,
        }
    ]
})
print(manifest.models[0].name)
```
