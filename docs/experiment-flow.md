# Experiment Flow

## 1. Incident Preparation

Raw JSON articles are grouped into incidents with balanced ideological coverage.

## 2. Condition Bundling

Condition bundles are generated from left/center/right triplets with optional shuffling and reproducible seeds.

Reference: [Experiment builder API reference](api/experiment.md)

## 3. Prompt Construction

Prompts are synthesized per condition with strict JSON output requirements.

Reference: [Prompt builder API reference](api/experiment.md)

## 4. Model Execution

Ollama client executes model calls using endpoint fallbacks and optional response schemas.

Reference: [Ollama client API reference](api/ollama_client.md)

## 5. Parsing

Responses are parsed with strict JSON first, then robust fallbacks, and categorized with ParseStatus.

Reference: [Response parsing API reference](api/parsing.md)

## 6. Ingestion & Metrics

Run artifacts are normalized into analytics storage and exposed through FastAPI and Streamlit.

Reference: [Analytics engine API reference](api/engine_analytics.md)
