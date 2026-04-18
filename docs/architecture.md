# Architecture

## High-Level System

Sourcerers follows a batch-first architecture with optional online analytics:

1. Prepare incidents from raw article JSON data.
2. Build condition-specific candidate bundles.
3. Execute prompts against one or more Ollama models.
4. Parse and persist model decisions per run.
5. Ingest run artifacts into analytics storage.
6. Compute metrics for API, dashboard, and report assets.

## Runtime Surfaces

- CLI modules in app/cli: experiment orchestration and asset generation.
- FastAPI app in app/api/engine_analytics.py: ingestion + metrics endpoints.
- Streamlit dashboard in dashboard.py: visualization and operational controls.

## Data Artifacts

Per run (outputs/run_*/):

- experiment_requests.jsonl: prompt inputs and candidate order.
- model_decisions.jsonl: parsed outputs + status + latency.
- raw_outputs.jsonl: raw generation payloads for auditing.

Global analytics storage:

- experiment_database.csv: deduplicated request-level analytics table.

## Design Principles

- Reproducibility: explicit seeds and persisted run artifacts.
- Auditability: raw outputs retained alongside parsed decisions.
- Composability: shared metrics engine powers both API and dashboard.
- Failure tolerance: parser fallback pathways and qualitative error extraction.
