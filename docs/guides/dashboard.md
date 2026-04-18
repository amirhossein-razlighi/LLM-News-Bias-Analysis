# Dashboard Guide

## Dashboard Sections

- Analytics tab: metrics cards, comparisons, condition breakdowns, outlet stats, and scenario simulator.
- Run Models tab: local Ollama probing and batch experiment execution.
- Metrics explanations tab: definitions, formulas, and directional optimization hints.

## LLM Summary Section

The top-level LLM summary is loaded from outputs/llm_dashboard_summary.json.

Key behavior:

- Uses offline-generated summary content when available.
- Falls back to snapshot-derived deterministic content if fields are missing.
- Supports model-insight cards and model-vs-model comparison table.

## Common Maintenance

- Use the sidebar ingest controls after creating new runs.
- Refresh snapshot-dependent sections after ingestion.
- Keep outputs and experiment_database.csv aligned when reporting.
