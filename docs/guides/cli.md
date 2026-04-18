# CLI Guide

## Primary Commands

- app.cli.prepare_real_incidents: builds experiment-ready incidents from raw article JSON.
- app.cli.run_experiments: executes all model-condition-request combinations and writes run artifacts.
- app.cli.generate_report_assets: computes figures and markdown/json analysis tables from saved runs.
- app.cli.generate_llm_dashboard_summary: generates offline LLM narrative summary for dashboard cards.
- app.cli.probe_model: single-prompt health check against Ollama model.
- app.cli.list_models: lists local Ollama models.
- app.cli.benchmark_models: quick latency comparison across models.

## Operational Tips

- Keep one output folder per run for reproducibility.
- Use deterministic seeds for comparable experiment slices.
- Regenerate report assets and LLM summary after adding new runs.
- Treat parse_failure and fallback records as first-class diagnostics.
