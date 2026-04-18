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

## Runtime Optimization Flags (run_experiments)

The experiment runner supports optional low-level runtime options for Ollama:

- `--enable-flash-attention`
- `--enable-kv-cache`
- `--kv-cache-type <value>`

Example:

```bash
uv run python -m app.cli.run_experiments \
	--input data/real_incidents_all.jsonl \
	--models-manifest configs/models.example.yaml \
	--output-dir outputs \
	--enable-flash-attention \
	--enable-kv-cache \
	--kv-cache-type q8_0
```

Defaults intentionally keep these unset so existing workflows and historical output reproducibility stay unchanged.
