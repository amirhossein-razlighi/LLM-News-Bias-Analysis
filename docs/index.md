# Sourcerers Technical Documentation

Sourcerers is an experimentation platform for measuring how LLMs select among politically diverse news sources under controlled prompt conditions.

This documentation is organized for three audiences:

- Developers extending experiment logic and analytics.
- Researchers interpreting robustness and bias signals.
- Operators deploying the API/dashboard stack.

## Core Components

- Experiment pipeline: incident preparation, condition bundle generation, prompt construction, model execution, and parsing.
- Analytics engine: FastAPI endpoints + shared metrics logic backed by the experiment database.
- Dashboard: Streamlit interface for run ingestion, diagnostics, model comparison, and narrative summary.
- Report assets: generated plots/tables under docs/figures for paper/report workflows.

## Where To Start

- New contributors: read [Quickstart](guides/quickstart.md) and [Architecture](architecture.md).
- API users: read [API Guide](guides/api.md), then [Analytics API Module](api/engine_analytics.md).
- Model evaluation workflows: read [CLI Guide](guides/cli.md) and [Experiment Flow](experiment-flow.md).
- Practical function/class usage: read [Examples](guides/examples.md).
