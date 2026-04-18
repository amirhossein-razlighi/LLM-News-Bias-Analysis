# Deployment

## Dashboard

Deploy Streamlit app using dashboard.py as entrypoint.

## API

Run local or cloud FastAPI deployment:

```bash
uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000
```

## Documentation Site

Build docs locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Build static site:

```bash
uv run mkdocs build --strict
```

GitHub Pages publishing is automated via workflow in .github/workflows/docs.yml.
