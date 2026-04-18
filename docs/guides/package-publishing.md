# Package Publishing

This project is publish-ready as a Python package with distribution name:

- sourcerers

Install/import example:

```bash
pip install sourcerers
```

```python
from app import OllamaClient, parse_model_response, build_selection_prompt
```

## Local Build And Validation

```bash
uv sync
uv run python -m build
uv run twine check dist/*
```

## GitHub Release Publishing

Publishing workflow:

- .github/workflows/publish-pypi.yml

Trigger options:

- Publish a GitHub Release (recommended)
- Manual workflow_dispatch

The workflow:

1. Builds source and wheel distributions.
2. Validates metadata with twine.
3. Publishes to PyPI via trusted publishing (OIDC).

## One-Time PyPI Setup

1. Create the project on PyPI (or first publish if name is available).
2. In PyPI project settings, add a Trusted Publisher:
   - Owner: amirhossein-razlighi
   - Repository: LLM-News-Bias-Analysis
   - Workflow: publish-pypi.yml
   - Environment: pypi
3. In GitHub repository settings, create environment pypi.
4. Ensure Actions workflow permissions allow id-token usage.

## TestPyPI (Optional)

If you want a dry run, duplicate the workflow for TestPyPI with:

- `repository_url: https://test.pypi.org/legacy/`
- environment: testpypi

## Public API Surface

The package exports stable entry points from app:

- Core schemas and enums
- Experiment builders
- Parsing helpers
- Ollama client
- Analytics helpers

Prefer importing from app for long-term stability.
