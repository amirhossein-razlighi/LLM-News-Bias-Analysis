# API Guide

Base app: app.api.engine_analytics:app

## Health

- GET /healthz

## Ingestion

- POST /ingest
- POST /ingest/run
- POST /ingest/runs

## Metrics

- GET /metrics/summary
- GET /metrics/inter-model
- GET /metrics/conditions
- GET /metrics/conditions-by-model
- GET /metrics/run-summaries
- GET /metrics/top-outlets
- GET /metrics/top-outlets-by-model
- GET /metrics/records
- GET /metrics/runs
- GET /metrics/compare-runs

## Export

- GET /export/csv

## Notes

- Write endpoints can be disabled via environment setting ENABLE_ANALYTICS_WRITE_ENDPOINTS.
- Optional CORS origins can be set via API_ALLOW_ORIGINS.
