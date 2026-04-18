# Data Model

## Core Domain Objects

Defined in [API schemas reference](api/schemas_models.md):

- Article: source article metadata (id, headline, outlet, leaning).
- PreparedIncident: topic-level bundle of candidate articles.
- PresentedArticle: condition-specific view of article fields shown to model.
- ExperimentRequest: prompt payload persisted to experiment_requests.jsonl.
- ModelDecision: parsed model result persisted to model_decisions.jsonl.

## Enumerations

- ConditionName:
  - headlines_only
  - headlines_with_sources
  - sources_only
  - headlines_with_manipulated_sources
- ParseStatus:
  - success
  - fallback
  - failed

## Analytics Table Fields

Key columns in experiment_database.csv:

- request_id, run_id, incident_id, model_name, condition
- selected_article_id, selected_outlet, selected_bucket
- parsed_successfully, parse_status, latency_ms
- selected_position, candidate_signature
- timestamp_utc, error, raw_response, justification
