from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConditionName(str, Enum):
    HEADLINES_ONLY = "headlines_only"
    HEADLINES_WITH_SOURCES = "headlines_with_sources"
    SOURCES_ONLY = "sources_only"
    HEADLINES_WITH_MANIPULATED_SOURCES = "headlines_with_manipulated_sources"


class ParseStatus(str, Enum):
    SUCCESS = "success"
    FALLBACK = "fallback"
    FAILED = "failed"


class Article(BaseModel):
    article_id: str
    headline: str
    body: str | None = None
    outlet_name: str
    leaning: str

    @field_validator("leaning")
    @classmethod
    def validate_leaning(cls, value: str) -> str:
        normalized = value.strip().lower()
        allowed = {"left", "center", "right"}
        if normalized not in allowed:
            raise ValueError(f"leaning must be one of {sorted(allowed)}")
        return normalized


class PreparedIncident(BaseModel):
    incident_id: str
    topic: str | None = None
    neutral_summary: str
    articles: list[Article]
    metadata: dict[str, Any] = Field(default_factory=dict)


class PresentedArticle(BaseModel):
    article_id: str
    headline: str | None = None
    outlet_name: str | None = None
    leaning: str | None = None


class ExperimentRequest(BaseModel):
    request_id: str
    run_id: str
    incident_id: str
    model_name: str
    condition: ConditionName
    prompt: str
    candidates: list[PresentedArticle]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelDecision(BaseModel):
    request_id: str
    run_id: str
    incident_id: str
    model_name: str
    condition: ConditionName
    selected_article_id: str | None = None
    reason: str | None = None
    parse_status: ParseStatus
    raw_response: str
    response_json: dict[str, Any] | None = None
    error: str | None = None
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelSpec(BaseModel):
    name: str
    temperature: float = 0.0
    max_tokens: int = 300
    timeout_seconds: int = 60
    think: bool | None = None


class ModelManifest(BaseModel):
    models: list[ModelSpec]
