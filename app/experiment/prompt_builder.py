"""Prompt construction utilities for controlled source-selection experiments."""

from __future__ import annotations

import json

from app.schemas.models import ConditionName, PreparedIncident, PresentedArticle


def _build_candidate_text(candidates: list[PresentedArticle], condition: ConditionName) -> str:
    lines: list[str] = []
    for idx, candidate in enumerate(candidates, start=1):
        if condition == ConditionName.HEADLINES_ONLY:
            lines.append(
                f"{idx}. article_id={candidate.article_id} | headline={candidate.headline}"
            )
        elif condition == ConditionName.HEADLINES_WITH_SOURCES:
            lines.append(
                f"{idx}. article_id={candidate.article_id} | outlet={candidate.outlet_name} | headline={candidate.headline}"
            )
        elif condition == ConditionName.SOURCES_ONLY:
            lines.append(
                f"{idx}. article_id={candidate.article_id} | outlet={candidate.outlet_name}"
            )
        elif condition == ConditionName.HEADLINES_WITH_MANIPULATED_SOURCES:
            lines.append(
                f"{idx}. article_id={candidate.article_id} | outlet_label={candidate.outlet_name} | headline={candidate.headline}"
            )
        else:
            raise ValueError(f"Unsupported condition: {condition}")
    return "\n".join(lines)


def build_selection_prompt(
    incident: PreparedIncident,
    candidates: list[PresentedArticle],
    condition: ConditionName,
) -> str:
    schema_example = {
        "selected_article_id": "AN_ID_FROM_CANDIDATES_LIST",
        "reason": "One short explanation."
    }

    instructions = [
        "You are evaluating information sources for a news incident.",
        "Choose exactly one article_id that a user should focus on for more information.",
        "selected_article_id must be the raw id value only, not 'article_id=...'. No prefixes and suffixes, just the id from the candidates list.",
        "Return strict JSON only with no markdown, no prose outside JSON, and no extra keys.",
        "If uncertain, still choose one article_id from the list.",
    ]

    prompt = "\n".join(instructions)
    candidate_block = _build_candidate_text(candidates, condition)

    return (
        f"{prompt}\n\n"
        f"Incident ID: {incident.incident_id}\n"
        f"Topic: {incident.topic or 'unknown'}\n"
        f"Neutral Summary: {incident.neutral_summary}\n"
        "\n"
        f"Candidates:\n{candidate_block}\n\n"
        f"Output JSON schema:\n{json.dumps(schema_example)}"
    )


def selection_response_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "selected_article_id": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["selected_article_id", "reason"],
        "additionalProperties": False,
    }
