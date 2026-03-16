from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.schemas.models import ParseStatus


@dataclass
class ParseResult:
    selected_article_id: str | None
    reason: str | None
    status: ParseStatus
    parsed_json: dict | None
    error: str | None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _fallback_extract_article_id(text: str, allowed_article_ids: set[str]) -> str | None:
    for article_id in allowed_article_ids:
        if re.search(rf"\b{re.escape(article_id)}\b", text):
            return article_id
    return None


def _normalize_selected_article_id(
    value: object,
    allowed_article_ids: set[str],
) -> str | None:
    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if candidate in allowed_article_ids:
        return candidate

    # Models often return prefixed formats like "article_id=a_left_1".
    prefix_match = re.match(r"^\s*article_id\s*[:=]\s*(.+?)\s*$", candidate, flags=re.IGNORECASE)
    if prefix_match is not None:
        candidate = prefix_match.group(1).strip().strip('"\'')
        if candidate in allowed_article_ids:
            return candidate

    # Last resort: detect any known candidate ID inside the returned string.
    return _fallback_extract_article_id(candidate, allowed_article_ids)


def parse_model_response(text: str, allowed_article_ids: set[str]) -> ParseResult:
    json_blob = _extract_first_json_object(text)
    if json_blob is not None:
        try:
            payload = json.loads(json_blob)
            selected_article_id = _normalize_selected_article_id(
                payload.get("selected_article_id"),
                allowed_article_ids,
            )
            reason = payload.get("reason")
            if selected_article_id in allowed_article_ids:
                return ParseResult(
                    selected_article_id=selected_article_id,
                    reason=reason,
                    status=ParseStatus.SUCCESS,
                    parsed_json=payload,
                    error=None,
                )
            return ParseResult(
                selected_article_id=None,
                reason=reason,
                status=ParseStatus.FAILED,
                parsed_json=payload,
                error="selected_article_id missing or not in candidates",
            )
        except json.JSONDecodeError as exc:
            fallback_id = _fallback_extract_article_id(text, allowed_article_ids)
            if fallback_id is not None:
                return ParseResult(
                    selected_article_id=fallback_id,
                    reason="Fallback parse from non-JSON output.",
                    status=ParseStatus.FALLBACK,
                    parsed_json=None,
                    error=f"json decode error: {exc}",
                )
            return ParseResult(
                selected_article_id=None,
                reason=None,
                status=ParseStatus.FAILED,
                parsed_json=None,
                error=f"json decode error: {exc}",
            )

    fallback_id = _fallback_extract_article_id(text, allowed_article_ids)
    if fallback_id is not None:
        return ParseResult(
            selected_article_id=fallback_id,
            reason="Fallback parse from plain text output.",
            status=ParseStatus.FALLBACK,
            parsed_json=None,
            error="No JSON object found in output",
        )

    return ParseResult(
        selected_article_id=None,
        reason=None,
        status=ParseStatus.FAILED,
        parsed_json=None,
        error="No JSON object found and no candidate id detected",
    )
