from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

TQDM_AVAILABLE = True

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal installs
    TQDM_AVAILABLE = False

    class _NullProgress:
        def update(self, _: int) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, total=None, **_: object):
        if iterable is not None:
            return iterable
        return _NullProgress()

from app.experiment.condition_builder import build_condition_bundles
from app.experiment.prompt_builder import build_selection_prompt, selection_response_json_schema
from app.models.ollama_client import OllamaClient
from app.parsing.response_parser import parse_model_response
from app.schemas.models import (
    ConditionName,
    ExperimentRequest,
    ModelDecision,
    ModelManifest,
    ParseStatus,
    PreparedIncident,
)
from app.utils.io import append_jsonl, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local source-selection experiments with Ollama.")
    parser.add_argument("--input", required=True, help="Path to prepared incidents JSONL")
    parser.add_argument("--models-manifest", required=True, help="Path to YAML model manifest")
    parser.add_argument("--output-dir", required=True, help="Directory for output JSONL files")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[c.value for c in ConditionName],
        help="Conditions to run",
    )
    parser.add_argument("--max-combinations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument(
        "--shuffle-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize candidate display order per bundle for position-bias control",
    )
    return parser.parse_args()


def _load_conditions(raw_conditions: list[str]) -> list[ConditionName]:
    return [ConditionName(c) for c in raw_conditions]


def _load_manifest(path: str) -> ModelManifest:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return ModelManifest.model_validate(payload)


def _prepare_bundles(
    incidents: list[PreparedIncident],
    conditions: list[ConditionName],
    max_combinations: int,
    seed: int,
    shuffle_candidates: bool,
) -> list[tuple[PreparedIncident, dict[ConditionName, list[list]]]]:
    prepared: list[tuple[PreparedIncident, dict[ConditionName, list[list]]]] = []
    for incident in incidents:
        bundles = build_condition_bundles(
            incident,
            conditions=conditions,
            max_combinations=max_combinations,
            seed=seed,
            shuffle_candidates=shuffle_candidates,
        )
        prepared.append((incident, bundles))
    return prepared


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    if not TQDM_AVAILABLE:
        print(
            "Warning: tqdm is not installed, so no progress bar will be shown. "
            "Run `uv sync` to install project dependencies.",
            file=sys.stderr,
        )

    incidents = [PreparedIncident.model_validate(row) for row in read_jsonl(args.input)]
    conditions = _load_conditions(args.conditions)
    manifest = _load_manifest(args.models_manifest)

    client = OllamaClient(base_url=args.base_url)
    response_schema = selection_response_json_schema()

    prepared_runs = _prepare_bundles(
        incidents,
        conditions=conditions,
        max_combinations=args.max_combinations,
        seed=args.seed,
        shuffle_candidates=args.shuffle_candidates,
    )
    total_requests = sum(
        len(manifest.models) * sum(len(condition_bundles) for condition_bundles in bundles.values())
        for _, bundles in prepared_runs
    )

    output_dir = Path(args.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "experiment_requests.jsonl"
    decisions_path = output_dir / "model_decisions.jsonl"
    raw_outputs_path = output_dir / "raw_outputs.jsonl"
    for path in (requests_path, decisions_path, raw_outputs_path):
        path.touch()

    request_count = 0
    decision_count = 0

    progress = tqdm(total=total_requests, desc="Running experiments", unit="request")

    for model in manifest.models:
        for incident, bundles in prepared_runs:
            for condition, condition_bundles in bundles.items():
                for bundle_idx, candidates in enumerate(condition_bundles, start=1):
                    request_id = str(uuid4())
                    prompt = build_selection_prompt(
                        incident=incident,
                        candidates=candidates,
                        condition=condition,
                    )

                    request = ExperimentRequest(
                        request_id=request_id,
                        run_id=run_id,
                        incident_id=incident.incident_id,
                        model_name=model.name,
                        condition=condition,
                        prompt=prompt,
                        candidates=candidates,
                    )
                    request_record = request.model_dump(mode="json")
                    request_record["candidate_order"] = [c.article_id for c in candidates]
                    append_jsonl(requests_path, request_record)
                    request_count += 1

                    try:
                        generation = client.generate(
                            model=model.name,
                            prompt=prompt,
                            temperature=model.temperature,
                            max_tokens=model.max_tokens,
                            timeout_seconds=model.timeout_seconds,
                            retries=args.retries,
                            think=model.think,
                            response_schema=response_schema,
                        )
                        parsed = parse_model_response(
                            text=generation.text,
                            allowed_article_ids={c.article_id for c in candidates},
                        )

                        raw_record = {
                            "request_id": request_id,
                            "run_id": run_id,
                            "incident_id": incident.incident_id,
                            "model_name": model.name,
                            "condition": condition.value,
                            "bundle_index": bundle_idx,
                            "candidate_order": [c.article_id for c in candidates],
                            "raw_payload": generation.raw_payload,
                        }
                        append_jsonl(raw_outputs_path, raw_record)

                        decision = ModelDecision(
                            request_id=request_id,
                            run_id=run_id,
                            incident_id=incident.incident_id,
                            model_name=model.name,
                            condition=condition,
                            selected_article_id=parsed.selected_article_id,
                            reason=parsed.reason,
                            parse_status=parsed.status,
                            raw_response=generation.text,
                            response_json=parsed.parsed_json,
                            error=parsed.error,
                            latency_ms=generation.latency_ms,
                        )
                    except Exception as exc:
                        decision = ModelDecision(
                            request_id=request_id,
                            run_id=run_id,
                            incident_id=incident.incident_id,
                            model_name=model.name,
                            condition=condition,
                            selected_article_id=None,
                            reason=None,
                            parse_status=ParseStatus.FAILED,
                            raw_response="",
                            response_json=None,
                            error=str(exc),
                            latency_ms=None,
                        )

                    append_jsonl(decisions_path, decision.model_dump(mode="json"))
                    decision_count += 1
                    progress.update(1)

    progress.close()

    summary = {
        "run_id": run_id,
        "incident_count": len(incidents),
        "request_count": request_count,
        "decision_count": decision_count,
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
