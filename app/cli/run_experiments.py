from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from app.experiment.condition_builder import build_condition_bundles
from app.experiment.prompt_builder import build_selection_prompt
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
from app.utils.io import read_jsonl, write_jsonl


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
    return parser.parse_args()


def _load_conditions(raw_conditions: list[str]) -> list[ConditionName]:
    return [ConditionName(c) for c in raw_conditions]


def _load_manifest(path: str) -> ModelManifest:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return ModelManifest.model_validate(payload)


def main() -> None:
    args = parse_args()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    incidents = [PreparedIncident.model_validate(row) for row in read_jsonl(args.input)]
    conditions = _load_conditions(args.conditions)
    manifest = _load_manifest(args.models_manifest)

    client = OllamaClient(base_url=args.base_url)

    request_records: list[dict] = []
    decision_records: list[dict] = []
    raw_records: list[dict] = []

    for incident in incidents:
        bundles = build_condition_bundles(
            incident,
            conditions=conditions,
            max_combinations=args.max_combinations,
            seed=args.seed,
        )

        for model in manifest.models:
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
                    request_records.append(request.model_dump(mode="json"))

                    try:
                        generation = client.generate(
                            model=model.name,
                            prompt=prompt,
                            temperature=model.temperature,
                            max_tokens=model.max_tokens,
                            timeout_seconds=model.timeout_seconds,
                            retries=args.retries,
                        )
                        parsed = parse_model_response(
                            text=generation.text,
                            allowed_article_ids={c.article_id for c in candidates},
                        )

                        raw_records.append(
                            {
                                "request_id": request_id,
                                "run_id": run_id,
                                "incident_id": incident.incident_id,
                                "model_name": model.name,
                                "condition": condition.value,
                                "bundle_index": bundle_idx,
                                "raw_payload": generation.raw_payload,
                            }
                        )

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

                    decision_records.append(decision.model_dump(mode="json"))

    output_dir = Path(args.output_dir) / run_id
    write_jsonl(output_dir / "experiment_requests.jsonl", request_records)
    write_jsonl(output_dir / "model_decisions.jsonl", decision_records)
    write_jsonl(output_dir / "raw_outputs.jsonl", raw_records)

    summary = {
        "run_id": run_id,
        "incident_count": len(incidents),
        "request_count": len(request_records),
        "decision_count": len(decision_records),
        "output_dir": str(output_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
