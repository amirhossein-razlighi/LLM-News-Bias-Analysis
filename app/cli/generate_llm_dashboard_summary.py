from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.models.ollama_client import OllamaClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dashboard-ready LLM summary from outputs/run_* artifacts using Ollama."
    )
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing run_* folders")
    parser.add_argument(
        "--summary-json",
        default="outputs/llm_dashboard_summary.json",
        help="Output file path for generated summary JSON",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--model", default="gemma4:latest", help="Ollama model used for analysis")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="Ollama request timeout")
    parser.add_argument("--max-tokens", type=int, default=1800, help="Max generation tokens")
    parser.add_argument("--max-runs", type=int, default=12, help="Limit number of newest runs analyzed")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def infer_bucket(article_id: str) -> str:
    token = (article_id or "").lower()
    if "_left" in token or token.startswith("left"):
        return "left"
    if "_center" in token or token.startswith("center"):
        return "center"
    if "_right" in token or token.startswith("right"):
        return "right"
    return "unknown"


def resolve_selected_bucket(selected_article_id: str, candidates: list[dict[str, Any]]) -> str:
    selected = (selected_article_id or "").strip()
    if selected and candidates:
        for candidate in candidates:
            if str(candidate.get("article_id") or "") != selected:
                continue
            leaning = str(candidate.get("leaning") or "").strip().lower()
            if leaning in {"left", "center", "right"}:
                return leaning
            break
    return infer_bucket(selected)


def load_decisions(outputs_dir: Path, max_runs: int) -> pd.DataFrame:
    runs = sorted(outputs_dir.glob("run_*"))
    if max_runs > 0:
        runs = runs[-max_runs:]

    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        requests_path = run_dir / "experiment_requests.jsonl"
        decisions_path = run_dir / "model_decisions.jsonl"
        if not decisions_path.exists():
            continue

        request_index: dict[str, dict[str, Any]] = {}
        if requests_path.exists():
            request_index = {
                str(r.get("request_id")): r
                for r in read_jsonl(requests_path)
                if r.get("request_id") is not None
            }

        for dec in read_jsonl(decisions_path):
            request_id = str(dec.get("request_id") or "")
            req = request_index.get(request_id, {})
            selected_article_id = str(dec.get("selected_article_id") or "")
            candidates = req.get("candidates") or []
            selected_bucket = resolve_selected_bucket(selected_article_id, candidates)
            rows.append(
                {
                    "run_id": str(dec.get("run_id") or run_dir.name),
                    "incident_id": str(dec.get("incident_id") or ""),
                    "model_name": str(dec.get("model_name") or ""),
                    "condition": str(dec.get("condition") or ""),
                    "parse_status": str(dec.get("parse_status") or ""),
                    "latency_ms": pd.to_numeric(dec.get("latency_ms"), errors="coerce"),
                    "selected_article_id": selected_article_id,
                    "selected_bucket": selected_bucket,
                    "candidate_signature": "|".join(
                        sorted(
                            str(c.get("article_id") or "")
                            for c in candidates
                            if c.get("article_id")
                        )
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df


def _label_sensitivity_by_model(df: pd.DataFrame) -> dict[str, float]:
    required = {"headlines_with_sources", "headlines_with_manipulated_sources"}
    work = df[df["condition"].isin(required)].copy()
    work = work[work["selected_bucket"].isin(["left", "center", "right"])].copy()
    if work.empty:
        return {}

    key_cols = ["run_id", "model_name", "incident_id", "candidate_signature"]
    pivot = (
        work.pivot_table(index=key_cols, columns="condition", values="selected_bucket", aggfunc="first")
        .dropna(subset=["headlines_with_sources", "headlines_with_manipulated_sources"])
        .reset_index()
    )
    if pivot.empty:
        return {}

    pivot["changed"] = (
        pivot["headlines_with_sources"] != pivot["headlines_with_manipulated_sources"]
    ).astype(float)
    return (
        pivot.groupby("model_name", dropna=False)["changed"]
        .mean()
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )


def build_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "record_count": 0,
            "models": [],
            "runs": [],
            "per_model": [],
            "best_model_by_composite": None,
            "notes": ["No records were found in outputs/run_*/model_decisions.jsonl"],
        }

    sensitivity_map = _label_sensitivity_by_model(df)
    rows: list[dict[str, Any]] = []
    for model_name, group in df.groupby("model_name", dropna=False):
        known = group[group["selected_bucket"].isin(["left", "center", "right"])]
        left_rate = float((known["selected_bucket"] == "left").mean()) if not known.empty else 0.0
        center_rate = float((known["selected_bucket"] == "center").mean()) if not known.empty else 0.0
        right_rate = float((known["selected_bucket"] == "right").mean()) if not known.empty else 0.0
        rows.append(
            {
                "model": str(model_name),
                "records": int(len(group)),
                "parse_success_rate": float((group["parse_status"] == "success").mean()),
                "parse_failure_rate": float((group["parse_status"] == "failed").mean()),
                "avg_latency_ms": float(group["latency_ms"].dropna().mean()) if group["latency_ms"].notna().any() else 0.0,
                "p95_latency_ms": float(group["latency_ms"].dropna().quantile(0.95)) if group["latency_ms"].notna().any() else 0.0,
                "left_rate": left_rate,
                "center_rate": center_rate,
                "right_rate": right_rate,
                "partisan_skew_score": left_rate - right_rate,
                "label_sensitivity_rate": float(sensitivity_map.get(str(model_name), 0.0)),
            }
        )

    per_model = pd.DataFrame(rows)
    latency_max = float(per_model["avg_latency_ms"].max()) if not per_model.empty else 1.0
    if latency_max <= 0:
        latency_max = 1.0
    per_model["latency_speed_score"] = 1.0 - (per_model["avg_latency_ms"] / latency_max)
    per_model["robustness_score"] = 1.0 - per_model["label_sensitivity_rate"]
    per_model["composite_score"] = (
        0.45 * per_model["parse_success_rate"]
        + 0.30 * per_model["latency_speed_score"]
        + 0.25 * per_model["robustness_score"]
    )

    best_model = None
    if not per_model.empty:
        top = per_model.sort_values(["composite_score", "records"], ascending=[False, False]).iloc[0]
        best_model = {
            "model": str(top["model"]),
            "composite_score": float(top["composite_score"]),
            "why": (
                "Best weighted balance of reliability, speed, and robustness "
                "using composite = 0.45*parse_success + 0.30*speed + 0.25*robustness"
            ),
        }

    return {
        "record_count": int(len(df)),
        "models": sorted([m for m in df["model_name"].dropna().astype(str).unique().tolist() if m]),
        "runs": sorted([r for r in df["run_id"].dropna().astype(str).unique().tolist() if r]),
        "per_model": per_model.sort_values("composite_score", ascending=False).round(6).to_dict(orient="records"),
        "best_model_by_composite": best_model,
        "notes": [
            "Composite ranking favors parse reliability first, then speed, then robustness.",
            "Label sensitivity is measured from bucket changes between headlines_with_sources and headlines_with_manipulated_sources.",
        ],
    }


def build_prompt(snapshot: dict[str, Any]) -> str:
    instructions = {
        "task": "You are producing a concise, honest executive summary for a model-bias experiment dashboard.",
        "requirements": [
            "Focus on what each metric means in plain language.",
            "Pick the best model using the provided composite winner and justify trade-offs.",
            "Discuss strengths and flaws for each model listed in per_model.",
            "Include direct model-vs-model comparisons for key trade-offs (reliability, speed, robustness, skew).",
            "Call out flaws, possible biases, and limitations.",
            "Provide actionable recommendations.",
            "Do not invent data not present in INPUT_SNAPSHOT.",
            "Every bullet should include at least one concrete metric value where possible.",
            "Keep each bullet concise and specific.",
        ],
    }

    return (
        "Return STRICT JSON only. No markdown. No prose before/after JSON.\n"
        "JSON schema keys required:\n"
        "headline (string),\n"
        "executive_summary (string),\n"
        "best_model (object: name, rationale, tradeoffs),\n"
        "metric_highlights (array of strings, 3 to 6 items),\n"
        "flaws_and_biases (array of strings, 4 to 8 items),\n"
        "recommendations (array of strings, 4 to 8 items),\n"
        "confidence_and_caveats (array of strings, 3 to 6 items),\n"
        "per_model_insights (array of objects, one per model):\n"
        "  - model (string)\n"
        "  - strengths (array of 2 to 4 strings)\n"
        "  - weaknesses (array of 2 to 4 strings)\n"
        "  - deployment_fit (string, one sentence)\n"
        "model_comparisons (array of objects, 3 to 8 items):\n"
        "  - title (string)\n"
        "  - winner (string)\n"
        "  - loser_or_tradeoff (string)\n"
        "  - evidence (string with metric values)\n"
        "  - takeaway (string).\n\n"
        f"INSTRUCTIONS={json.dumps(instructions, ensure_ascii=True)}\n"
        f"INPUT_SNAPSHOT={json.dumps(snapshot, ensure_ascii=True)}"
    )


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    # Try decoding from any '{' offset; this is more robust than naive slicing.
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text, idx)
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    raise ValueError("Could not parse JSON object from model output")


def build_fallback_llm_summary(snapshot: dict[str, Any], reason: str) -> dict[str, Any]:
    per_model = snapshot.get("per_model") if isinstance(snapshot.get("per_model"), list) else []
    best = snapshot.get("best_model_by_composite") if isinstance(snapshot.get("best_model_by_composite"), dict) else {}
    best_name = str(best.get("model") or "N/A")

    per_model_insights: list[dict[str, Any]] = []
    for row in per_model:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "unknown")
        parse_rate = float(row.get("parse_success_rate") or 0.0)
        latency = float(row.get("avg_latency_ms") or 0.0)
        sens = float(row.get("label_sensitivity_rate") or 0.0)
        per_model_insights.append(
            {
                "model": model,
                "strengths": [
                    f"Parse success: {parse_rate:.2%}",
                    f"Average latency: {latency:.0f} ms",
                ],
                "weaknesses": [
                    f"Label sensitivity: {sens:.3f}",
                    "Needs qualitative error review on sample prompts.",
                ],
                "deployment_fit": "General-purpose candidate; validate on target latency and risk profile.",
            }
        )

    model_comparisons: list[dict[str, Any]] = []
    if per_model and isinstance(per_model[0], dict):
        fastest = min(per_model, key=lambda r: float(r.get("avg_latency_ms") or 0.0))
        most_reliable = max(per_model, key=lambda r: float(r.get("parse_success_rate") or 0.0))
        model_comparisons.append(
            {
                "title": "Fastest model",
                "winner": str(fastest.get("model") or "N/A"),
                "loser_or_tradeoff": "Others are slower on avg latency",
                "evidence": f"avg_latency_ms={float(fastest.get('avg_latency_ms') or 0.0):.0f}",
                "takeaway": "Choose for latency-sensitive serving if reliability is acceptable.",
            }
        )
        model_comparisons.append(
            {
                "title": "Most reliable parser",
                "winner": str(most_reliable.get("model") or "N/A"),
                "loser_or_tradeoff": "Others have lower parse success",
                "evidence": f"parse_success_rate={float(most_reliable.get('parse_success_rate') or 0.0):.2%}",
                "takeaway": "Choose for automation workflows that require strict JSON stability.",
            }
        )

    return {
        "headline": "Model Bias Experiment Dashboard Summary",
        "executive_summary": (
            "Generated fallback summary because the model output could not be parsed as strict JSON. "
            "Snapshot metrics below are valid and can still be used for model comparison."
        ),
        "best_model": {
            "name": best_name,
            "rationale": "Selected from snapshot best_model_by_composite.",
            "tradeoffs": "Review latency vs reliability trade-offs in per-model table.",
        },
        "metric_highlights": [
            f"Records analyzed: {int(snapshot.get('record_count', 0) or 0)}",
            f"Models compared: {len(snapshot.get('models', []) or [])}",
            f"Best by composite: {best_name}",
        ],
        "flaws_and_biases": [
            "LLM narrative generation failed strict JSON parsing in this run.",
            "Bias interpretation should be cross-checked with raw metric tables.",
            "Composite ranking depends on the chosen weighting scheme.",
            f"Fallback reason: {reason}",
        ],
        "recommendations": [
            "Rerun summary generation with a higher timeout or max-tokens if needed.",
            "Inspect model_decisions.jsonl for qualitative error patterns.",
            "Keep using snapshot metrics; they are computed directly from artifacts.",
            "Compare reliability and latency objectives before choosing deployment model.",
        ],
        "confidence_and_caveats": [
            "Numeric snapshot metrics are deterministic from outputs artifacts.",
            "Narrative sections are fallback-generated in this run.",
            "Use regenerated LLM narrative for richer qualitative commentary.",
        ],
        "per_model_insights": per_model_insights,
        "model_comparisons": model_comparisons,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    quiet = bool(args.quiet)

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")

    log(f"[1/5] Loading run artifacts from {outputs_dir} ...", quiet)
    df = load_decisions(outputs_dir=outputs_dir, max_runs=int(args.max_runs))
    runs_seen = int(df["run_id"].nunique()) if not df.empty and "run_id" in df.columns else 0
    models_seen = int(df["model_name"].nunique()) if not df.empty and "model_name" in df.columns else 0
    log(
        f"Loaded {len(df)} decisions across {runs_seen} runs and {models_seen} models (max_runs={int(args.max_runs)}).",
        quiet,
    )

    log("[2/5] Computing snapshot metrics ...", quiet)
    snapshot = build_snapshot(df)
    log(
        f"Snapshot ready: records={int(snapshot.get('record_count', 0))}, "
        f"models={len(snapshot.get('models', []))}, runs={len(snapshot.get('runs', []))}",
        quiet,
    )

    log("[3/5] Building LLM prompt and response schema ...", quiet)
    prompt = build_prompt(snapshot)

    response_schema = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "executive_summary": {"type": "string"},
            "best_model": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "tradeoffs": {"type": "string"},
                },
                "required": ["name", "rationale", "tradeoffs"],
                "additionalProperties": False,
            },
            "metric_highlights": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "flaws_and_biases": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "recommendations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "confidence_and_caveats": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "per_model_insights": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "strengths": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "weaknesses": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "deployment_fit": {"type": "string"},
                    },
                    "required": ["model", "strengths", "weaknesses", "deployment_fit"],
                    "additionalProperties": False,
                },
            },
            "model_comparisons": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "winner": {"type": "string"},
                        "loser_or_tradeoff": {"type": "string"},
                        "evidence": {"type": "string"},
                        "takeaway": {"type": "string"},
                    },
                    "required": ["title", "winner", "loser_or_tradeoff", "evidence", "takeaway"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "headline",
            "executive_summary",
            "best_model",
            "metric_highlights",
            "flaws_and_biases",
            "recommendations",
            "confidence_and_caveats",
            "per_model_insights",
            "model_comparisons",
        ],
        "additionalProperties": False,
    }

    client = OllamaClient(base_url=args.base_url)
    log(
        f"[4/5] Requesting summary from Ollama model={args.model} at {args.base_url} "
        f"(timeout={int(args.timeout_seconds)}s, max_tokens={int(args.max_tokens)}) ...",
        quiet,
    )
    generation = client.generate(
        model=args.model,
        prompt=prompt,
        temperature=0.2,
        max_tokens=int(args.max_tokens),
        timeout_seconds=int(args.timeout_seconds),
        retries=2,
        think=False,
        response_schema=response_schema,
    )
    log(f"Model response received (latency={int(generation.latency_ms)} ms).", quiet)

    generation_used = generation
    retry_used = False
    max_tokens_used = int(args.max_tokens)

    try:
        llm_summary = _extract_first_json_object(generation.text)
    except Exception as exc:
        retry_tokens = min(max(int(args.max_tokens) * 2, int(args.max_tokens) + 800), 4096)
        if retry_tokens > int(args.max_tokens):
            log(
                f"Warning: strict JSON parse failed ({exc}). Retrying once with max_tokens={retry_tokens} ...",
                quiet,
            )
            retry_generation = client.generate(
                model=args.model,
                prompt=prompt,
                temperature=0.1,
                max_tokens=retry_tokens,
                timeout_seconds=int(args.timeout_seconds),
                retries=1,
                think=False,
                response_schema=response_schema,
            )
            generation_used = retry_generation
            retry_used = True
            max_tokens_used = retry_tokens
            log(f"Retry response received (latency={int(retry_generation.latency_ms)} ms).", quiet)
            try:
                llm_summary = _extract_first_json_object(retry_generation.text)
            except Exception as retry_exc:
                log(
                    f"Warning: retry parsing also failed ({retry_exc}). Falling back to deterministic summary.",
                    quiet,
                )
                llm_summary = build_fallback_llm_summary(snapshot, reason=str(retry_exc))
        else:
            log(f"Warning: strict JSON parse failed ({exc}). Falling back to deterministic summary.", quiet)
            llm_summary = build_fallback_llm_summary(snapshot, reason=str(exc))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "model": args.model,
            "base_url": args.base_url,
            "timeout_seconds": int(args.timeout_seconds),
            "max_tokens": int(args.max_tokens),
            "max_tokens_used": int(max_tokens_used),
            "retry_used": bool(retry_used),
            "max_runs": int(args.max_runs),
        },
        "snapshot": snapshot,
        "llm_summary": llm_summary,
        "latency_ms": int(generation_used.latency_ms),
    }

    output_path = Path(args.summary_json)
    ensure_parent(output_path)
    log(f"[5/5] Writing summary JSON to {output_path} ...", quiet)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    log("Done. LLM dashboard summary generated successfully.", quiet)

    print(json.dumps({"summary_json": str(output_path), "records": snapshot.get("record_count", 0)}, indent=2))


if __name__ == "__main__":
    main()
