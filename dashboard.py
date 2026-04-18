from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from app.api.engine_analytics import (
    DB_FILE,
    _apply_filters,
    _condition_metrics_by_model,
    _ingest_run_directory,
    _run_summaries,
    _sample_records,
    _top_outlets_by_model,
    calculate_all_metrics,
    load_db,
    sync_outputs_to_db,
)
from app.experiment.condition_builder import build_condition_bundles
from app.experiment.prompt_builder import build_selection_prompt, selection_response_json_schema
from app.models.litellm_client import PROVIDER_MODELS, LiteLLMClient
from app.models.ollama_client import OllamaClient
from app.parsing.response_parser import parse_model_response
from app.schemas.models import (
    Article,
    ConditionName,
    ExperimentRequest,
    ModelDecision,
    ModelManifest,
    ParseStatus,
    PreparedIncident,
)
from app.utils.io import append_jsonl, read_jsonl

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_OUTPUTS_DIR = "outputs"
IS_STREAMLIT_CLOUD = bool(os.getenv("STREAMLIT_SHARING_MODE"))


def render_filter_scope(selected_run: str, *, label: str = "Scope") -> None:
    run_text = selected_run if selected_run else "All runs"
    st.caption(f"{label}: cross-model comparison | run = {run_text}")


@st.cache_data(ttl=5)
def get_dashboard_snapshot(selected_run: str = "", record_limit: int = 50) -> dict[str, Any]:
    sync_outputs_to_db()

    df = load_db()
    filtered = _apply_filters(df, run_id=selected_run or None)
    metrics = calculate_all_metrics(filtered)

    if filtered.empty:
        inter_model: dict[str, Any] = {}
    else:
        inter_model = {
            model: {
                **calculate_all_metrics(filtered[filtered["model_name"] == model]),
                "record_count": int(len(filtered[filtered["model_name"] == model])),
            }
            for model in sorted(filtered["model_name"].dropna().astype(str).unique())
        }

    runs = []
    if not df.empty and "run_id" in df.columns:
        runs = sorted(
            [r for r in df["run_id"].dropna().astype(str).unique().tolist() if r])

    return {
        "metrics": metrics,
        "count": int(len(filtered)),
        "runs": runs,
        "inter_model": inter_model,
        "condition_rows": _condition_metrics_by_model(filtered),
        "top_outlets": _top_outlets_by_model(filtered),
        "run_summaries": _run_summaries(df),
        "records": _sample_records(filtered, limit=record_limit),
    }


@st.cache_data(ttl=10)
def load_llm_summary_file(summary_path: str) -> dict[str, Any] | None:
    path = Path(summary_path)
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    return payload


def _render_bullets(title: str, items: list[str]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {str(item)}")


def _render_insight_cards(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    st.markdown("**Per-Model Strengths and Weaknesses**")
    columns = st.columns(2)
    for idx, item in enumerate(items):
        model = str(item.get("model") or "Unknown model")
        strengths = [str(x) for x in (item.get("strengths") or [])]
        weaknesses = [str(x) for x in (item.get("weaknesses") or [])]
        deployment_fit = str(item.get("deployment_fit") or "")

        with columns[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{model}**")
                if strengths:
                    st.caption("Strengths")
                    for row in strengths:
                        st.markdown(f"- {row}")
                if weaknesses:
                    st.caption("Weaknesses")
                    for row in weaknesses:
                        st.markdown(f"- {row}")
                if deployment_fit:
                    st.caption(f"Deployment fit: {deployment_fit}")


def _render_comparison_table(items: list[dict[str, Any]]) -> None:
    if not items:
        return
    rows: list[dict[str, str]] = []
    for item in items:
        rows.append(
            {
                "Comparison": str(item.get("title") or ""),
                "Winner": str(item.get("winner") or ""),
                "Trade-off": str(item.get("loser_or_tradeoff") or ""),
                "Evidence": str(item.get("evidence") or ""),
                "Takeaway": str(item.get("takeaway") or ""),
            }
        )
    if rows:
        st.markdown("**Model-vs-Model Comparisons**")
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_snapshot_fallbacks(snapshot: dict[str, Any]) -> dict[str, Any]:
    per_model_raw = snapshot.get("per_model") if isinstance(
        snapshot.get("per_model"), list) else []
    per_model = [row for row in per_model_raw if isinstance(row, dict)]

    best_meta = snapshot.get("best_model_by_composite") if isinstance(
        snapshot.get("best_model_by_composite"), dict) else {}
    best_name = str(best_meta.get("model") or "N/A")
    best_score = _safe_float(best_meta.get("composite_score"), 0.0)

    record_count = int(snapshot.get("record_count", 0) or 0)
    model_count = len(snapshot.get("models", []) or [])
    run_count = len(snapshot.get("runs", []) or [])

    fastest = min(per_model, key=lambda r: _safe_float(
        r.get("avg_latency_ms"), 10**12)) if per_model else None
    reliable = max(per_model, key=lambda r: _safe_float(
        r.get("parse_success_rate"), -1.0)) if per_model else None
    robust = min(per_model, key=lambda r: _safe_float(
        r.get("label_sensitivity_rate"), 10**12)) if per_model else None

    fallback_highlights = [
        f"Evaluated {record_count} records across {model_count} models and {run_count} runs.",
        f"Best composite model: {best_name} (score {best_score:.3f}).",
    ]
    if reliable is not None:
        fallback_highlights.append(
            f"Most reliable parser: {str(reliable.get('model') or 'N/A')} with parse success {_safe_float(reliable.get('parse_success_rate')):.2%}."
        )
    if fastest is not None:
        fallback_highlights.append(
            f"Fastest model: {str(fastest.get('model') or 'N/A')} at avg latency {_safe_float(fastest.get('avg_latency_ms')):.0f} ms."
        )

    fallback_recommendations = [
        "Use the composite winner for balanced production behavior.",
        "If latency is critical, prioritize the fastest model and monitor parse reliability.",
        "Track label sensitivity and skew over time to detect drift.",
        "Re-run summary after each major experiment batch.",
    ]

    fallback_risks = [
        "Composite ranking depends on weighting choices.",
        "Bias metrics are proxy signals, not full fairness audits.",
        "Latency depends on hardware and runtime load.",
        "Some qualitative failure modes require manual review.",
    ]

    fallback_insights: list[dict[str, Any]] = []
    for row in per_model:
        model = str(row.get("model") or "unknown")
        parse_rate = _safe_float(row.get("parse_success_rate"))
        latency = _safe_float(row.get("avg_latency_ms"))
        sensitivity = _safe_float(row.get("label_sensitivity_rate"))
        skew = _safe_float(row.get("partisan_skew_score"))
        fallback_insights.append(
            {
                "model": model,
                "strengths": [
                    f"Parse success {parse_rate:.2%}",
                    f"Average latency {latency:.0f} ms",
                ],
                "weaknesses": [
                    f"Label sensitivity {sensitivity:.3f}",
                    f"Partisan skew score {skew:.3f}",
                ],
                "deployment_fit": "Use when this model's latency-reliability profile matches product constraints.",
            }
        )

    fallback_comparisons: list[dict[str, Any]] = []
    if reliable is not None:
        fallback_comparisons.append(
            {
                "title": "Reliability leader",
                "winner": str(reliable.get("model") or "N/A"),
                "loser_or_tradeoff": "Other models have lower parse success",
                "evidence": f"parse_success_rate={_safe_float(reliable.get('parse_success_rate')):.2%}",
                "takeaway": "Use for automation paths where structured parsing robustness matters most.",
            }
        )
    if fastest is not None:
        fallback_comparisons.append(
            {
                "title": "Latency leader",
                "winner": str(fastest.get("model") or "N/A"),
                "loser_or_tradeoff": "Other models are slower",
                "evidence": f"avg_latency_ms={_safe_float(fastest.get('avg_latency_ms')):.0f}",
                "takeaway": "Use for low-latency experiences if reliability remains acceptable.",
            }
        )
    if robust is not None:
        fallback_comparisons.append(
            {
                "title": "Robustness leader",
                "winner": str(robust.get("model") or "N/A"),
                "loser_or_tradeoff": "Other models show higher label sensitivity",
                "evidence": f"label_sensitivity_rate={_safe_float(robust.get('label_sensitivity_rate')):.3f}",
                "takeaway": "Use when robustness to source-label perturbations is a priority.",
            }
        )

    fallback_summary = (
        f"Snapshot covers {record_count} decisions across {model_count} models and {run_count} runs. "
        f"Composite ranking currently selects {best_name} as the best overall trade-off."
    )

    return {
        "best_name": best_name,
        "best_rationale": f"Best by composite score ({best_score:.3f}) from snapshot metrics.",
        "best_tradeoffs": "Use model comparison table below to decide latency vs reliability vs robustness trade-offs.",
        "executive_summary": fallback_summary,
        "metric_highlights": fallback_highlights,
        "recommendations": fallback_recommendations,
        "flaws_and_biases": fallback_risks,
        "confidence_and_caveats": [
            "Snapshot-derived fallback content is deterministic from run artifacts.",
            "Narrative quality improves when LLM summary fields are present.",
        ],
        "per_model_insights": fallback_insights,
        "model_comparisons": fallback_comparisons,
    }


def render_llm_summary_section(summary_path: str) -> None:
    st.subheader("✨ LLM Summary")
    payload = load_llm_summary_file(summary_path)
    if payload is None:
        st.info(
            "No LLM summary file found yet. Generate one offline, then refresh. "
            "Example: uv run python -m app.cli.generate_llm_dashboard_summary "
            "--outputs-dir outputs --model gemma4:latest --summary-json outputs/llm_dashboard_summary.json"
        )
        return

    llm_summary = payload.get("llm_summary") if isinstance(
        payload.get("llm_summary"), dict) else {}
    snapshot = payload.get("snapshot") if isinstance(
        payload.get("snapshot"), dict) else {}
    generator = payload.get("generator") if isinstance(
        payload.get("generator"), dict) else {}
    fallback = _build_snapshot_fallbacks(snapshot)

    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].markdown(
        f"**{str(llm_summary.get('headline') or 'Experiment Intelligence Snapshot')}**")
    header_cols[1].metric("Records", int(snapshot.get("record_count", 0) or 0))
    header_cols[2].metric("Models", len(snapshot.get("models", []) or []))
    header_cols[3].metric("Runs", len(snapshot.get("runs", []) or []))

    best = llm_summary.get("best_model") if isinstance(
        llm_summary.get("best_model"), dict) else {}
    best_name = str(best.get("name") or fallback["best_name"])
    best_rationale = str(best.get("rationale") or fallback["best_rationale"])
    best_tradeoffs = str(best.get("tradeoffs") or fallback["best_tradeoffs"])

    with st.container(border=True):
        st.markdown("**Best Model (LLM Judgement)**")
        st.markdown(f"**{best_name}**")
        if best_rationale:
            st.caption(best_rationale)
        if best_tradeoffs:
            st.caption(f"Trade-offs: {best_tradeoffs}")

    summary_text = str(llm_summary.get("executive_summary")
                       or fallback["executive_summary"]).strip()

    with st.container(border=True):
        st.markdown("**Executive Summary**")
        st.markdown(summary_text)

    top_tab, model_tab, recommendation_tab, risk_tab = st.tabs(
        ["Highlights", "Model Insights", "Recommendations", "Risks and Caveats"]
    )

    with top_tab:
        c1, c2 = st.columns(2)
        with c1:
            highlights = llm_summary.get(
                "metric_highlights") or fallback["metric_highlights"]
            _render_bullets("Metric Highlights", [str(x) for x in highlights])
        with c2:
            comparisons = llm_summary.get(
                "model_comparisons") or fallback["model_comparisons"]
            _render_comparison_table(
                [x for x in comparisons if isinstance(x, dict)])

    with model_tab:
        insights_source = llm_summary.get(
            "per_model_insights") or fallback["per_model_insights"]
        insight_rows = [x for x in insights_source if isinstance(x, dict)]
        _render_insight_cards(insight_rows)

    with recommendation_tab:
        with st.container(border=True):
            recommendations = llm_summary.get(
                "recommendations") or fallback["recommendations"]
            _render_bullets("Actionable Recommendations", [
                            str(x) for x in recommendations])

    with risk_tab:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                flaws = llm_summary.get(
                    "flaws_and_biases") or fallback["flaws_and_biases"]
                _render_bullets("Flaws and Biases", [str(x) for x in flaws])
        with right:
            with st.container(border=True):
                caveats = llm_summary.get(
                    "confidence_and_caveats") or fallback["confidence_and_caveats"]
                _render_bullets(
                    "Confidence and Caveats",
                    [str(x) for x in caveats],
                )

    with st.expander("LLM Summary Metadata", expanded=False):
        st.json(
            {
                "generated_at_utc": payload.get("generated_at_utc"),
                "generator_model": generator.get("model"),
                "source_file": summary_path,
                "latency_ms": payload.get("latency_ms"),
                "best_model_by_composite": snapshot.get("best_model_by_composite"),
            }
        )


def load_manifest(path: str) -> ModelManifest:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return ModelManifest.model_validate(payload)


def run_batch_experiment(
    *,
    input_path: str,
    models_manifest_path: str,
    output_root: str,
    condition_values: list[str],
    max_combinations: int,
    seed: int,
    retries: int,
    ollama_base_url: str,
    shuffle_candidates: bool,
    progress_bar: Any | None = None,
    status_slot: Any | None = None,
    provider: str = "ollama",
    api_keys: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    incidents = [PreparedIncident.model_validate(
        row) for row in read_jsonl(input_path)]
    conditions = [ConditionName(c) for c in condition_values]
    manifest = load_manifest(models_manifest_path)

    # When a commercial provider is selected, only run models whose name begins
    # with the expected LiteLLM provider prefix so that selecting "google" does
    # not accidentally call OpenAI or Anthropic models that may also be present
    # in the same manifest file.
    _PROVIDER_PREFIXES: dict[str, tuple[str, ...]] = {
        "openai": ("openai/",),
        "anthropic": ("anthropic/",),
        "google": ("gemini/", "google/"),
    }
    if provider != "ollama":
        prefixes = _PROVIDER_PREFIXES.get(provider, ())
        if prefixes:
            filtered_models = [m for m in manifest.models if any(
                m.name.startswith(p) for p in prefixes)]
            if not filtered_models:
                raise ValueError(
                    f"No models in the manifest match provider '{provider}'. "
                    f"Expected names starting with: {', '.join(prefixes)}"
                )
            manifest = manifest.model_copy(update={"models": filtered_models})

    if provider == "ollama":
        client = OllamaClient(base_url=ollama_base_url)
    else:
        client = LiteLLMClient(api_keys=api_keys)
    response_schema = selection_response_json_schema()

    prepared_runs: list[tuple[PreparedIncident,
                              dict[ConditionName, list[list[Article]]]]] = []
    for incident in incidents:
        prepared_runs.append(
            (
                incident,
                build_condition_bundles(
                    incident,
                    conditions=conditions,
                    max_combinations=max_combinations,
                    seed=seed,
                    shuffle_candidates=shuffle_candidates,
                ),
            )
        )

    total_requests = sum(
        len(manifest.models) * sum(len(condition_bundles)
                                   for condition_bundles in bundles.values())
        for _, bundles in prepared_runs
    )
    completed = 0
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "experiment_requests.jsonl"
    decisions_path = output_dir / "model_decisions.jsonl"
    raw_outputs_path = output_dir / "raw_outputs.jsonl"
    for path in (requests_path, decisions_path, raw_outputs_path):
        path.touch()

    request_count = 0
    decision_count = 0

    for model in manifest.models:
        if status_slot is not None:
            status_slot.caption(f"Running model {model.name}")
        for incident, bundles in prepared_runs:
            if status_slot is not None:
                status_slot.caption(
                    f"Running model {model.name} on {incident.incident_id}")
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
                    request_record["candidate_order"] = [
                        c.article_id for c in candidates]
                    append_jsonl(requests_path, request_record)
                    request_count += 1

                    try:
                        generation = client.generate(
                            model=model.name,
                            prompt=prompt,
                            temperature=model.temperature,
                            max_tokens=model.max_tokens,
                            timeout_seconds=model.timeout_seconds,
                            retries=retries,
                            think=model.think,
                            response_schema=response_schema,
                        )
                        parsed = parse_model_response(
                            text=generation.text,
                            allowed_article_ids={
                                c.article_id for c in candidates},
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

                    append_jsonl(decisions_path,
                                 decision.model_dump(mode="json"))
                    decision_count += 1
                    completed += 1
                    if progress_bar is not None and total_requests:
                        progress_bar.progress(
                            completed / total_requests, text=f"{completed}/{total_requests} requests")

    return {
        "run_id": run_id,
        "incident_count": len(incidents),
        "request_count": request_count,
        "decision_count": decision_count,
        "output_dir": str(output_dir),
    }


def metrics_cards(metrics: dict[str, Any], total_count: int) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Records", total_count)
    c2.metric("Parse Success", f"{metrics.get('parse_success_rate', 0.0):.1%}")
    c3.metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0.0):.0f} ms")
    c4.metric("P95 Latency", f"{metrics.get('p95_latency_ms', 0.0):.0f} ms")
    c5.metric("Center Preference",
              f"{metrics.get('center_preference_index', 0.0):.3f}")
    c6.metric("Robustness",
              f"{metrics.get('content_robustness_score', 0.0):.3f}")


def takeaway_panel(metrics: dict[str, Any]) -> None:
    bias_sensitivity_index = float(metrics.get(
        "label_sensitivity_rate", metrics.get("identity_dominance_rate", 0.0)))
    reliability_index = float(metrics.get("parse_success_rate", 0.0))
    avg_latency = float(metrics.get("avg_latency_ms", 0.0))
    speed_index = 1.0 / (1.0 + max(0.0, avg_latency) / 1000.0)

    st.subheader("Takeaway Panel")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bias Sensitivity Index ⬇", f"{bias_sensitivity_index:.3f}")
    c2.metric("Reliability Index ⬆", f"{reliability_index:.3f}")
    c3.metric("Speed Index ⬆", f"{speed_index:.3f}")

    with st.expander("How these 3 are computed", expanded=False):
        st.markdown(
            "- **Bias Sensitivity Index ⬇** = `label_sensitivity_rate` (fallback to `identity_dominance_rate`).\n"
            "  - Formula: `mean(bucket_headlines_sources != bucket_swapped_sources)`\n"
            "  - Interpretation: lower is better (less sensitive to source-label swaps).\n"
            "- **Reliability Index ⬆** = `parse_success_rate`\n"
            "  - Formula: `count(parse_status = success) / total`\n"
            "  - Interpretation: higher is better.\n"
            "- **Speed Index ⬆** = `1 / (1 + avg_latency_ms / 1000)`\n"
            "  - Monotonic transform of average latency into (0,1].\n"
            "  - Interpretation: higher is faster; this is not a standard benchmark metric, just a normalized dashboard score."
        )


def render_distribution_charts(metrics: dict[str, Any]) -> None:
    selection_distribution = metrics.get("selection_distribution", {})
    if selection_distribution:
        dist_df = pd.DataFrame(
            {
                "bucket": list(selection_distribution.keys()),
                "ratio": list(selection_distribution.values()),
            }
        )
        fig = px.bar(
            dist_df,
            x="bucket",
            y="ratio",
            title="Selection Distribution",
            color="bucket",
            color_discrete_sequence=["#3366CC",
                                     "#109618", "#DC3912", "#FF9900"],
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    position_distribution = metrics.get("selected_position_distribution", {})
    if position_distribution:
        pos_df = pd.DataFrame(
            {
                "position": [int(k) for k in position_distribution.keys()],
                "ratio": list(position_distribution.values()),
            }
        ).sort_values("position")
        fig = px.bar(
            pos_df,
            x="position",
            y="ratio",
            title="Selected Position Distribution",
            color_discrete_sequence=["#FF9900"],
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)


def render_inter_model_overview(inter_model_data: dict[str, Any]) -> None:
    if not inter_model_data:
        st.info("No inter-model data available yet.")
        return

    rows: list[dict[str, Any]] = []
    for model_name, metrics in inter_model_data.items():
        rows.append(
            {
                "model": model_name,
                "records": metrics.get("record_count", 0),
                "parse_success_rate": metrics.get("parse_success_rate", 0.0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0.0),
                "p95_latency_ms": metrics.get("p95_latency_ms", 0.0),
                "center_preference_index": metrics.get("center_preference_index", 0.0),
                "partisan_skew_score": metrics.get("partisan_skew_score", 0.0),
                "selection_stability_score": metrics.get("selection_stability_score", 0.0),
                "identity_dominance_rate": metrics.get("identity_dominance_rate", 0.0),
                "content_robustness_score": metrics.get("content_robustness_score", 0.0),
                "label_sensitivity_rate": metrics.get("label_sensitivity_rate", 0.0),
                "cross_model_agreement_rate": metrics.get("cross_model_agreement_rate", 0.0),
                "cross_model_instability": metrics.get("cross_model_instability", 0.0),
                "model_instability_score": metrics.get("model_instability_score", 0.0),
            }
        )

    inter_df = pd.DataFrame(rows).sort_values("model")
    st.subheader("Model Comparison Table")
    st.dataframe(inter_df, use_container_width=True, hide_index=True)

    card_cols = st.columns(max(1, len(inter_df)))
    for idx, row in inter_df.reset_index(drop=True).iterrows():
        with card_cols[idx]:
            st.markdown(f"**{row['model']}**")
            st.metric("Parse success", f"{row['parse_success_rate']:.1%}")
            st.metric("Avg latency", f"{row['avg_latency_ms']:.0f} ms")
            st.metric("Robustness", f"{row['content_robustness_score']:.3f}")

    objective_options = {
        "Most reliable": ("parse_success_rate", False),
        "Fastest": ("avg_latency_ms", True),
        "Least label-sensitive": ("label_sensitivity_rate", True),
        "Most robust": ("content_robustness_score", False),
        "Most stable": ("model_instability_score", True),
    }
    objective = st.selectbox(
        "Best model objective",
        options=list(objective_options.keys()),
        key="best_model_objective",
    )
    sort_col, ascending = objective_options[objective]
    ranking_columns = ["model", sort_col, "parse_success_rate",
                       "avg_latency_ms", "content_robustness_score"]
    # Keep stable order while removing duplicates to avoid non-unique label errors in sort_values.
    ranking_columns = list(dict.fromkeys(ranking_columns))
    ranking = inter_df[ranking_columns].sort_values(
        sort_col, ascending=ascending)
    st.markdown("**Objective Ranking**")
    st.dataframe(ranking, use_container_width=True, hide_index=True)


def _overlap_score(incident_id: str, reason: str) -> float:
    incident_tokens = [t for t in re.split(
        r"[_\W]+", str(incident_id).lower()) if len(t) >= 4]
    if not incident_tokens:
        return 0.0
    reason_tokens = set(re.findall(r"[a-zA-Z]{4,}", str(reason).lower()))
    if not reason_tokens:
        return 0.0
    overlap = sum(1 for t in incident_tokens if t in reason_tokens)
    return float(overlap / len(incident_tokens))


def render_scenario_simulator(selected_run: str) -> None:
    df = load_db()
    df = _apply_filters(df, run_id=selected_run or None)
    if df.empty:
        st.info("No data available for scenario simulation.")
        return

    st.subheader("Scenario Simulator")
    incidents = sorted(
        df["incident_id"].dropna().astype(str).unique().tolist())
    incident_id = st.selectbox(
        "Incident", options=incidents, key="scenario_incident")
    incident_df = df[df["incident_id"] == incident_id].copy()
    conditions = sorted(
        incident_df["condition"].dropna().astype(str).unique().tolist())
    condition = st.selectbox(
        "Condition", options=conditions, key="scenario_condition")
    scoped = incident_df[incident_df["condition"] == condition].copy()

    if scoped.empty:
        st.info("No records for selected scenario.")
        return

    dist = (
        scoped[scoped["selected_bucket"].isin(["left", "center", "right"])]
        .groupby(["model_name", "selected_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if not dist.empty:
        fig = px.bar(
            dist,
            x="model_name",
            y="count",
            color="selected_bucket",
            barmode="group",
            title="Selected Leaning by Model for Scenario",
            color_discrete_map={"left": "#d62728",
                                "center": "#2ca02c", "right": "#1f77b4"},
        )
        st.plotly_chart(fig, use_container_width=True)

    explain = scoped[[
        "model_name",
        "selected_article_id",
        "selected_outlet",
        "selected_bucket",
        "justification",
        "parse_status",
        "latency_ms",
    ]].copy()
    explain["reason_excerpt"] = explain["justification"].fillna(
        "").astype(str).str.slice(0, 180)
    explain["keyword_overlap"] = explain.apply(
        lambda row: _overlap_score(incident_id, row.get("justification", "")),
        axis=1,
    )
    explain = explain.drop(columns=["justification"])
    st.markdown("**Explainability Snippets**")
    st.dataframe(explain, use_container_width=True, hide_index=True)


def render_metrics_explanations() -> None:
    st.subheader("Metrics Explanations")
    st.caption(
        "Short, precise definitions for every metric shown in this dashboard.")
    st.caption(
        "Direction legend: ⬆ means higher is better, ⬇ means lower is better.")

    core_rows = [
        {
            "Metric": "Bias Sensitivity Index ⬇",
            "Definition": "Headline takeaway score for sensitivity to source-label swaps.",
            "Formula": "mean(bucket_headlines_sources != bucket_swapped_sources)",
            "Better": "⬇ lower is better.",
        },
        {
            "Metric": "Reliability Index ⬆",
            "Definition": "Headline takeaway score for strict parse reliability.",
            "Formula": "count(parse_status=success) / total",
            "Better": "⬆ higher is better.",
        },
        {
            "Metric": "Speed Index ⬆",
            "Definition": "Headline takeaway score derived from average latency.",
            "Formula": "1 / (1 + avg_latency_ms / 1000)",
            "Better": "⬆ higher is better.",
        },
        {
            "Metric": "selection_distribution",
            "Definition": "Normalized share of selected buckets among {left, center, right}.",
            "Formula": "count(bucket) / total_known_bucket_selections",
            "Better": "N/A: no universal best; context-dependent.",
        },
        {
            "Metric": "center_preference_index",
            "Definition": "How often the model selects center-labeled sources.",
            "Formula": "P(selected_bucket = center)",
            "Better": "N/A: task-dependent; compare to baseline.",
        },
        {
            "Metric": "partisan_skew_score",
            "Definition": "Directional skew toward left vs right selections.",
            "Formula": "P(left) - P(right)",
            "Better": "N/A: closer to 0 means less directional skew.",
        },
        {
            "Metric": "selection_stability_score ⬆",
            "Definition": "Consistency of selected article between headlines_only and headlines_with_sources.",
            "Formula": "mean(selected_id_headlines_only == selected_id_headlines_sources)",
            "Better": "UP (^): higher is better.",
        },
        {
            "Metric": "identity_dominance_rate ⬇",
            "Definition": "Sensitivity to source-label manipulation.",
            "Formula": "mean(selected_id_headlines_sources != selected_id_swapped_sources)",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "content_robustness_score ⬆",
            "Definition": "Robustness to swapped source labels.",
            "Formula": "1 - identity_dominance_rate",
            "Better": "UP (^): higher is better.",
        },
        {
            "Metric": "label_sensitivity_rate ⬇",
            "Definition": "Bucket-level change rate under source label swap.",
            "Formula": "mean(bucket_headlines_sources != bucket_swapped_sources)",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "cross_model_agreement_rate ⬆",
            "Definition": "How much models agree on one bucket for same incident-condition.",
            "Formula": "mean(max_bucket_probability over incident-condition groups)",
            "Better": "UP (^): higher is better.",
        },
        {
            "Metric": "cross_model_instability ⬇",
            "Definition": "Entropy-based disagreement across models for same incident-condition.",
            "Formula": "mean(H(bucket_dist)/log2(3))",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "model_instability_score ⬇",
            "Definition": "Within-model variation of selected bucket across conditions per incident.",
            "Formula": "mean((unique_buckets-1)/(unique_conditions-1))",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "selected_position_distribution",
            "Definition": "How often each candidate position is selected.",
            "Formula": "count(position) / total_with_known_position",
            "Better": "N/A: balanced shape usually preferred.",
        },
        {
            "Metric": "parse_success_rate ⬆",
            "Definition": "Structured parsing success rate.",
            "Formula": "count(parse_status=success) / total",
            "Better": "UP (^): higher is better.",
        },
        {
            "Metric": "parse_fallback_rate ⬇",
            "Definition": "Rate of non-strict parsing recovered via fallback logic.",
            "Formula": "count(parse_status=fallback) / total",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "parse_failure_rate ⬇",
            "Definition": "Rate of unrecoverable parsing failure.",
            "Formula": "count(parse_status=failed) / total",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "unknown_bucket_rate ⬇",
            "Definition": "Selections whose bucket could not be mapped to left/center/right.",
            "Formula": "count(selected_bucket=unknown) / total",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "avg_latency_ms ⬇",
            "Definition": "Mean response latency in milliseconds.",
            "Formula": "mean(latency_ms)",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "p50_latency_ms ⬇",
            "Definition": "Median latency.",
            "Formula": "quantile(latency_ms, 0.50)",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "p95_latency_ms ⬇",
            "Definition": "Tail latency (95th percentile).",
            "Formula": "quantile(latency_ms, 0.95)",
            "Better": "DOWN (v): lower is better.",
        },
    ]
    st.markdown("**Core Metrics**")
    st.dataframe(pd.DataFrame(core_rows),
                 use_container_width=True, hide_index=True)

    condition_rows = [
        {
            "Metric": "count ⬆",
            "Definition": "Number of records in model-condition slice.",
            "Formula": "len(slice)",
            "Better": "UP (^): higher gives more stable estimates.",
        },
        {
            "Metric": "left_ratio / center_ratio / right_ratio",
            "Definition": "Bucket distribution within a condition/model.",
            "Formula": "count(bucket)/count(known_bucket)",
            "Better": "N/A: no universal best; compare patterns.",
        },
        {
            "Metric": "unknown_ratio ⬇",
            "Definition": "Unknown bucket share within condition/model.",
            "Formula": "count(unknown)/count(total)",
            "Better": "DOWN (v): lower is better.",
        },
        {
            "Metric": "mean_selected_position",
            "Definition": "Average selected candidate index.",
            "Formula": "mean(selected_position)",
            "Better": "N/A: task-dependent; watch for position bias.",
        },
    ]
    st.markdown("**Condition-Level Metrics**")
    st.dataframe(pd.DataFrame(condition_rows),
                 use_container_width=True, hide_index=True)

    run_rows = [
        {
            "Metric": "records ⬆",
            "Definition": "Total decisions in selected scope.",
            "Formula": "len(filtered_df)",
            "Better": "UP (^): higher gives tighter confidence.",
        },
        {
            "Metric": "record_count (inter-model) ⬆",
            "Definition": "Per-model decision count for comparison cards.",
            "Formula": "len(df[df.model_name == model])",
            "Better": "UP (^): higher gives more stable model estimates.",
        },
        {
            "Metric": "runs / incidents / models ⬆",
            "Definition": "Coverage counters in summaries.",
            "Formula": "nunique(run_id/incident_id/model_name)",
            "Better": "UP (^): higher coverage improves representativeness.",
        },
    ]
    st.markdown("**Coverage / Summary Metrics**")
    st.dataframe(pd.DataFrame(run_rows),
                 use_container_width=True, hide_index=True)

    st.markdown("**Condition Labels Used Internally**")
    st.markdown(
        "- headlines_only\n"
        "- headlines_with_sources (aliased internally to headlines_sources in some backend calculations)\n"
        "- sources_only\n"
        "- headlines_with_manipulated_sources (aliased internally to swapped_sources in some backend calculations)"
    )


def render_condition_metrics_by_model(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No condition-by-model metrics available yet.")
        return

    df = pd.DataFrame(rows)
    st.subheader("Condition by Model")
    st.dataframe(df, use_container_width=True, hide_index=True)

    condition_order = [
        "headlines_only",
        "headlines_with_sources",
        "sources_only",
        "headlines_with_manipulated_sources",
    ]
    bucket_order = ["left", "center", "right"]
    condition_labels = {
        "headlines_only": "Headlines",
        "headlines_with_sources": "+ Sources",
        "sources_only": "Sources only",
        "headlines_with_manipulated_sources": "Swapped sources",
    }
    bucket_labels = {
        "left": "Left",
        "center": "Center",
        "right": "Right",
    }

    heatmap_df = df.melt(
        id_vars=["model", "condition"],
        value_vars=["left_ratio", "center_ratio", "right_ratio"],
        var_name="bucket",
        value_name="ratio",
    )
    heatmap_df["bucket"] = heatmap_df["bucket"].str.replace(
        "_ratio", "", regex=False)
    heatmap_df["condition_label"] = heatmap_df["condition"].map(
        condition_labels)
    heatmap_df["bucket_label"] = heatmap_df["bucket"].map(bucket_labels)

    st.markdown("**Overview Grid**")
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    grid_cols = st.columns(2)
    for idx, model_name in enumerate(models):
        model_heatmap = heatmap_df[heatmap_df["model"] == model_name].copy()
        pivot = (
            model_heatmap
            .pivot(index="condition_label", columns="bucket_label", values="ratio")
            .reindex(
                index=[condition_labels[c] for c in condition_order],
                columns=[bucket_labels[b] for b in bucket_order],
            )
            .fillna(0.0)
        )
        with grid_cols[idx % 2]:
            fig = px.imshow(
                pivot,
                text_auto=".0%",
                color_continuous_scale="Blues",
                zmin=0.0,
                zmax=1.0,
                aspect="auto",
                title=model_name,
            )
            fig.update_layout(
                height=320,
                margin=dict(l=8, r=8, t=42, b=8),
                coloraxis_showscale=False,
            )
            fig.update_xaxes(title=None, side="bottom", tickangle=0)
            fig.update_yaxes(title=None, automargin=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Focused View**")
    selected_model = st.select_slider(
        "Choose model",
        options=models,
        value=models[0] if models else None,
    )
    focused_df = df[df["model"] == selected_model].copy()
    focused_heatmap = heatmap_df[heatmap_df["model"] == selected_model].copy()
    focused_pivot = (
        focused_heatmap
        .pivot(index="condition_label", columns="bucket_label", values="ratio")
        .reindex(
            index=[condition_labels[c] for c in condition_order],
            columns=[bucket_labels[b] for b in bucket_order],
        )
        .fillna(0.0)
    )

    heatmap_tab, distribution_tab, quality_tab = st.tabs(
        ["Heatmap", "Distribution", "Quality"])
    with heatmap_tab:
        fig = px.imshow(
            focused_pivot,
            text_auto=".0%",
            color_continuous_scale="Blues",
            zmin=0.0,
            zmax=1.0,
            aspect="auto",
            title=f"Selected Leaning Heatmap: {selected_model}",
        )
        fig.update_layout(
            height=430,
            margin=dict(l=120, r=10, t=48, b=10),
            coloraxis_colorbar_title="ratio",
        )
        fig.update_xaxes(title=None, side="bottom", tickangle=0)
        fig.update_yaxes(title="Condition", automargin=True,
                         tickfont=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

    with distribution_tab:
        dist_fig = px.bar(
            focused_heatmap,
            x="condition_label",
            y="ratio",
            color="bucket_label",
            barmode="group",
            category_orders={
                "condition_label": [condition_labels[c] for c in condition_order],
                "bucket_label": [bucket_labels[b] for b in bucket_order],
            },
            title=f"Selection Distribution by Condition: {selected_model}",
        )
        dist_fig.update_layout(yaxis_tickformat=".0%", height=430)
        st.plotly_chart(dist_fig, use_container_width=True)

    with quality_tab:
        quality_df = focused_df[
            ["condition", "parse_success_rate", "avg_latency_ms",
                "p95_latency_ms", "unknown_ratio"]
        ].copy()
        quality_df["parse_success_rate"] = quality_df["parse_success_rate"].fillna(
            0.0)
        quality_df["unknown_ratio"] = quality_df["unknown_ratio"].fillna(0.0)
        quality_df = quality_df.set_index(
            "condition").reindex(condition_order).reset_index()
        quality_df["condition_label"] = quality_df["condition"].map(
            condition_labels)

        qcol1, qcol2 = st.columns(2)
        with qcol1:
            parse_fig = px.bar(
                quality_df,
                x="condition_label",
                y="parse_success_rate",
                title=f"Parse Success by Condition: {selected_model}",
            )
            parse_fig.update_layout(yaxis_tickformat=".0%", height=360)
            st.plotly_chart(parse_fig, use_container_width=True)
        with qcol2:
            latency_fig = px.bar(
                quality_df,
                x="condition_label",
                y="avg_latency_ms",
                title=f"Average Latency by Condition: {selected_model}",
            )
            latency_fig.update_layout(height=360)
            st.plotly_chart(latency_fig, use_container_width=True)

        unknown_fig = px.bar(
            quality_df,
            x="condition_label",
            y="unknown_ratio",
            title=f"Unknown Ratio by Condition: {selected_model}",
        )
        unknown_fig.update_layout(yaxis_tickformat=".0%", height=300)
        st.plotly_chart(unknown_fig, use_container_width=True)

    center_fig = px.bar(
        df,
        x=df["condition"].map(condition_labels),
        y="center_ratio",
        color="model",
        barmode="group",
        title="Center Selection Ratio by Condition and Model",
    )
    center_fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(center_fig, use_container_width=True)

    parse_fig = px.bar(
        df,
        x=df["condition"].map(condition_labels),
        y="parse_success_rate",
        color="model",
        barmode="group",
        title="Parse Success Rate by Condition and Model",
    )
    parse_fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(parse_fig, use_container_width=True)

    latency_fig = px.bar(
        df,
        x=df["condition"].map(condition_labels),
        y="avg_latency_ms",
        color="model",
        barmode="group",
        title="Average Latency by Condition and Model",
    )
    st.plotly_chart(latency_fig, use_container_width=True)

    lean_fig = px.bar(
        heatmap_df,
        x="condition_label",
        y="ratio",
        color="bucket_label",
        facet_col="model",
        barmode="group",
        title="Selection Distribution by Condition, Bucket, and Model",
    )
    lean_fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(lean_fig, use_container_width=True)


def render_top_outlets(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No outlet selections available yet.")
        return
    outlet_df = pd.DataFrame(rows)
    st.subheader("Top Selected Outlets by Model")
    st.dataframe(outlet_df, use_container_width=True, hide_index=True)

    models = sorted(outlet_df["model"].dropna().astype(str).unique().tolist())
    selected_model = st.selectbox(
        "Outlet chart model",
        options=models,
        key="top_outlets_model",
    )

    filtered_df = (
        outlet_df[outlet_df["model"] == selected_model]
        .sort_values("count", ascending=True)
        .copy()
    )

    fig = px.bar(
        filtered_df,
        x="count",
        y="selected_outlet",
        orientation="h",
        title=f"Top Selected Outlets: {selected_model}",
        color_discrete_sequence=["#3366CC"],
        text="count",
    )
    fig.update_layout(
        showlegend=False,
        height=max(320, 34 * len(filtered_df) + 140),
        margin=dict(l=8, r=8, t=56, b=8),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def render_run_summaries(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No run summaries available yet.")
        return
    run_df = pd.DataFrame(rows)
    st.subheader("Run Summaries")
    st.dataframe(run_df, use_container_width=True, hide_index=True)


def render_sample_records(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No records available for inspection.")
        return
    records_df = pd.DataFrame(rows)
    st.subheader("Recent Records")
    st.dataframe(records_df, use_container_width=True, hide_index=True)


def render_inter_model(inter_model_data: dict[str, Any]) -> None:
    if not inter_model_data:
        st.info("No inter-model data available yet.")
        return

    rows: list[dict[str, Any]] = []
    for model_name, metrics in inter_model_data.items():
        rows.append(
            {
                "model": model_name,
                "center_preference_index": metrics.get("center_preference_index", 0.0),
                "partisan_skew_score": metrics.get("partisan_skew_score", 0.0),
                "selection_stability_score": metrics.get("selection_stability_score", 0.0),
                "identity_dominance_rate": metrics.get("identity_dominance_rate", 0.0),
                "content_robustness_score": metrics.get("content_robustness_score", 0.0),
                "parse_success_rate": metrics.get("parse_success_rate", 0.0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0.0),
            }
        )

    inter_df = pd.DataFrame(rows).sort_values("model")
    st.subheader("Inter-Model Plots")
    st.dataframe(inter_df, use_container_width=True, hide_index=True)

    fig1 = px.bar(
        inter_df,
        x="model",
        y="center_preference_index",
        title="Center Preference by Model",
        color="model",
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        inter_df,
        x="model",
        y="content_robustness_score",
        title="Content Robustness by Model",
        color="model",
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        inter_df,
        x="model",
        y="avg_latency_ms",
        title="Average Latency by Model",
        color="model",
    )
    st.plotly_chart(fig3, use_container_width=True)


st.set_page_config(page_title="Sourcerers Dashboard", layout="wide")
st.title("Sourcerers Analytics Dashboard")

with st.sidebar:
    st.header("Settings")
    ollama_base = st.text_input(
        "Ollama Base URL", value=DEFAULT_OLLAMA_BASE).rstrip("/")
    selected_run = st.text_input("Run filter (optional)", value="")
    st.caption(
        "Analytics mode: embedded in Streamlit using the same backend logic as the FastAPI app.")
    llm_summary_path = st.text_input(
        "LLM summary file", value="outputs/llm_dashboard_summary.json")

    st.divider()
    st.subheader("Commercial Model API Keys")
    with st.expander("Configure API keys for commercial LLMs", expanded=False):
        st.caption("Keys are stored in memory only and never written to disk.")
        _openai_key = st.text_input(
            "OpenAI API Key", type="password", key="openai_api_key", value="")
        _anthropic_key = st.text_input(
            "Anthropic API Key", type="password", key="anthropic_api_key", value="")
        _google_key = st.text_input(
            "Google (Gemini) API Key", type="password", key="google_api_key", value="")
        _configured_providers = [p for p, k in {
            "openai": _openai_key, "anthropic": _anthropic_key, "google": _google_key}.items() if k]
        if _configured_providers:
            st.caption(f"Configured: {', '.join(_configured_providers)}")
        else:
            st.caption("No commercial API keys configured.")

    def _get_api_keys() -> dict[str, str]:
        return {
            "openai": st.session_state.get("openai_api_key", ""),
            "anthropic": st.session_state.get("anthropic_api_key", ""),
            "google": st.session_state.get("google_api_key", ""),
        }

    st.divider()
    st.subheader("Bulk Ingest")
    outputs_dir = st.text_input("Outputs root", value=DEFAULT_OUTPUTS_DIR)
    if st.button("Ingest all runs in outputs"):
        try:
            ingest_resp = sync_outputs_to_db(
                outputs_dir=Path(outputs_dir.strip() or DEFAULT_OUTPUTS_DIR),
                db_file=DB_FILE,
            )
            st.success(
                f"Runs ingested: {ingest_resp.get('runs_ingested', 0)} | "
                f"Records added: {ingest_resp.get('records_added', 0)}"
            )
            st.cache_data.clear()
        except Exception as exc:
            st.error(f"Bulk ingest failed: {exc}")

    st.subheader("Single Run Ingest")
    run_dir = st.text_input("Run directory path", value="")
    if st.button("Ingest one run"):
        if not run_dir.strip():
            st.warning("Provide a run directory path first.")
        else:
            try:
                ingest_result = _ingest_run_directory(
                    run_dir=Path(run_dir.strip()),
                    model_decisions_file="model_decisions.jsonl",
                    experiment_requests_file="experiment_requests.jsonl",
                    db_file=DB_FILE,
                )
                ingest_resp = {
                    "ingested": ingest_result["records_seen"],
                    "added": ingest_result["records_added"],
                }
                st.success(
                    f"Ingested {ingest_resp.get('ingested', 0)} records | "
                    f"Added {ingest_resp.get('added', 0)}"
                )
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Single-run ingest failed: {exc}")

    st.divider()
    st.subheader("API Surface")
    st.code("uv run uvicorn app.api.engine_analytics:app --host 0.0.0.0 --port 8000", language="bash")
    st.caption(
        "The FastAPI app stays in the repo for local demos, client integrations, and presentation samples.")

render_llm_summary_section(llm_summary_path)

analytics_tab, runner_tab, explainer_tab = st.tabs(
    ["Analytics", "Run Models", "Metrics explanations"])

with analytics_tab:
    render_filter_scope(selected_run, label="Analytics scope")
    with st.expander("FastAPI sample usage", expanded=False):
        st.code(
            """
curl http://127.0.0.1:8000/metrics/inter-model

curl "http://127.0.0.1:8000/metrics/conditions-by-model?run_id=run_20260414_103158"

curl -X POST http://127.0.0.1:8000/ingest/runs \\
  -H "Content-Type: application/json" \\
  -d '{"outputs_dir":"outputs"}'
            """.strip(),
            language="bash",
        )

    try:
        snapshot = get_dashboard_snapshot(
            selected_run=selected_run, record_limit=50)
    except Exception as exc:
        st.error(f"Could not load analytics snapshot: {exc}")
        snapshot = None

    if snapshot is not None:
        metrics_cards(snapshot["metrics"], snapshot["count"])
        takeaway_panel(snapshot["metrics"])
        st.subheader("Overview Distributions")
        render_distribution_charts(snapshot["metrics"])
        with st.expander("Detailed metrics snapshot", expanded=False):
            st.json(snapshot["metrics"])

        st.subheader("Runs")
        if snapshot["runs"]:
            st.code("\n".join(snapshot["runs"]))
        else:
            st.info("No run IDs found in analytics database yet.")

        render_filter_scope(selected_run, label="Inter-model scope")
        inter_model = snapshot["inter_model"]
        if inter_model:
            render_inter_model_overview(inter_model)
            render_inter_model(inter_model)
        else:
            st.info("Inter-model metrics are not available yet.")

        render_filter_scope(selected_run, label="Condition comparison scope")
        render_condition_metrics_by_model(snapshot["condition_rows"])

        col1, col2 = st.columns(2)
        with col1:
            render_filter_scope(selected_run, label="Outlet metrics scope")
            render_top_outlets(snapshot["top_outlets"])
        with col2:
            render_filter_scope(selected_run, label="Run summary scope")
            run_rows = snapshot["run_summaries"]
            if selected_run:
                run_rows = [row for row in run_rows if row.get(
                    "run_id") == selected_run]
            render_run_summaries(run_rows)

        render_filter_scope(selected_run, label="Recent records scope")
        render_sample_records(snapshot["records"])
        render_filter_scope(selected_run, label="Scenario simulation scope")
        render_scenario_simulator(selected_run)

with runner_tab:
    if IS_STREAMLIT_CLOUD:
        st.info("Community Cloud is configured for the analytics dashboard. Model-running tabs are intended for local use with Ollama.")
    probe_tab, batch_tab = st.tabs(["Probe Model", "Batch Experiment"])

    with probe_tab:
        st.write(
            "Run a single prompt against a local Ollama model or a commercial LLM.")
        probe_provider = st.selectbox(
            "Provider",
            options=["ollama", "openai", "anthropic", "google"],
            index=0,
            key="probe_provider",
        )

        if probe_provider == "ollama":
            probe_model = st.text_input(
                "Model name", value="qwen2.5:7b", key="probe_model_name")
        else:
            _api_keys = _get_api_keys()
            _provider_key = _api_keys.get(probe_provider, "")
            if not _provider_key:
                st.warning(
                    f"No API key configured for **{probe_provider}**. Add one in the sidebar under *Commercial Model API Keys*.")
            available_models = PROVIDER_MODELS.get(probe_provider, [])
            probe_model = st.selectbox(
                "Model",
                options=available_models,
                index=0 if available_models else 0,
                key="probe_model_select",
            )

        probe_prompt = st.text_area(
            "Prompt",
            value="Return strict JSON with selected_article_id and reason.",
            height=140,
        )
        probe_temperature = st.number_input(
            "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        probe_max_tokens = st.number_input(
            "Max tokens", min_value=1, max_value=4096, value=300, step=50)
        probe_timeout = st.number_input(
            "Timeout seconds", min_value=1, max_value=300, value=60, step=5)

        if st.button("Run probe"):
            if probe_provider != "ollama":
                _api_keys = _get_api_keys()
                _provider_key = _api_keys.get(probe_provider, "")
                if not _provider_key:
                    st.error(
                        f"Cannot run probe: no API key for **{probe_provider}**.")
                    st.stop()
            try:
                spinner_label = f"Calling {probe_provider}..."
                with st.spinner(spinner_label):
                    if probe_provider == "ollama":
                        generation = OllamaClient(base_url=ollama_base).generate(
                            model=probe_model,
                            prompt=probe_prompt,
                            temperature=float(probe_temperature),
                            max_tokens=int(probe_max_tokens),
                            timeout_seconds=int(probe_timeout),
                        )
                    else:
                        generation = LiteLLMClient(api_keys=_get_api_keys()).generate(
                            model=probe_model,
                            prompt=probe_prompt,
                            temperature=float(probe_temperature),
                            max_tokens=int(probe_max_tokens),
                            timeout_seconds=int(probe_timeout),
                        )
                st.success(f"Latency: {generation.latency_ms} ms")
                st.text_area("Model response",
                             value=generation.text, height=220)
                st.expander("Raw payload").json(generation.raw_payload)
            except Exception as exc:
                st.error(f"Probe failed: {exc}")

    with batch_tab:
        st.write(
            "Run the full experiment pipeline using existing builders, parser, and schemas.")
        batch_provider = st.selectbox(
            "Provider",
            options=["ollama", "openai", "anthropic", "google"],
            index=0,
            key="batch_provider",
        )
        if batch_provider != "ollama":
            _batch_api_keys = _get_api_keys()
            _batch_provider_key = _batch_api_keys.get(batch_provider, "")
            if not _batch_provider_key:
                st.warning(
                    f"No API key configured for **{batch_provider}**. Add one in the sidebar under *Commercial Model API Keys*.")
            _example_models = PROVIDER_MODELS.get(batch_provider, [""])
            st.info(
                f"Provider **{batch_provider}** selected. Only models in the manifest whose "
                f"`name:` starts with the matching prefix will be run — other providers in the "
                f"same YAML are automatically skipped.\n\n"
                f"Supported examples for {batch_provider}: "
                + ", ".join(f"`{m}`" for m in _example_models)
            )

        input_path = st.text_input(
            "Input incidents JSONL", value="data/real_incidents_all.jsonl")
        manifest_path = st.text_input(
            "Models manifest YAML", value="configs/models.example.yaml")
        output_root = st.text_input("Output root", value="outputs")

        condition_values = st.multiselect(
            "Conditions",
            options=[c.value for c in ConditionName],
            default=[c.value for c in ConditionName],
        )
        max_combinations = st.number_input(
            "Max combinations", min_value=1, max_value=50, value=3, step=1)
        seed = st.number_input("Seed", min_value=0,
                               max_value=1_000_000, value=42, step=1)
        retries = st.number_input(
            "Retries", min_value=0, max_value=10, value=2, step=1)
        shuffle_candidates = st.checkbox("Shuffle candidate order", value=True)
        auto_ingest = st.checkbox(
            "Auto-ingest run into embedded analytics DB", value=True)

        if st.button("Run batch experiment"):
            if not condition_values:
                st.warning("Select at least one condition.")
            elif batch_provider != "ollama" and not _get_api_keys().get(batch_provider, ""):
                st.error(
                    f"Cannot run batch: no API key for **{batch_provider}**.")
            else:
                try:
                    progress_bar = st.progress(0.0, text="Starting run")
                    status_slot = st.empty()
                    with st.spinner("Running experiment. This can take a while..."):
                        summary = run_batch_experiment(
                            input_path=input_path,
                            models_manifest_path=manifest_path,
                            output_root=output_root,
                            condition_values=condition_values,
                            max_combinations=int(max_combinations),
                            seed=int(seed),
                            retries=int(retries),
                            ollama_base_url=ollama_base,
                            shuffle_candidates=shuffle_candidates,
                            progress_bar=progress_bar,
                            status_slot=status_slot,
                            provider=batch_provider,
                            api_keys=_get_api_keys(),
                        )
                    progress_bar.progress(1.0, text="Completed")
                    status_slot.caption(f"Run finished: {summary['run_id']}")
                    st.success("Batch run completed")
                    st.json(summary)

                    if auto_ingest:
                        try:
                            ingest_result = _ingest_run_directory(
                                run_dir=Path(summary["output_dir"]),
                                model_decisions_file="model_decisions.jsonl",
                                experiment_requests_file="experiment_requests.jsonl",
                                db_file=DB_FILE,
                            )
                            ingest_resp = {
                                "ingested": ingest_result["records_seen"],
                                "added": ingest_result["records_added"],
                            }
                            st.success(
                                f"Auto-ingest complete: {ingest_resp.get('ingested', 0)} records "
                                f"(added {ingest_resp.get('added', 0)})"
                            )
                            st.cache_data.clear()
                        except Exception as exc:
                            st.warning(
                                f"Run completed but auto-ingest failed: {exc}")
                except Exception as exc:
                    st.error(f"Batch run failed: {exc}")

with explainer_tab:
    render_metrics_explanations()
