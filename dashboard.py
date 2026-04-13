from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yaml

from app.experiment.condition_builder import build_condition_bundles
from app.experiment.prompt_builder import build_selection_prompt
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

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_OLLAMA_BASE = "http://localhost:11434"


@st.cache_data(ttl=5)
def fetch_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


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
) -> dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    incidents = [PreparedIncident.model_validate(row) for row in read_jsonl(input_path)]
    conditions = [ConditionName(c) for c in condition_values]
    manifest = load_manifest(models_manifest_path)
    client = OllamaClient(base_url=ollama_base_url)

    prepared_runs: list[tuple[PreparedIncident, dict[ConditionName, list[list[Article]]]]] = []
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
        len(manifest.models) * sum(len(condition_bundles) for condition_bundles in bundles.values())
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

    for incident, bundles in prepared_runs:
        if status_slot is not None:
            status_slot.caption(f"Running {incident.incident_id}")

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
                            retries=retries,
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
                    completed += 1
                    if progress_bar is not None and total_requests:
                        progress_bar.progress(completed / total_requests, text=f"{completed}/{total_requests} requests")

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
    c5.metric("Center Preference", f"{metrics.get('center_preference_index', 0.0):.3f}")
    c6.metric("Robustness", f"{metrics.get('content_robustness_score', 0.0):.3f}")


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
            color_discrete_sequence=["#3366CC", "#109618", "#DC3912", "#FF9900"],
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


def render_condition_metrics(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No condition-level metrics available yet.")
        return

    condition_df = pd.DataFrame(rows)
    st.subheader("Condition Metrics")
    st.dataframe(condition_df, use_container_width=True, hide_index=True)

    heatmap_df = (
        condition_df[["condition", "left_ratio", "center_ratio", "right_ratio"]]
        .melt(id_vars="condition", var_name="bucket", value_name="ratio")
    )
    heatmap_df["bucket"] = heatmap_df["bucket"].str.replace("_ratio", "", regex=False)
    heatmap = px.density_heatmap(
        heatmap_df,
        x="bucket",
        y="condition",
        z="ratio",
        color_continuous_scale="Blues",
        text_auto=".0%",
        title="Selected Leaning by Condition",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    latency_fig = px.bar(
        condition_df,
        x="condition",
        y="avg_latency_ms",
        title="Average Latency by Condition",
        color="condition",
    )
    st.plotly_chart(latency_fig, use_container_width=True)


def render_top_outlets(rows: list[dict[str, Any]]) -> None:
    if not rows:
        st.info("No outlet selections available yet.")
        return
    outlet_df = pd.DataFrame(rows)
    st.subheader("Top Selected Outlets")
    fig = px.bar(
        outlet_df.sort_values("count", ascending=True),
        x="count",
        y="selected_outlet",
        orientation="h",
        title="Top Selected Outlets",
        color="count",
        color_continuous_scale="Viridis",
    )
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
    st.subheader("Inter-Model Metrics")
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
    api_base = st.text_input("API Base URL", value=DEFAULT_API_BASE).rstrip("/")
    ollama_base = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE).rstrip("/")
    model_filter = st.text_input("Model filter (optional)", value="")
    selected_run = st.text_input("Run filter (optional)", value="")

    st.divider()
    st.subheader("Bulk Ingest")
    outputs_dir = st.text_input("Outputs root", value="outputs")
    if st.button("Ingest all runs in outputs"):
        try:
            ingest_resp = post_json(
                f"{api_base}/ingest/runs",
                {"outputs_dir": outputs_dir.strip() or "outputs"},
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
                ingest_resp = post_json(
                    f"{api_base}/ingest/run",
                    {"run_dir": run_dir.strip()},
                )
                st.success(
                    f"Ingested {ingest_resp.get('ingested', 0)} records | "
                    f"Added {ingest_resp.get('added', 0)}"
                )
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"Single-run ingest failed: {exc}")

analytics_tab, runner_tab = st.tabs(["Analytics", "Run Models"])

with analytics_tab:
    st.subheader("Summary")

    try:
        summary_params = {}
        if model_filter:
            summary_params["model"] = model_filter
        if selected_run:
            summary_params["run_id"] = selected_run
        summary_params = summary_params or None
        summary = fetch_json(f"{api_base}/metrics/summary", params=summary_params)
        metrics = summary.get("metrics", {})
        count = int(summary.get("count", 0))

        metrics_cards(metrics, count)
        render_distribution_charts(metrics)
    except Exception as exc:
        st.error(f"Could not load summary metrics: {exc}")

    st.subheader("Runs")
    try:
        runs_payload = fetch_json(f"{api_base}/metrics/runs")
        runs = runs_payload.get("runs", [])
        if runs:
            st.code("\n".join(runs))
        else:
            st.info("No run IDs found in analytics database yet.")
    except Exception as exc:
        st.error(f"Could not load run list: {exc}")

    try:
        inter_model_params = {"run_id": selected_run} if selected_run else None
        inter_model = fetch_json(f"{api_base}/metrics/inter-model", params=inter_model_params)
        if isinstance(inter_model, dict) and "error" not in inter_model:
            render_inter_model(inter_model)
        else:
            st.info("Inter-model metrics are not available yet.")
    except Exception as exc:
        st.error(f"Could not load inter-model metrics: {exc}")

    try:
        condition_params = {}
        if model_filter:
            condition_params["model"] = model_filter
        if selected_run:
            condition_params["run_id"] = selected_run
        condition_rows = fetch_json(f"{api_base}/metrics/conditions", params=condition_params or None).get("rows", [])
        render_condition_metrics(condition_rows)
    except Exception as exc:
        st.error(f"Could not load condition metrics: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        try:
            outlet_params = {}
            if model_filter:
                outlet_params["model"] = model_filter
            if selected_run:
                outlet_params["run_id"] = selected_run
            top_outlets = fetch_json(f"{api_base}/metrics/top-outlets", params=outlet_params or None).get("rows", [])
            render_top_outlets(top_outlets)
        except Exception as exc:
            st.error(f"Could not load outlet metrics: {exc}")
    with col2:
        try:
            run_params = {"model": model_filter} if model_filter else None
            run_rows = fetch_json(f"{api_base}/metrics/run-summaries", params=run_params).get("rows", [])
            render_run_summaries(run_rows)
        except Exception as exc:
            st.error(f"Could not load run summaries: {exc}")

    try:
        record_params = {"limit": 50}
        if model_filter:
            record_params["model"] = model_filter
        if selected_run:
            record_params["run_id"] = selected_run
        records = fetch_json(f"{api_base}/metrics/records", params=record_params).get("rows", [])
        render_sample_records(records)
    except Exception as exc:
        st.error(f"Could not load record samples: {exc}")

with runner_tab:
    probe_tab, batch_tab = st.tabs(["Probe Model", "Batch Experiment"])

    with probe_tab:
        st.write("Run a single prompt against an Ollama model using the project client.")
        probe_model = st.text_input("Model name", value="qwen2.5:7b")
        probe_prompt = st.text_area(
            "Prompt",
            value="Return strict JSON with selected_article_id and reason.",
            height=140,
        )
        probe_temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
        probe_max_tokens = st.number_input("Max tokens", min_value=1, max_value=4096, value=300, step=50)
        probe_timeout = st.number_input("Timeout seconds", min_value=1, max_value=300, value=60, step=5)

        if st.button("Run probe"):
            try:
                with st.spinner("Calling Ollama..."):
                    generation = OllamaClient(base_url=ollama_base).generate(
                        model=probe_model,
                        prompt=probe_prompt,
                        temperature=float(probe_temperature),
                        max_tokens=int(probe_max_tokens),
                        timeout_seconds=int(probe_timeout),
                    )
                st.success(f"Latency: {generation.latency_ms} ms")
                st.text_area("Model response", value=generation.text, height=220)
                st.expander("Raw payload").json(generation.raw_payload)
            except Exception as exc:
                st.error(f"Probe failed: {exc}")

    with batch_tab:
        st.write("Run the full experiment pipeline using existing builders, parser, and schemas.")
        input_path = st.text_input("Input incidents JSONL", value="data/real_incidents_random_train.jsonl")
        manifest_path = st.text_input("Models manifest YAML", value="configs/models.example.yaml")
        output_root = st.text_input("Output root", value="outputs")

        condition_values = st.multiselect(
            "Conditions",
            options=[c.value for c in ConditionName],
            default=[c.value for c in ConditionName],
        )
        max_combinations = st.number_input("Max combinations", min_value=1, max_value=50, value=3, step=1)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)
        retries = st.number_input("Retries", min_value=0, max_value=10, value=2, step=1)
        shuffle_candidates = st.checkbox("Shuffle candidate order", value=True)
        auto_ingest = st.checkbox("Auto-ingest run into analytics API", value=True)

        if st.button("Run batch experiment"):
            if not condition_values:
                st.warning("Select at least one condition.")
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
                        )
                    progress_bar.progress(1.0, text="Completed")
                    status_slot.caption(f"Run finished: {summary['run_id']}")
                    st.success("Batch run completed")
                    st.json(summary)

                    if auto_ingest:
                        try:
                            ingest_resp = post_json(
                                f"{api_base}/ingest/run",
                                {"run_dir": summary["output_dir"]},
                            )
                            st.success(
                                f"Auto-ingest complete: {ingest_resp.get('ingested', 0)} records "
                                f"(added {ingest_resp.get('added', 0)})"
                            )
                            st.cache_data.clear()
                        except Exception as exc:
                            st.warning(f"Run completed but auto-ingest failed: {exc}")
                except Exception as exc:
                    st.error(f"Batch run failed: {exc}")
