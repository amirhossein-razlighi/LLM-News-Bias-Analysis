from __future__ import annotations

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
        runs = sorted([r for r in df["run_id"].dropna().astype(str).unique().tolist() if r])

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
    response_schema = selection_response_json_schema()

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

    for model in manifest.models:
        if status_slot is not None:
            status_slot.caption(f"Running model {model.name}")
        for incident, bundles in prepared_runs:
            if status_slot is not None:
                status_slot.caption(f"Running model {model.name} on {incident.incident_id}")
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


def takeaway_panel(metrics: dict[str, Any]) -> None:
    bias_sensitivity_index = float(metrics.get("label_sensitivity_rate", metrics.get("identity_dominance_rate", 0.0)))
    reliability_index = float(metrics.get("parse_success_rate", 0.0))
    avg_latency = float(metrics.get("avg_latency_ms", 0.0))
    speed_index = 1.0 / (1.0 + max(0.0, avg_latency) / 1000.0)

    st.subheader("Takeaway Panel")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bias Sensitivity Index", f"{bias_sensitivity_index:.3f}")
    c2.metric("Reliability Index", f"{reliability_index:.3f}")
    c3.metric("Speed Index", f"{speed_index:.3f}")


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
    ranking = inter_df[["model", sort_col, "parse_success_rate", "avg_latency_ms", "content_robustness_score"]].sort_values(sort_col, ascending=ascending)
    st.markdown("**Objective Ranking**")
    st.dataframe(ranking, use_container_width=True, hide_index=True)


def _overlap_score(incident_id: str, reason: str) -> float:
    incident_tokens = [t for t in re.split(r"[_\W]+", str(incident_id).lower()) if len(t) >= 4]
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
    incidents = sorted(df["incident_id"].dropna().astype(str).unique().tolist())
    incident_id = st.selectbox("Incident", options=incidents, key="scenario_incident")
    incident_df = df[df["incident_id"] == incident_id].copy()
    conditions = sorted(incident_df["condition"].dropna().astype(str).unique().tolist())
    condition = st.selectbox("Condition", options=conditions, key="scenario_condition")
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
            color_discrete_map={"left": "#d62728", "center": "#2ca02c", "right": "#1f77b4"},
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
    explain["reason_excerpt"] = explain["justification"].fillna("").astype(str).str.slice(0, 180)
    explain["keyword_overlap"] = explain.apply(
        lambda row: _overlap_score(incident_id, row.get("justification", "")),
        axis=1,
    )
    explain = explain.drop(columns=["justification"])
    st.markdown("**Explainability Snippets**")
    st.dataframe(explain, use_container_width=True, hide_index=True)


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
    heatmap_df["bucket"] = heatmap_df["bucket"].str.replace("_ratio", "", regex=False)
    heatmap_df["condition_label"] = heatmap_df["condition"].map(condition_labels)
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

    heatmap_tab, distribution_tab, quality_tab = st.tabs(["Heatmap", "Distribution", "Quality"])
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
        fig.update_yaxes(title="Condition", automargin=True, tickfont=dict(size=14))
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
            ["condition", "parse_success_rate", "avg_latency_ms", "p95_latency_ms", "unknown_ratio"]
        ].copy()
        quality_df["parse_success_rate"] = quality_df["parse_success_rate"].fillna(0.0)
        quality_df["unknown_ratio"] = quality_df["unknown_ratio"].fillna(0.0)
        quality_df = quality_df.set_index("condition").reindex(condition_order).reset_index()
        quality_df["condition_label"] = quality_df["condition"].map(condition_labels)

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
    ollama_base = st.text_input("Ollama Base URL", value=DEFAULT_OLLAMA_BASE).rstrip("/")
    selected_run = st.text_input("Run filter (optional)", value="")
    st.caption("Analytics mode: embedded in Streamlit using the same backend logic as the FastAPI app.")

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
    st.caption("The FastAPI app stays in the repo for local demos, client integrations, and presentation samples.")

analytics_tab, runner_tab = st.tabs(["Analytics", "Run Models"])

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
        snapshot = get_dashboard_snapshot(selected_run=selected_run, record_limit=50)
    except Exception as exc:
        st.error(f"Could not load analytics snapshot: {exc}")
        snapshot = None

    if snapshot is not None:
        metrics_cards(snapshot["metrics"], snapshot["count"])
        takeaway_panel(snapshot["metrics"])

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
                run_rows = [row for row in run_rows if row.get("run_id") == selected_run]
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
        input_path = st.text_input("Input incidents JSONL", value="data/real_incidents_all.jsonl")
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
        auto_ingest = st.checkbox("Auto-ingest run into embedded analytics DB", value=True)

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
                            st.warning(f"Run completed but auto-ingest failed: {exc}")
                except Exception as exc:
                    st.error(f"Batch run failed: {exc}")
