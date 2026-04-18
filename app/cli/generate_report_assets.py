from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready plots and summary tables from outputs/run_* artifacts."
    )
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing run_* folders")
    parser.add_argument(
        "--assets-dir",
        default="docs/figures",
        help="Directory where generated figures and markdown summary will be written",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples used for confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed used for bootstrap confidence intervals",
    )
    parser.add_argument(
        "--error-sample-size",
        type=int,
        default=10,
        help="Maximum number of representative parse failures/fallbacks to include",
    )
    parser.add_argument(
        "--frozen-manifest",
        default="configs/models.final.yaml",
        help="Path to the frozen model manifest used for final reproducibility metadata",
    )
    parser.add_argument(
        "--frozen-seed",
        type=int,
        default=42,
        help="Frozen seed value used for final reproducibility metadata",
    )
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


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return (int(base_seed) + int(digest, 16)) % (2**32)


def bootstrap_mean_ci(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        val = float(arr[0])
        return val, val

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(samples, arr.size))
    means = arr[idx].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_decision_frame(outputs_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for run_dir in sorted(outputs_dir.glob("run_*")):
        requests_path = run_dir / "experiment_requests.jsonl"
        decisions_path = run_dir / "model_decisions.jsonl"
        if not requests_path.exists() or not decisions_path.exists():
            continue

        request_rows = read_jsonl(requests_path)
        request_index = {
            str(row.get("request_id")): row
            for row in request_rows
            if row.get("request_id") is not None
        }

        for decision in read_jsonl(decisions_path):
            request_id = str(decision.get("request_id") or "")
            req = request_index.get(request_id, {})

            candidates = req.get("candidates") or []
            candidate_order = req.get("candidate_order") or [
                c.get("article_id") for c in candidates if c.get("article_id")
            ]

            selected_article_id = str(decision.get("selected_article_id") or "")
            selected_bucket = "unknown"
            for candidate in candidates:
                if candidate.get("article_id") == selected_article_id:
                    selected_bucket = str(candidate.get("leaning") or "").lower().strip() or "unknown"
                    break
            if selected_bucket == "unknown":
                selected_bucket = infer_bucket(selected_article_id)

            selected_position = None
            if selected_article_id and selected_article_id in candidate_order:
                selected_position = int(candidate_order.index(selected_article_id) + 1)

            center_candidates = 0
            if candidates:
                center_candidates = sum(
                    1 for c in candidates if str(c.get("leaning") or "").strip().lower() == "center"
                )

            records.append(
                {
                    "run_id": str(decision.get("run_id") or run_dir.name),
                    "request_id": request_id,
                    "incident_id": str(decision.get("incident_id") or ""),
                    "model_name": str(decision.get("model_name") or ""),
                    "condition": str(decision.get("condition") or ""),
                    "parse_status": str(decision.get("parse_status") or ""),
                    "parsed_successfully": str(decision.get("parse_status") or "").lower() == "success",
                    "latency_ms": pd.to_numeric(decision.get("latency_ms"), errors="coerce"),
                    "selected_article_id": selected_article_id,
                    "selected_bucket": selected_bucket,
                    "selected_position": selected_position,
                    "candidate_signature": "|".join(sorted(str(c.get("article_id") or "") for c in candidates if c.get("article_id"))),
                    "candidate_count": len(candidates),
                    "center_candidate_ratio": (center_candidates / len(candidates)) if candidates else 0.0,
                    "reason": str(decision.get("reason") or ""),
                    "error": str(decision.get("error") or ""),
                    "raw_response": str(decision.get("raw_response") or ""),
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    return df


def build_model_summary(df: pd.DataFrame, bootstrap_samples: int, bootstrap_seed: int) -> pd.DataFrame:
    if "condition" in df.columns:
        instability = build_model_instability(df)
        sensitivity = build_counterfactual_effects(df, bootstrap_samples, bootstrap_seed)["by_model"]
    else:
        instability = pd.DataFrame(columns=["model", "instability_score"])
        sensitivity = pd.DataFrame(columns=["model", "label_sensitivity_rate"])
    instability_map = dict(zip(instability["model"], instability["instability_score"])) if not instability.empty else {}
    sensitivity_map = dict(zip(sensitivity["model"], sensitivity["label_sensitivity_rate"])) if not sensitivity.empty else {}

    by_model = []
    for model_name, group in df.groupby("model_name", dropna=False):
        latency = pd.to_numeric(group["latency_ms"], errors="coerce").dropna()
        known = group[group["selected_bucket"].isin(["left", "center", "right"])]
        center_rate = (known["selected_bucket"] == "center").mean() if not known.empty else 0.0
        success_series = (group["parse_status"] == "success").astype(float)
        center_series = (known["selected_bucket"] == "center").astype(float)

        model_seed = _stable_seed(bootstrap_seed, str(model_name))
        parse_ci_low, parse_ci_high = bootstrap_mean_ci(success_series, bootstrap_samples, model_seed)
        center_ci_low, center_ci_high = bootstrap_mean_ci(center_series, bootstrap_samples, model_seed + 1)

        by_model.append(
            {
                "model": str(model_name),
                "n": int(len(group)),
                "parse_success_rate": float((group["parse_status"] == "success").mean()),
                "parse_success_ci95_low": parse_ci_low,
                "parse_success_ci95_high": parse_ci_high,
                "parse_fallback_rate": float((group["parse_status"] == "fallback").mean()),
                "parse_failure_rate": float((group["parse_status"] == "failed").mean()),
                "avg_latency_ms": float(latency.mean()) if not latency.empty else 0.0,
                "p95_latency_ms": float(latency.quantile(0.95)) if not latency.empty else 0.0,
                "center_selection_rate": float(center_rate),
                "center_selection_ci95_low": center_ci_low,
                "center_selection_ci95_high": center_ci_high,
                "instability_score": float(instability_map.get(str(model_name), 0.0)),
                "label_sensitivity_rate": float(sensitivity_map.get(str(model_name), 0.0)),
            }
        )

    return pd.DataFrame(by_model).sort_values("parse_success_rate", ascending=False)


def _clip_text(value: str, limit: int = 160) -> str:
    text = (value or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_qualitative_errors(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    errors = df[df["parse_status"].isin(["failed", "fallback"])].copy()
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "model_name",
                "condition",
                "incident_id",
                "parse_status",
                "selected_article_id",
                "error_excerpt",
                "response_excerpt",
            ]
        )

    errors["error_excerpt"] = errors["error"].map(_clip_text)
    errors["response_excerpt"] = errors["raw_response"].map(_clip_text)

    rank = {"failed": 0, "fallback": 1}
    errors["_rank"] = errors["parse_status"].map(rank).fillna(99)
    errors["_lat"] = pd.to_numeric(errors["latency_ms"], errors="coerce").fillna(-1)

    deduped = errors.drop_duplicates(subset=["model_name", "condition", "incident_id", "parse_status"])
    ordered = deduped.sort_values(["_rank", "_lat"], ascending=[True, False]).head(sample_size)

    return ordered[
        [
            "run_id",
            "model_name",
            "condition",
            "incident_id",
            "parse_status",
            "selected_article_id",
            "error_excerpt",
            "response_excerpt",
        ]
    ].rename(columns={"model_name": "model"})


def classify_parse_error(parse_status: str, error: str) -> str:
    status = (parse_status or "").strip().lower()
    message = (error or "").strip().lower()
    if status == "fallback":
        if "json decode" in message:
            return "fallback_after_malformed_json"
        if "no json object" in message:
            return "fallback_from_plain_text"
        return "fallback_other"
    if "selected_article_id missing or not in candidates" in message:
        return "invalid_or_missing_selected_article_id"
    if "json decode" in message:
        return "malformed_json"
    if "no json object found" in message:
        return "no_json_object"
    if "timeout" in message:
        return "timeout"
    if "connection" in message:
        return "connection_error"
    if status == "success" and "recovered from malformed json" in message:
        return "success_recovered_from_malformed_json"
    return "other"


def build_failure_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model", "parse_status", "error_category", "count", "ratio_within_model"])

    work = df.copy()
    work["error_category"] = work.apply(
        lambda row: classify_parse_error(str(row.get("parse_status") or ""), str(row.get("error") or "")),
        axis=1,
    )
    grouped = (
        work.groupby(["model_name", "parse_status", "error_category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped["ratio_within_model"] = grouped.groupby("model_name")["count"].transform(lambda s: s / s.sum())
    return grouped.rename(columns={"model_name": "model"}).sort_values(["model", "count"], ascending=[True, False])


def build_counterfactual_effects(
    df: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, pd.DataFrame]:
    required_conditions = {"headlines_with_sources", "headlines_with_manipulated_sources"}
    pairable = df[df["condition"].isin(required_conditions)].copy()
    pairable = pairable[pairable["selected_bucket"].isin(["left", "center", "right"])].copy()
    if pairable.empty:
        empty = pd.DataFrame(columns=["model", "n_pairs", "label_sensitivity_rate", "ci95_low", "ci95_high"])
        return {"overall": empty, "by_model": empty}

    key_cols = ["run_id", "model_name", "incident_id", "candidate_signature"]
    pivot = (
        pairable.pivot_table(index=key_cols, columns="condition", values="selected_bucket", aggfunc="first")
        .dropna(subset=["headlines_with_sources", "headlines_with_manipulated_sources"])
        .reset_index()
    )
    if pivot.empty:
        empty = pd.DataFrame(columns=["model", "n_pairs", "label_sensitivity_rate", "ci95_low", "ci95_high"])
        return {"overall": empty, "by_model": empty}

    pivot["changed"] = (
        pivot["headlines_with_sources"] != pivot["headlines_with_manipulated_sources"]
    ).astype(float)

    by_model = []
    for model_name, group in pivot.groupby("model_name"):
        model_seed = _stable_seed(bootstrap_seed, f"counterfactual:{model_name}")
        ci_low, ci_high = bootstrap_mean_ci(group["changed"], bootstrap_samples, model_seed)
        by_model.append(
            {
                "model": str(model_name),
                "n_pairs": int(len(group)),
                "label_sensitivity_rate": float(group["changed"].mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
            }
        )

    overall_seed = _stable_seed(bootstrap_seed, "counterfactual:overall")
    overall_low, overall_high = bootstrap_mean_ci(pivot["changed"], bootstrap_samples, overall_seed)
    overall = pd.DataFrame(
        [
            {
                "model": "ALL",
                "n_pairs": int(len(pivot)),
                "label_sensitivity_rate": float(pivot["changed"].mean()),
                "ci95_low": float(overall_low),
                "ci95_high": float(overall_high),
            }
        ]
    )

    return {
        "overall": overall,
        "by_model": pd.DataFrame(by_model).sort_values("label_sensitivity_rate", ascending=False),
    }


def build_cross_model_agreement(df: pd.DataFrame) -> pd.DataFrame:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return pd.DataFrame(columns=["condition", "n_groups", "mean_agreement_rate", "mean_normalized_entropy", "instability_score"])

    rows = []
    for (run_id, incident_id, condition), group in known.groupby(["run_id", "incident_id", "condition"], dropna=False):
        n = len(group)
        if n < 2:
            continue
        probs = group["selected_bucket"].value_counts(normalize=True)
        agreement = float(probs.max())
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        norm_entropy = entropy / np.log2(3)
        rows.append(
            {
                "run_id": str(run_id),
                "incident_id": str(incident_id),
                "condition": str(condition),
                "n_models": int(n),
                "agreement_rate": agreement,
                "normalized_entropy": float(norm_entropy),
                "instability": float(norm_entropy),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(columns=["condition", "n_groups", "mean_agreement_rate", "mean_normalized_entropy", "instability_score"])

    summary = (
        detail.groupby("condition", dropna=False)
        .agg(
            n_groups=("agreement_rate", "size"),
            mean_agreement_rate=("agreement_rate", "mean"),
            mean_normalized_entropy=("normalized_entropy", "mean"),
            instability_score=("instability", "mean"),
        )
        .reset_index()
        .sort_values("condition")
    )
    return summary


def build_model_instability(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "run_id", "incident_id", "condition", "selected_bucket"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["model", "n_incidents", "instability_score"])

    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return pd.DataFrame(columns=["model", "n_incidents", "instability_score"])

    rows = []
    for model_name, group in known.groupby("model_name", dropna=False):
        incident_scores = []
        for _, g in group.groupby(["run_id", "incident_id"], dropna=False):
            conditions_seen = int(g["condition"].nunique())
            unique_buckets = int(g["selected_bucket"].nunique())
            if conditions_seen < 2:
                continue
            score = (unique_buckets - 1) / (conditions_seen - 1)
            incident_scores.append(float(max(0.0, min(1.0, score))))
        rows.append(
            {
                "model": str(model_name),
                "n_incidents": int(len(incident_scores)),
                "instability_score": float(np.mean(incident_scores)) if incident_scores else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("instability_score", ascending=False)


def write_markdown_table(table: pd.DataFrame, output_path: Path, columns: list[str]) -> None:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_limitations(output_path: Path) -> None:
    lines = [
        "## Limitations",
        "",
        "- External validity is bounded by the selected outlets and incident sampling procedure.",
        "- Prompt-conditioned source-selection behavior may not transfer to free-form generation settings.",
        "- Parsing reliability varies by model and can suppress measured selection behavior when failures are high.",
        "- Latency reflects local Ollama runtime and hardware; cross-machine comparisons should be normalized.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_condition_to_bucket_sankey_html(df: pd.DataFrame, output_path: Path) -> None:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return

    grouped = known.groupby(["condition", "selected_bucket"], dropna=False).size().reset_index(name="count")
    condition_labels = sorted(grouped["condition"].astype(str).unique().tolist())
    bucket_labels = ["left", "center", "right"]
    labels = condition_labels + bucket_labels

    source = []
    target = []
    values = []
    for _, row in grouped.iterrows():
        source.append(condition_labels.index(str(row["condition"])))
        target.append(len(condition_labels) + bucket_labels.index(str(row["selected_bucket"])))
        values.append(int(row["count"]))

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.2),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=values),
            )
        ]
    )
    fig.update_layout(title_text="Condition to Selected Leaning Flow", font_size=12)
    fig.write_html(output_path, include_plotlyjs="cdn")


def plot_center_delta_heatmap(df: pd.DataFrame, output_path: Path, bootstrap_samples: int, bootstrap_seed: int) -> None:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return

    rows = []
    for (model_name, condition), group in known.groupby(["model_name", "condition"], dropna=False):
        baseline = float(df[df["condition"] == condition]["center_candidate_ratio"].mean())
        center_series = (group["selected_bucket"] == "center").astype(float)
        seed = _stable_seed(bootstrap_seed, f"center_delta:{model_name}:{condition}")
        ci_low, ci_high = bootstrap_mean_ci(center_series, bootstrap_samples, seed)
        center_rate = float(center_series.mean())
        delta = center_rate - baseline
        rows.append(
            {
                "model": str(model_name),
                "condition": str(condition),
                "delta": float(delta),
                "marker": "ns" if (ci_low <= baseline <= ci_high) else "*",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return

    models = sorted(table["model"].unique().tolist())
    conditions = sorted(table["condition"].unique().tolist())
    pivot = table.pivot(index="model", columns="condition", values="delta").reindex(index=models, columns=conditions)
    matrix = pivot.fillna(0.0).values
    vmax = max(0.01, float(np.abs(matrix).max()))

    plt.figure(figsize=(1.8 * max(3, len(conditions)), 0.6 * max(4, len(models)) + 2.2))
    im = plt.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, fraction=0.035, pad=0.02, label="Center rate delta vs baseline")
    plt.xticks(range(len(conditions)), conditions, rotation=25, ha="right")
    plt.yticks(range(len(models)), models)
    plt.title("Center Selection Delta by Model and Condition")

    marker_lookup = {(r["model"], r["condition"]): r["marker"] for _, r in table.iterrows()}
    for i, model_name in enumerate(models):
        for j, cond_name in enumerate(conditions):
            value = matrix[i, j]
            marker = marker_lookup.get((model_name, cond_name), "")
            plt.text(j, i, f"{value:+.2f}\n{marker}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_reliability_speed_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    work = summary.copy()
    size = 120 + 500 * work.get("instability_score", pd.Series([0.0] * len(work)))

    plt.figure(figsize=(10, 6))
    plt.scatter(
        work["avg_latency_ms"],
        work["parse_success_rate"],
        s=size,
        alpha=0.75,
        c=work.get("label_sensitivity_rate", pd.Series([0.0] * len(work))),
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
    )
    for _, row in work.iterrows():
        plt.annotate(str(row["model"]), (row["avg_latency_ms"], row["parse_success_rate"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.xlabel("Average latency (ms)")
    plt.ylabel("Parse success rate")
    plt.ylim(0, 1.02)
    plt.title("Reliability-Speed Pareto Frontier (bubble size = instability)")
    cbar = plt.colorbar()
    cbar.set_label("Label sensitivity rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_quality_score(row: pd.Series) -> float:
    text = str(row.get("raw_response") or "").strip()
    status = str(row.get("parse_status") or "").lower()
    score = 0.0
    if text.startswith("{"):
        score += 0.35
    if "selected_article_id" in text:
        score += 0.35
    if "reason" in text:
        score += 0.20
    if status == "success":
        score += 0.10
    return float(max(0.0, min(1.0, score)))


def plot_parse_calibration(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        return
    work = df.copy()
    work["quality_score"] = work.apply(parse_quality_score, axis=1)
    work["success"] = (work["parse_status"] == "success").astype(float)

    try:
        work["bin"] = pd.qcut(work["quality_score"], q=5, duplicates="drop")
    except ValueError:
        return

    curve = (
        work.groupby("bin", dropna=False, observed=False)
        .agg(
            mean_quality=("quality_score", "mean"),
            observed_success=("success", "mean"),
            n=("success", "size"),
        )
        .reset_index(drop=True)
        .sort_values("mean_quality")
    )
    if curve.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(curve["mean_quality"], curve["observed_success"], marker="o", label="Observed")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    for _, row in curve.iterrows():
        plt.annotate(f"n={int(row['n'])}", (row["mean_quality"], row["observed_success"]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.xlabel("Response format quality score")
    plt.ylabel("Observed parse success")
    plt.ylim(0, 1.02)
    plt.xlim(0, 1.0)
    plt.title("Parse Reliability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_parse_success(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("parse_success_rate", ascending=False)
    plt.figure(figsize=(11, 5))
    plt.bar(ordered["model"], ordered["parse_success_rate"], color="#1f77b4")
    plt.ylim(0, 1.05)
    plt.ylabel("Parse success rate")
    plt.title("Parse Reliability by Model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_latency(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.sort_values("avg_latency_ms", ascending=True)
    x = range(len(ordered))
    width = 0.4

    plt.figure(figsize=(11, 5))
    plt.bar([i - width / 2 for i in x], ordered["avg_latency_ms"], width=width, label="Avg latency")
    plt.bar([i + width / 2 for i in x], ordered["p95_latency_ms"], width=width, label="P95 latency")
    plt.ylabel("Milliseconds")
    plt.title("Latency by Model")
    plt.xticks(list(x), ordered["model"], rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_condition_bucket_mix(df: pd.DataFrame, output_path: Path) -> None:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])]
    if known.empty:
        return

    mix = (
        known.groupby(["condition", "selected_bucket"]).size().reset_index(name="count")
    )
    mix["ratio"] = mix.groupby("condition")["count"].transform(lambda s: s / s.sum())

    pivot = (
        mix.pivot(index="condition", columns="selected_bucket", values="ratio")
        .fillna(0.0)
        .reindex(columns=["left", "center", "right"], fill_value=0.0)
    )

    plt.figure(figsize=(11, 5.5))
    bottom = None
    colors = {"left": "#d62728", "center": "#2ca02c", "right": "#1f77b4"}
    for bucket in ["left", "center", "right"]:
        values = pivot[bucket].values
        if bottom is None:
            plt.bar(pivot.index, values, label=bucket, color=colors[bucket])
            bottom = values
        else:
            plt.bar(pivot.index, values, bottom=bottom, label=bucket, color=colors[bucket])
            bottom = bottom + values

    plt.ylabel("Selection ratio")
    plt.ylim(0, 1.02)
    plt.title("Selection Mix by Experimental Condition")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Selected bucket")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_center_vs_baseline(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    baseline = float(df["center_candidate_ratio"].mean()) if not df.empty else 0.0
    ordered = summary.sort_values("center_selection_rate", ascending=False)

    plt.figure(figsize=(11, 5))
    plt.bar(ordered["model"], ordered["center_selection_rate"], color="#2ca02c")
    plt.axhline(y=baseline, color="#ff7f0e", linestyle="--", linewidth=2, label=f"Random baseline ({baseline:.3f})")
    plt.ylabel("Center selection rate")
    plt.ylim(0, 1.02)
    plt.title("Center Selection vs Candidate-Mix Baseline")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_summary_table(summary: pd.DataFrame, output_path: Path) -> None:
    table = summary.copy()
    if "parse_success_ci95_low" not in table.columns:
        table["parse_success_ci95_low"] = table["parse_success_rate"]
    if "parse_success_ci95_high" not in table.columns:
        table["parse_success_ci95_high"] = table["parse_success_rate"]
    if "center_selection_ci95_low" not in table.columns:
        table["center_selection_ci95_low"] = table["center_selection_rate"]
    if "center_selection_ci95_high" not in table.columns:
        table["center_selection_ci95_high"] = table["center_selection_rate"]

    table["parse_success_rate"] = table["parse_success_rate"].map(lambda v: f"{v:.2%}")
    table["parse_success_ci95"] = table.apply(
        lambda r: f"[{r['parse_success_ci95_low']:.2%}, {r['parse_success_ci95_high']:.2%}]",
        axis=1,
    )
    table["parse_fallback_rate"] = table["parse_fallback_rate"].map(lambda v: f"{v:.2%}")
    table["parse_failure_rate"] = table["parse_failure_rate"].map(lambda v: f"{v:.2%}")
    table["center_selection_rate"] = table["center_selection_rate"].map(lambda v: f"{v:.2%}")
    table["center_selection_ci95"] = table.apply(
        lambda r: f"[{r['center_selection_ci95_low']:.2%}, {r['center_selection_ci95_high']:.2%}]",
        axis=1,
    )
    table["avg_latency_ms"] = table["avg_latency_ms"].map(lambda v: f"{v:.0f}")
    table["p95_latency_ms"] = table["p95_latency_ms"].map(lambda v: f"{v:.0f}")
    if "instability_score" not in table.columns:
        table["instability_score"] = 0.0
    if "label_sensitivity_rate" not in table.columns:
        table["label_sensitivity_rate"] = 0.0
    table["instability_score"] = table["instability_score"].map(lambda v: f"{v:.3f}")
    table["label_sensitivity_rate"] = table["label_sensitivity_rate"].map(lambda v: f"{v:.2%}")
    columns = [
        "model",
        "n",
        "parse_success_rate",
        "parse_success_ci95",
        "parse_fallback_rate",
        "parse_failure_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "center_selection_rate",
        "center_selection_ci95",
        "instability_score",
        "label_sensitivity_rate",
    ]
    write_markdown_table(table, output_path, columns)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    df = load_decision_frame(outputs_dir)
    if df.empty:
        raise SystemExit(f"No run artifacts found under {outputs_dir}")

    summary = build_model_summary(
        df,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    counterfactual = build_counterfactual_effects(df, args.bootstrap_samples, args.bootstrap_seed)
    agreement = build_cross_model_agreement(df)
    failure_taxonomy = build_failure_taxonomy(df)
    model_instability = build_model_instability(df)
    qualitative_errors = build_qualitative_errors(df, sample_size=args.error_sample_size)

    plot_parse_success(summary, assets_dir / "parse_success_by_model.png")
    plot_latency(summary, assets_dir / "latency_by_model.png")
    plot_condition_bucket_mix(df, assets_dir / "condition_bucket_mix.png")
    plot_center_vs_baseline(df, summary, assets_dir / "center_vs_baseline.png")
    plot_condition_to_bucket_sankey_html(df, assets_dir / "condition_to_bucket_sankey.html")
    plot_center_delta_heatmap(df, assets_dir / "center_delta_heatmap.png", args.bootstrap_samples, args.bootstrap_seed)
    plot_reliability_speed_pareto(summary, assets_dir / "reliability_speed_pareto.png")
    plot_parse_calibration(df, assets_dir / "parse_reliability_calibration.png")
    write_summary_table(summary, assets_dir / "model_summary.md")
    write_markdown_table(
        failure_taxonomy,
        assets_dir / "failure_taxonomy.md",
        ["model", "parse_status", "error_category", "count", "ratio_within_model"],
    )
    write_markdown_table(
        agreement,
        assets_dir / "cross_model_agreement.md",
        ["condition", "n_groups", "mean_agreement_rate", "mean_normalized_entropy", "instability_score"],
    )
    write_markdown_table(
        counterfactual["by_model"],
        assets_dir / "counterfactual_effects.md",
        ["model", "n_pairs", "label_sensitivity_rate", "ci95_low", "ci95_high"],
    )
    write_markdown_table(
        model_instability,
        assets_dir / "model_instability.md",
        ["model", "n_incidents", "instability_score"],
    )
    write_markdown_table(
        qualitative_errors,
        assets_dir / "qualitative_errors.md",
        [
            "run_id",
            "model",
            "condition",
            "incident_id",
            "parse_status",
            "selected_article_id",
            "error_excerpt",
            "response_excerpt",
        ],
    )
    (assets_dir / "qualitative_errors.json").write_text(
        json.dumps(qualitative_errors.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    (assets_dir / "failure_taxonomy.json").write_text(
        json.dumps(failure_taxonomy.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    (assets_dir / "counterfactual_effects.json").write_text(
        json.dumps(
            {
                "overall": counterfactual["overall"].to_dict(orient="records"),
                "by_model": counterfactual["by_model"].to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (assets_dir / "cross_model_agreement.json").write_text(
        json.dumps(agreement.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    write_limitations(assets_dir / "limitations.md")

    parse_ci_low, parse_ci_high = bootstrap_mean_ci(
        (df["parse_status"] == "success").astype(float),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    known = df[df["selected_bucket"].isin(["left", "center", "right"])]
    center_ci_low, center_ci_high = bootstrap_mean_ci(
        (known["selected_bucket"] == "center").astype(float),
        args.bootstrap_samples,
        args.bootstrap_seed + 1,
    )

    frozen_manifest_path = Path(args.frozen_manifest)

    manifest = {
        "total_decisions": int(len(df)),
        "run_count": int(df["run_id"].nunique()),
        "model_count": int(df["model_name"].nunique()),
        "condition_count": int(df["condition"].nunique()),
        "parse_success_rate": float((df["parse_status"] == "success").mean()),
        "parse_success_ci95_low": parse_ci_low,
        "parse_success_ci95_high": parse_ci_high,
        "parse_failure_rate": float((df["parse_status"] == "failed").mean()),
        "baseline_center_rate": float(df["center_candidate_ratio"].mean()),
        "center_selection_rate": float((known["selected_bucket"] == "center").mean()) if not known.empty else 0.0,
        "center_selection_ci95_low": center_ci_low,
        "center_selection_ci95_high": center_ci_high,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "frozen_seed": int(args.frozen_seed),
        "frozen_manifest": str(frozen_manifest_path),
        "frozen_manifest_sha256": file_sha256(frozen_manifest_path),
        "error_sample_size": int(args.error_sample_size),
        "qualitative_error_count": int(len(qualitative_errors)),
        "counterfactual_label_sensitivity_rate": float(counterfactual["overall"].iloc[0]["label_sensitivity_rate"]) if not counterfactual["overall"].empty else 0.0,
        "mean_cross_model_agreement": float(agreement["mean_agreement_rate"].mean()) if not agreement.empty else 0.0,
        "mean_cross_model_instability": float(agreement["instability_score"].mean()) if not agreement.empty else 0.0,
        "assets_dir": str(assets_dir),
    }
    (assets_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()