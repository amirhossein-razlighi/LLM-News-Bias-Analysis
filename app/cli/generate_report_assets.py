from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    qualitative_errors = build_qualitative_errors(df, sample_size=args.error_sample_size)

    plot_parse_success(summary, assets_dir / "parse_success_by_model.png")
    plot_latency(summary, assets_dir / "latency_by_model.png")
    plot_condition_bucket_mix(df, assets_dir / "condition_bucket_mix.png")
    plot_center_vs_baseline(df, summary, assets_dir / "center_vs_baseline.png")
    write_summary_table(summary, assets_dir / "model_summary.md")
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
        "assets_dir": str(assets_dir),
    }
    (assets_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()