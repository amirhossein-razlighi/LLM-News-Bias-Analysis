import json
import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Sourcerers Analytics Engine")

raw_origins = os.getenv("API_ALLOW_ORIGINS", "")
allow_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
if allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- DATABASE LAYER ---
BASE_DIR = Path(__file__).resolve().parents[2]
DB_FILE = BASE_DIR / "experiment_database.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
WRITE_ENDPOINTS_ENABLED = os.getenv(
    "ENABLE_ANALYTICS_WRITE_ENDPOINTS",
    "0" if os.getenv("VERCEL") else "1",
).strip().lower() not in {"0", "false", "no"}


def load_db(db_file: Path = DB_FILE):
    if db_file.exists() and db_file.stat().st_size > 0:
        return pd.read_csv(db_file)
    return pd.DataFrame()


def save_to_db(df: pd.DataFrame, db_file: Path = DB_FILE) -> None:
    db_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(db_file, index=False)


def _require_write_access() -> None:
    if not WRITE_ENDPOINTS_ENABLED:
        raise HTTPException(
            status_code=403,
            detail="Write endpoints are disabled in this deployment.",
        )


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value) or np.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


def _json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_safe_records(rows: List[dict]) -> List[dict]:
    return [
        {key: _json_safe_value(value) for key, value in row.items()}
        for row in rows
    ]


def _ensure_analytics_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()
    if "candidate_signature" not in data.columns:
        data["candidate_signature"] = data["request_id"].astype(str)
    else:
        data["candidate_signature"] = (
            data["candidate_signature"]
            .fillna("")
            .astype(str)
            .replace("", pd.NA)
            .fillna(data["request_id"].astype(str))
        )

    if "selected_bucket" not in data.columns:
        data["selected_bucket"] = "unknown"
    else:
        data["selected_bucket"] = data["selected_bucket"].fillna("unknown").astype(str)

    if "parsed_successfully" not in data.columns:
        data["parsed_successfully"] = False
    else:
        data["parsed_successfully"] = data["parsed_successfully"].fillna(False)

    if "latency_ms" not in data.columns:
        data["latency_ms"] = 0

    if "selected_position" not in data.columns:
        data["selected_position"] = pd.NA

    return data


# --- SCHEMA ---
class ExperimentResult(BaseModel):
    request_id: str
    incident_id: str
    condition: str
    model_name: str
    selected_article_id: str
    selected_outlet: str
    selected_bucket: str  # 'left', 'center', or 'right'
    justification: str
    raw_response: str
    parsed_successfully: bool
    latency_ms: int
    timestamp_utc: str


class IngestRunRequest(BaseModel):
    run_dir: str
    model_decisions_file: str = "model_decisions.jsonl"
    experiment_requests_file: str = "experiment_requests.jsonl"


class IngestRunsRequest(BaseModel):
    outputs_dir: str = "outputs"
    run_dir_prefix: str = "run_"
    model_decisions_file: str = "model_decisions.jsonl"
    experiment_requests_file: str = "experiment_requests.jsonl"


# --- ANALYTICS CALCULATIONS ---
def calculate_all_metrics(df: pd.DataFrame):
    if df.empty:
        return {}

    data = _ensure_analytics_columns(df)
    if "condition" in data.columns:
        data["condition"] = data["condition"].replace(
            {
                "headlines_with_sources": "headlines_sources",
                "headlines_with_manipulated_sources": "swapped_sources",
            }
        )

    known_buckets = data[data["selected_bucket"].isin(["left", "center", "right"])].copy()
    counts = known_buckets['selected_bucket'].value_counts(normalize=True).to_dict()
    center_pref = counts.get('center', 0.0)
    skew = counts.get('left', 0.0) - counts.get('right', 0.0)

    stability_score = 0.0
    identity_dominance = 0.0

    pivot = data.pivot_table(
        index=['incident_id', 'model_name', 'candidate_signature'],
        columns='condition',
        values='selected_article_id',
        aggfunc='first'
    )

    selected_position_distribution = {}
    if "selected_position" in data.columns:
        valid_positions = data["selected_position"].dropna()
        if not valid_positions.empty:
            selected_position_distribution = (
                valid_positions.astype(int).value_counts(normalize=True).sort_index().to_dict()
            )

    # Metric: Selection Stability (Headlines Only vs Headlines+Sources)
    if 'headlines_only' in pivot.columns and 'headlines_sources' in pivot.columns:
        valid = pivot[['headlines_only', 'headlines_sources']].dropna()
        if not valid.empty:
            stability_score = (valid['headlines_only'] == valid['headlines_sources']).mean()

    # Metric: Identity Dominance (Sources vs Swapped Labels)
    if 'headlines_sources' in pivot.columns and 'swapped_sources' in pivot.columns:
        valid_swap = pivot[['headlines_sources', 'swapped_sources']].dropna()
        if not valid_swap.empty:
            identity_dominance = (valid_swap['headlines_sources'] != valid_swap['swapped_sources']).mean()

    parse_rate = _safe_float(data["parsed_successfully"].mean()) if "parsed_successfully" in data.columns else 0.0
    fallback_rate = _safe_float((data.get("parse_status") == "fallback").mean()) if "parse_status" in data.columns else 0.0
    failure_rate = _safe_float((data.get("parse_status") == "failed").mean()) if "parse_status" in data.columns else 0.0
    latency = pd.to_numeric(data.get("latency_ms"), errors="coerce").dropna()
    latency_avg = _safe_float(latency.mean()) if not latency.empty else 0.0
    latency_p50 = _safe_float(latency.quantile(0.5)) if not latency.empty else 0.0
    latency_p95 = _safe_float(latency.quantile(0.95)) if not latency.empty else 0.0
    unknown_bucket_rate = _safe_float((data["selected_bucket"] == "unknown").mean()) if "selected_bucket" in data.columns else 0.0
    label_sensitivity_rate = _label_sensitivity_rate(data)
    cross_model_agreement_rate, cross_model_instability = _cross_model_agreement(data)
    model_instability_score = _model_instability_score(data)

    return {
        "selection_distribution": counts,
        "center_preference_index": float(center_pref),
        "partisan_skew_score": float(skew),
        "selection_stability_score": float(stability_score),
        "identity_dominance_rate": float(identity_dominance),
        "content_robustness_score": float(1 - identity_dominance),
        "selected_position_distribution": selected_position_distribution,
        "parse_success_rate": parse_rate,
        "parse_fallback_rate": fallback_rate,
        "parse_failure_rate": failure_rate,
        "avg_latency_ms": latency_avg,
        "p50_latency_ms": latency_p50,
        "p95_latency_ms": latency_p95,
        "unknown_bucket_rate": unknown_bucket_rate,
        "label_sensitivity_rate": label_sensitivity_rate,
        "cross_model_agreement_rate": cross_model_agreement_rate,
        "cross_model_instability": cross_model_instability,
        "model_instability_score": model_instability_score,
    }


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bucket_from_article_id(article_id: str) -> str:
    normalized = (article_id or "").lower()
    if re.search(r"(^|_)left(_|$)", normalized):
        return "left"
    if re.search(r"(^|_)center(_|$)", normalized):
        return "center"
    if re.search(r"(^|_)right(_|$)", normalized):
        return "right"
    return "unknown"


def _build_request_index(request_rows: List[dict]) -> dict[str, dict]:
    return {
        str(row.get("request_id")): row
        for row in request_rows
        if row.get("request_id") is not None
    }


def _normalize_generated_rows(model_decisions: List[dict], request_index: dict[str, dict]) -> List[dict]:
    normalized: List[dict] = []
    for decision in model_decisions:
        request_id = str(decision.get("request_id", ""))
        selected_article_id = str(decision.get("selected_article_id") or "")
        request_row = request_index.get(request_id, {})

        candidates = request_row.get("candidates") or []
        selected_outlet = ""
        selected_bucket = "unknown"
        for candidate in candidates:
            if candidate.get("article_id") == selected_article_id:
                selected_outlet = str(candidate.get("outlet_name") or "")
                selected_bucket = str(candidate.get("leaning") or "").strip().lower() or "unknown"
                break

        candidate_order = request_row.get("candidate_order") or []
        selected_position: int | None = None
        if selected_article_id and selected_article_id in candidate_order:
            selected_position = int(candidate_order.index(selected_article_id) + 1)
        if selected_bucket == "unknown":
            selected_bucket = _bucket_from_article_id(selected_article_id)

        candidate_signature = "|".join(sorted(str(candidate.get("article_id") or "") for candidate in candidates if candidate.get("article_id")))

        normalized.append(
            {
                "request_id": request_id,
                "incident_id": str(decision.get("incident_id") or ""),
                "condition": str(decision.get("condition") or ""),
                "model_name": str(decision.get("model_name") or ""),
                "selected_article_id": selected_article_id,
                "selected_outlet": selected_outlet,
                "selected_bucket": selected_bucket,
                "justification": str(decision.get("reason") or ""),
                "raw_response": str(decision.get("raw_response") or ""),
                "parsed_successfully": str(decision.get("parse_status") or "").lower() == "success",
                "latency_ms": int(decision.get("latency_ms") or 0),
                "timestamp_utc": str(decision.get("created_at") or ""),
                "run_id": str(decision.get("run_id") or ""),
                "parse_status": str(decision.get("parse_status") or ""),
                "error": decision.get("error"),
                "selected_position": selected_position,
                "candidate_signature": candidate_signature,
            }
        )
    return normalized


def _apply_filters(df: pd.DataFrame, model: Optional[str] = None, run_id: Optional[str] = None) -> pd.DataFrame:
    filtered = _ensure_analytics_columns(df)
    if model:
        filtered = filtered[filtered["model_name"] == model]
    if run_id:
        filtered = filtered[filtered["run_id"] == run_id]
    return filtered


def _condition_metrics(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []

    df = _ensure_analytics_columns(df)
    rows: List[dict] = []
    for condition, group in df.groupby("condition", dropna=False):
        known = group[group["selected_bucket"].isin(["left", "center", "right"])]
        distribution = known["selected_bucket"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "condition": str(condition),
                "count": int(len(group)),
                "parse_success_rate": _safe_float(group["parsed_successfully"].mean()),
                "avg_latency_ms": _safe_float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "p95_latency_ms": _safe_float(pd.to_numeric(group["latency_ms"], errors="coerce").quantile(0.95)),
                "left_ratio": _safe_float(distribution.get("left", 0.0)),
                "center_ratio": _safe_float(distribution.get("center", 0.0)),
                "right_ratio": _safe_float(distribution.get("right", 0.0)),
                "unknown_ratio": _safe_float((group["selected_bucket"] == "unknown").mean()),
                "mean_selected_position": _safe_float(pd.to_numeric(group["selected_position"], errors="coerce").mean()),
            }
        )
    return sorted(rows, key=lambda row: row["condition"])


def _condition_metrics_by_model(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []

    df = _ensure_analytics_columns(df)
    rows: List[dict] = []
    for (model_name, condition), group in df.groupby(["model_name", "condition"], dropna=False):
        known = group[group["selected_bucket"].isin(["left", "center", "right"])]
        distribution = known["selected_bucket"].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "model": str(model_name),
                "condition": str(condition),
                "count": int(len(group)),
                "parse_success_rate": _safe_float(group["parsed_successfully"].mean()),
                "avg_latency_ms": _safe_float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "p95_latency_ms": _safe_float(pd.to_numeric(group["latency_ms"], errors="coerce").quantile(0.95)),
                "left_ratio": _safe_float(distribution.get("left", 0.0)),
                "center_ratio": _safe_float(distribution.get("center", 0.0)),
                "right_ratio": _safe_float(distribution.get("right", 0.0)),
                "unknown_ratio": _safe_float((group["selected_bucket"] == "unknown").mean()),
                "mean_selected_position": _safe_float(pd.to_numeric(group["selected_position"], errors="coerce").mean()),
            }
        )
    return sorted(rows, key=lambda row: (row["condition"], row["model"]))


def _run_summaries(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    df = _ensure_analytics_columns(df)
    rows: List[dict] = []
    for run_id, group in df.groupby("run_id", dropna=False):
        rows.append(
            {
                "run_id": str(run_id),
                "count": int(len(group)),
                "models": int(group["model_name"].nunique()),
                "incidents": int(group["incident_id"].nunique()),
                "parse_success_rate": _safe_float(group["parsed_successfully"].mean()),
                "avg_latency_ms": _safe_float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "first_timestamp_utc": str(group["timestamp_utc"].dropna().min() or ""),
                "last_timestamp_utc": str(group["timestamp_utc"].dropna().max() or ""),
            }
        )
    return sorted(rows, key=lambda row: row["run_id"])


def _top_outlets(df: pd.DataFrame, limit: int = 15) -> List[dict]:
    if df.empty:
        return []
    df = _ensure_analytics_columns(df)
    counts = (
        df[df["selected_outlet"].fillna("").astype(str) != ""]
        .groupby("selected_outlet")
        .size()
        .sort_values(ascending=False)
        .head(limit)
    )
    return [{"selected_outlet": str(outlet), "count": int(count)} for outlet, count in counts.items()]


def _top_outlets_by_model(df: pd.DataFrame, limit: int = 10) -> List[dict]:
    if df.empty:
        return []
    df = _ensure_analytics_columns(df)
    df = df[df["selected_outlet"].fillna("").astype(str) != ""]
    if df.empty:
        return []

    rows: List[dict] = []
    for model_name, group in df.groupby("model_name", dropna=False):
        counts = group.groupby("selected_outlet").size().sort_values(ascending=False).head(limit)
        for outlet, count in counts.items():
            rows.append(
                {
                    "model": str(model_name),
                    "selected_outlet": str(outlet),
                    "count": int(count),
                }
            )
    return sorted(rows, key=lambda row: (row["model"], -row["count"], row["selected_outlet"]))


def _label_sensitivity_rate(df: pd.DataFrame) -> float:
    required_conditions = {"headlines_sources", "swapped_sources"}
    scoped = df[df["condition"].isin(required_conditions)].copy()
    scoped = scoped[scoped["selected_bucket"].isin(["left", "center", "right"])]
    if scoped.empty:
        return 0.0

    key_cols = ["run_id", "model_name", "incident_id", "candidate_signature"]
    pivot = scoped.pivot_table(index=key_cols, columns="condition", values="selected_bucket", aggfunc="first")
    if "headlines_sources" not in pivot.columns or "swapped_sources" not in pivot.columns:
        return 0.0
    pivot = pivot.dropna(subset=["headlines_sources", "swapped_sources"])
    if pivot.empty:
        return 0.0
    changed = (pivot["headlines_sources"] != pivot["swapped_sources"]).astype(float)
    return _safe_float(changed.mean())


def _cross_model_agreement(df: pd.DataFrame) -> tuple[float, float]:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return 0.0, 0.0

    agreements: List[float] = []
    entropies: List[float] = []
    for _, group in known.groupby(["run_id", "incident_id", "condition"], dropna=False):
        n = len(group)
        if n < 2:
            continue
        probs = group["selected_bucket"].value_counts(normalize=True)
        agreements.append(float(probs.max()))
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum()) / float(np.log2(3))
        entropies.append(max(0.0, min(1.0, entropy)))

    if not agreements:
        return 0.0, 0.0
    return _safe_float(float(np.mean(agreements))), _safe_float(float(np.mean(entropies)))


def _model_instability_score(df: pd.DataFrame) -> float:
    known = df[df["selected_bucket"].isin(["left", "center", "right"])].copy()
    if known.empty:
        return 0.0

    model_scores: List[float] = []
    for _, model_group in known.groupby("model_name", dropna=False):
        incident_scores: List[float] = []
        for _, group in model_group.groupby(["run_id", "incident_id"], dropna=False):
            n_conditions = int(group["condition"].nunique())
            if n_conditions < 2:
                continue
            n_unique = int(group["selected_bucket"].nunique())
            incident_scores.append((n_unique - 1) / (n_conditions - 1))
        if incident_scores:
            model_scores.append(float(np.mean(incident_scores)))

    if not model_scores:
        return 0.0
    return _safe_float(float(np.mean(model_scores)))


def _sample_records(df: pd.DataFrame, limit: int = 100) -> List[dict]:
    if df.empty:
        return []
    df = _ensure_analytics_columns(df)
    cols = [
        "run_id",
        "incident_id",
        "condition",
        "model_name",
        "selected_article_id",
        "selected_outlet",
        "selected_bucket",
        "parse_status",
        "latency_ms",
        "selected_position",
        "timestamp_utc",
    ]
    available_cols = [col for col in cols if col in df.columns]
    rows = df.sort_values("timestamp_utc", ascending=False)[available_cols].head(limit).to_dict(orient="records")
    return _json_safe_records(rows)


def _ingest_run_directory(
    run_dir: Path,
    model_decisions_file: str,
    experiment_requests_file: str,
    db_file: Path = DB_FILE,
) -> dict:
    model_decisions_path = run_dir / model_decisions_file
    experiment_requests_path = run_dir / experiment_requests_file

    if not model_decisions_path.exists():
        raise HTTPException(status_code=400, detail=f"Missing file: {model_decisions_path}")
    if not experiment_requests_path.exists():
        raise HTTPException(status_code=400, detail=f"Missing file: {experiment_requests_path}")

    model_decisions_rows = _read_jsonl(model_decisions_path)
    request_rows = _read_jsonl(experiment_requests_path)
    request_index = _build_request_index(request_rows)
    normalized_rows = _normalize_generated_rows(model_decisions_rows, request_index)

    global_df = load_db(db_file)
    before_count = len(global_df)
    new_data = pd.DataFrame(normalized_rows)
    updated_db = pd.concat([global_df, new_data], ignore_index=True).drop_duplicates(subset=['request_id'])
    save_to_db(updated_db, db_file)

    return {
        "run_dir": str(run_dir),
        "records_seen": len(normalized_rows),
        "records_added": int(len(updated_db) - before_count),
        "total_records": len(updated_db),
    }


def sync_outputs_to_db(
    outputs_dir: Path = OUTPUTS_DIR,
    *,
    db_file: Path = DB_FILE,
    run_dir_prefix: str = "run_",
    model_decisions_file: str = "model_decisions.jsonl",
    experiment_requests_file: str = "experiment_requests.jsonl",
) -> dict:
    outputs_dir = Path(outputs_dir)
    db_file = Path(db_file)

    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return {
            "status": "skipped",
            "outputs_dir": str(outputs_dir),
            "db_file": str(db_file),
            "runs_found": 0,
            "runs_ingested": 0,
            "records_seen": 0,
            "records_added": 0,
            "results": [],
        }

    run_dirs = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and d.name.startswith(run_dir_prefix)
    )

    results: List[dict] = []
    records_seen = 0
    records_added = 0
    runs_ingested = 0

    for run_dir in run_dirs:
        has_required = (
            (run_dir / model_decisions_file).exists()
            and (run_dir / experiment_requests_file).exists()
        )
        if not has_required:
            continue

        result = _ingest_run_directory(
            run_dir=run_dir,
            model_decisions_file=model_decisions_file,
            experiment_requests_file=experiment_requests_file,
            db_file=db_file,
        )
        results.append(result)
        records_seen += result["records_seen"]
        records_added += result["records_added"]
        runs_ingested += 1

    return {
        "status": "success",
        "outputs_dir": str(outputs_dir),
        "db_file": str(db_file),
        "runs_found": len(run_dirs),
        "runs_ingested": runs_ingested,
        "records_seen": records_seen,
        "records_added": records_added,
        "results": results,
    }


# --- ENDPOINTS ---
@app.post("/ingest")
async def ingest_results(results: List[ExperimentResult]):
    _require_write_access()
    global_df = load_db()
    new_data = pd.DataFrame([r.model_dump() for r in results])
    updated_db = pd.concat([global_df, new_data], ignore_index=True).drop_duplicates(subset=['request_id'])
    save_to_db(updated_db)
    return {"status": "success", "total_records": len(updated_db)}


@app.post("/ingest/run")
async def ingest_run_outputs(payload: IngestRunRequest):
    _require_write_access()
    run_dir = Path(payload.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"run_dir not found: {run_dir}")
    result = _ingest_run_directory(
        run_dir=run_dir,
        model_decisions_file=payload.model_decisions_file,
        experiment_requests_file=payload.experiment_requests_file,
    )

    return {
        "status": "success",
        "ingested": result["records_seen"],
        "added": result["records_added"],
        "total_records": result["total_records"],
        "run_dir": result["run_dir"],
    }


@app.post("/ingest/runs")
async def ingest_all_runs(payload: IngestRunsRequest):
    _require_write_access()
    outputs_dir = Path(payload.outputs_dir)
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"outputs_dir not found: {outputs_dir}")

    run_dirs = sorted(
        [
            d for d in outputs_dir.iterdir()
            if d.is_dir() and d.name.startswith(payload.run_dir_prefix)
        ]
    )

    if not run_dirs:
        return {
            "status": "success",
            "outputs_dir": str(outputs_dir),
            "runs_found": 0,
            "runs_ingested": 0,
            "records_seen": 0,
            "records_added": 0,
            "results": [],
        }

    results: List[dict] = []
    records_seen = 0
    records_added = 0
    for run_dir in run_dirs:
        has_required = (run_dir / payload.model_decisions_file).exists() and (run_dir / payload.experiment_requests_file).exists()
        if not has_required:
            continue
        result = _ingest_run_directory(
            run_dir=run_dir,
            model_decisions_file=payload.model_decisions_file,
            experiment_requests_file=payload.experiment_requests_file,
        )
        results.append(result)
        records_seen += result["records_seen"]
        records_added += result["records_added"]

    return {
        "status": "success",
        "outputs_dir": str(outputs_dir),
        "runs_found": len(run_dirs),
        "runs_ingested": len(results),
        "records_seen": records_seen,
        "records_added": records_added,
        "results": results,
    }


@app.get("/metrics/summary")
async def get_summary(model: Optional[str] = None, run_id: Optional[str] = None):
    df = load_db()
    df = _apply_filters(df, model=model, run_id=run_id)
    return {"metrics": calculate_all_metrics(df), "count": len(df)}


@app.get("/metrics/inter-model")
async def get_inter_model(run_id: Optional[str] = None):
    df = load_db()
    df = _apply_filters(df, run_id=run_id)
    if df.empty:
        return {"error": "No data"}
    return {
        model: {
            **calculate_all_metrics(df[df['model_name'] == model]),
            "record_count": int(len(df[df["model_name"] == model])),
        }
        for model in sorted(df['model_name'].dropna().astype(str).unique())
    }


@app.get("/metrics/conditions")
async def get_condition_metrics(model: Optional[str] = None, run_id: Optional[str] = None):
    df = load_db()
    df = _apply_filters(df, model=model, run_id=run_id)
    return {"rows": _condition_metrics(df)}


@app.get("/metrics/conditions-by-model")
async def get_condition_metrics_by_model(run_id: Optional[str] = None):
    df = load_db()
    df = _apply_filters(df, run_id=run_id)
    return {"rows": _condition_metrics_by_model(df)}


@app.get("/metrics/run-summaries")
async def get_run_summaries(model: Optional[str] = None):
    df = load_db()
    df = _apply_filters(df, model=model)
    return {"rows": _run_summaries(df)}


@app.get("/metrics/top-outlets")
async def get_top_outlets(model: Optional[str] = None, run_id: Optional[str] = None, limit: int = 15):
    df = load_db()
    df = _apply_filters(df, model=model, run_id=run_id)
    return {"rows": _top_outlets(df, limit=limit)}


@app.get("/metrics/top-outlets-by-model")
async def get_top_outlets_by_model(run_id: Optional[str] = None, limit: int = 10):
    df = load_db()
    df = _apply_filters(df, run_id=run_id)
    return {"rows": _top_outlets_by_model(df, limit=limit)}


@app.get("/metrics/records")
async def get_records(model: Optional[str] = None, run_id: Optional[str] = None, limit: int = 100):
    df = load_db()
    df = _apply_filters(df, model=model, run_id=run_id)
    return {"rows": _sample_records(df, limit=limit)}


@app.get("/metrics/runs")
async def get_run_ids():
    df = load_db()
    if df.empty or "run_id" not in df.columns:
        return {"runs": []}
    runs = [r for r in df["run_id"].dropna().astype(str).unique().tolist() if r]
    return {"runs": sorted(runs)}


@app.get("/metrics/compare-runs")
async def compare_runs(run_a: str, run_b: str, model: Optional[str] = None):
    if not run_a or not run_b:
        raise HTTPException(status_code=400, detail="run_a and run_b are required")

    df = load_db()
    if df.empty:
        raise HTTPException(status_code=404, detail="No analytics data available")

    scoped = _apply_filters(df, model=model)
    a_df = _apply_filters(scoped, run_id=run_a)
    b_df = _apply_filters(scoped, run_id=run_b)
    if a_df.empty or b_df.empty:
        raise HTTPException(status_code=404, detail="One or both runs were not found in the selected scope")

    a_metrics = calculate_all_metrics(a_df)
    b_metrics = calculate_all_metrics(b_df)
    comparable_keys = [
        "parse_success_rate",
        "parse_failure_rate",
        "avg_latency_ms",
        "p95_latency_ms",
        "center_preference_index",
        "content_robustness_score",
        "label_sensitivity_rate",
        "cross_model_agreement_rate",
        "cross_model_instability",
        "model_instability_score",
    ]
    delta = {
        key: _safe_float(b_metrics.get(key, 0.0) - a_metrics.get(key, 0.0))
        for key in comparable_keys
    }

    return {
        "scope": {"run_a": run_a, "run_b": run_b, "model": model or "ALL"},
        "run_a": {"count": int(len(a_df)), "metrics": a_metrics},
        "run_b": {"count": int(len(b_df)), "metrics": b_metrics},
        "delta_run_b_minus_run_a": delta,
    }


@app.get("/export/csv")
async def export_csv():
    return {"file_path": DB_FILE.name}


@app.get("/healthz")
async def healthcheck():
    return {
        "status": "ok",
        "db_exists": DB_FILE.exists(),
        "write_endpoints_enabled": WRITE_ENDPOINTS_ENABLED,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
