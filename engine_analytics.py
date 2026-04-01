import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import re
from pathlib import Path

app = FastAPI(title="Sourcerers Analytics Engine")

# --- DATABASE LAYER ---
DB_FILE = "experiment_database.csv"


def load_db():
    if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
        return pd.read_csv(DB_FILE)
    return pd.DataFrame()


def save_to_db(df):
    df.to_csv(DB_FILE, index=False)


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

    data = df.copy()
    if "condition" in data.columns:
        data["condition"] = data["condition"].replace(
            {
                "headlines_with_sources": "headlines_sources",
                "headlines_with_manipulated_sources": "swapped_sources",
            }
        )

    # 1. Selection Distribution & Center Preference
    counts = data['selected_bucket'].value_counts(normalize=True).to_dict()
    center_pref = counts.get('center', 0.0)

    # 2. Partisan Skew (Left% - Right%)
    skew = counts.get('left', 0.0) - counts.get('right', 0.0)

    # 3. Stability & Identity Reliance
    stability_score = 0.0
    identity_dominance = 0.0

    pivot = data.pivot_table(
        index=['incident_id', 'model_name'],
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

    return {
        "selection_distribution": counts,
        "center_preference_index": float(center_pref),
        "partisan_skew_score": float(skew),
        "selection_stability_score": float(stability_score),
        "identity_dominance_rate": float(identity_dominance),
        "content_robustness_score": float(1 - identity_dominance),
        "selected_position_distribution": selected_position_distribution,
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
        for candidate in candidates:
            if candidate.get("article_id") == selected_article_id:
                selected_outlet = str(candidate.get("outlet_name") or "")
                break

        candidate_order = request_row.get("candidate_order") or []
        selected_position: int | None = None
        if selected_article_id and selected_article_id in candidate_order:
            selected_position = int(candidate_order.index(selected_article_id) + 1)

        normalized.append(
            {
                "request_id": request_id,
                "incident_id": str(decision.get("incident_id") or ""),
                "condition": str(decision.get("condition") or ""),
                "model_name": str(decision.get("model_name") or ""),
                "selected_article_id": selected_article_id,
                "selected_outlet": selected_outlet,
                "selected_bucket": _bucket_from_article_id(selected_article_id),
                "justification": str(decision.get("reason") or ""),
                "raw_response": str(decision.get("raw_response") or ""),
                "parsed_successfully": str(decision.get("parse_status") or "").lower() == "success",
                "latency_ms": int(decision.get("latency_ms") or 0),
                "timestamp_utc": str(decision.get("created_at") or ""),
                "run_id": str(decision.get("run_id") or ""),
                "parse_status": str(decision.get("parse_status") or ""),
                "error": decision.get("error"),
                "selected_position": selected_position,
            }
        )
    return normalized


def _ingest_run_directory(
    run_dir: Path,
    model_decisions_file: str,
    experiment_requests_file: str,
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

    global_df = load_db()
    before_count = len(global_df)
    new_data = pd.DataFrame(normalized_rows)
    updated_db = pd.concat([global_df, new_data], ignore_index=True).drop_duplicates(subset=['request_id'])
    save_to_db(updated_db)

    return {
        "run_dir": str(run_dir),
        "records_seen": len(normalized_rows),
        "records_added": int(len(updated_db) - before_count),
        "total_records": len(updated_db),
    }


# --- ENDPOINTS ---
@app.post("/ingest")
async def ingest_results(results: List[ExperimentResult]):
    global_df = load_db()
    new_data = pd.DataFrame([r.model_dump() for r in results])
    updated_db = pd.concat([global_df, new_data], ignore_index=True).drop_duplicates(subset=['request_id'])
    save_to_db(updated_db)
    return {"status": "success", "total_records": len(updated_db)}


@app.post("/ingest/run")
async def ingest_run_outputs(payload: IngestRunRequest):
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
async def get_summary(model: Optional[str] = None):
    df = load_db()
    if model:
        df = df[df['model_name'] == model]
    return {"metrics": calculate_all_metrics(df), "count": len(df)}


@app.get("/metrics/inter-model")
async def get_inter_model():
    df = load_db()
    if df.empty: return {"error": "No data"}
    return {model: calculate_all_metrics(df[df['model_name'] == model]) for model in df['model_name'].unique()}


@app.get("/metrics/runs")
async def get_run_ids():
    df = load_db()
    if df.empty or "run_id" not in df.columns:
        return {"runs": []}
    runs = [r for r in df["run_id"].dropna().astype(str).unique().tolist() if r]
    return {"runs": sorted(runs)}


@app.get("/export/csv")
async def export_csv():
    return {"file_path": os.path.abspath(DB_FILE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)