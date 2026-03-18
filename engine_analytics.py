import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

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


# --- ANALYTICS CALCULATIONS ---
def calculate_all_metrics(df: pd.DataFrame):
    if df.empty:
        return {}

    # 1. Selection Distribution & Center Preference
    counts = df['selected_bucket'].value_counts(normalize=True).to_dict()
    center_pref = counts.get('center', 0.0)

    # 2. Partisan Skew (Left% - Right%)
    skew = counts.get('left', 0.0) - counts.get('right', 0.0)

    # 3. Stability & Identity Reliance
    stability_score = 0.0
    identity_dominance = 0.0

    pivot = df.pivot_table(
        index=['incident_id', 'model_name'],
        columns='condition',
        values='selected_article_id',
        aggfunc='first'
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
        "content_robustness_score": float(1 - identity_dominance)
    }


# --- ENDPOINTS ---
@app.post("/ingest")
async def ingest_results(results: List[ExperimentResult]):
    global_df = load_db()
    new_data = pd.DataFrame([r.model_dump() for r in results])
    updated_db = pd.concat([global_df, new_data], ignore_index=True).drop_duplicates(subset=['request_id'])
    save_to_db(updated_db)
    return {"status": "success", "total_records": len(updated_db)}


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


@app.get("/export/csv")
async def export_csv():
    return {"file_path": os.path.abspath(DB_FILE)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)