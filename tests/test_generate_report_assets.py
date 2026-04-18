from __future__ import annotations

import pandas as pd

from app.cli.generate_report_assets import build_model_summary, infer_bucket, write_summary_table


def test_infer_bucket_from_article_id() -> None:
    assert infer_bucket("incident_left_1") == "left"
    assert infer_bucket("incident_center_1") == "center"
    assert infer_bucket("incident_right_1") == "right"
    assert infer_bucket("incident_misc_1") == "unknown"


def test_build_model_summary_fields() -> None:
    df = pd.DataFrame(
        [
            {
                "model_name": "m1",
                "parse_status": "success",
                "latency_ms": 1000,
                "selected_bucket": "center",
            },
            {
                "model_name": "m1",
                "parse_status": "failed",
                "latency_ms": 3000,
                "selected_bucket": "left",
            },
            {
                "model_name": "m2",
                "parse_status": "fallback",
                "latency_ms": 2000,
                "selected_bucket": "right",
            },
        ]
    )

    summary = build_model_summary(df, bootstrap_samples=200, bootstrap_seed=42)
    row_m1 = summary[summary["model"] == "m1"].iloc[0]

    assert int(row_m1["n"]) == 2
    assert abs(float(row_m1["parse_success_rate"]) - 0.5) < 1e-9
    assert abs(float(row_m1["parse_failure_rate"]) - 0.5) < 1e-9
    assert abs(float(row_m1["avg_latency_ms"]) - 2000.0) < 1e-9
    assert 0.0 <= float(row_m1["parse_success_ci95_low"]) <= 1.0
    assert 0.0 <= float(row_m1["parse_success_ci95_high"]) <= 1.0


def test_write_summary_table_creates_markdown(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "m1",
                "n": 5,
                "parse_success_rate": 1.0,
                "parse_fallback_rate": 0.0,
                "parse_failure_rate": 0.0,
                "avg_latency_ms": 1234.0,
                "p95_latency_ms": 1500.0,
                "center_selection_rate": 0.4,
            }
        ]
    )

    out = tmp_path / "summary.md"
    write_summary_table(summary, out)
    text = out.read_text(encoding="utf-8")

    assert "| model | n |" in text
    assert "m1" in text
    assert "100.00%" in text