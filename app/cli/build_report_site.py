from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML report site from generated figure assets."
    )
    parser.add_argument("--figures-dir", default="docs/figures", help="Directory containing generated report assets")
    parser.add_argument("--output-dir", default="site", help="Directory where the static site is written")
    return parser.parse_args()


def must_exist(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def copy_required_assets(figures_dir: Path, output_dir: Path) -> list[str]:
    required_files = [
        "parse_success_by_model.png",
        "latency_by_model.png",
        "condition_bucket_mix.png",
        "center_vs_baseline.png",
        "model_summary.md",
        "summary.json",
    ]

    copied: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in required_files:
        src = must_exist(figures_dir / name)
        dst = output_dir / name
        shutil.copy2(src, dst)
        copied.append(name)

    return copied


def build_html(summary: dict, model_summary_md: str) -> str:
    total_decisions = int(summary["total_decisions"])
    run_count = int(summary["run_count"])
    model_count = int(summary["model_count"])
    condition_count = int(summary["condition_count"])
    parse_success_rate = float(summary["parse_success_rate"]) * 100.0
    parse_failure_rate = float(summary["parse_failure_rate"]) * 100.0
    baseline_center_rate = float(summary["baseline_center_rate"]) * 100.0

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Sourcerers Report</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --card: #ffffff;
      --text: #1a1f36;
      --muted: #57607a;
      --accent: #1f5fbf;
      --border: #d8deea;
    }}
    body {{
      margin: 0;
      font-family: \"IBM Plex Sans\", \"Segoe UI\", Tahoma, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{ margin-top: 0; }}
    .meta {{ color: var(--muted); margin-bottom: 24px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(21, 37, 74, 0.08);
    }}
    .metric {{ font-size: 1.3rem; font-weight: 700; color: var(--accent); }}
    .label {{ font-size: 0.88rem; color: var(--muted); }}
    .figure {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin: 16px 0;
      padding: 14px;
      box-shadow: 0 4px 14px rgba(21, 37, 74, 0.08);
    }}
    .figure img {{ width: 100%; height: auto; border-radius: 8px; }}
    pre {{
      background: #0f1426;
      color: #f6f8ff;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>Sourcerers Experiment Report</h1>
    <p class=\"meta\">Auto-generated from repository outputs by CI/CD.</p>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"metric\">{total_decisions}</div><div class=\"label\">Decisions</div></div>
      <div class=\"card\"><div class=\"metric\">{run_count}</div><div class=\"label\">Runs</div></div>
      <div class=\"card\"><div class=\"metric\">{model_count}</div><div class=\"label\">Models</div></div>
      <div class=\"card\"><div class=\"metric\">{condition_count}</div><div class=\"label\">Conditions</div></div>
      <div class=\"card\"><div class=\"metric\">{parse_success_rate:.2f}%</div><div class=\"label\">Parse Success</div></div>
      <div class=\"card\"><div class=\"metric\">{parse_failure_rate:.2f}%</div><div class=\"label\">Parse Failure</div></div>
      <div class=\"card\"><div class=\"metric\">{baseline_center_rate:.2f}%</div><div class=\"label\">Center Baseline</div></div>
    </div>

    <div class=\"figure\">
      <h2>Parse Reliability by Model</h2>
      <img src=\"parse_success_by_model.png\" alt=\"Parse reliability by model\" />
    </div>

    <div class=\"figure\">
      <h2>Latency by Model</h2>
      <img src=\"latency_by_model.png\" alt=\"Latency by model\" />
    </div>

    <div class=\"figure\">
      <h2>Selection Mix by Condition</h2>
      <img src=\"condition_bucket_mix.png\" alt=\"Selection mix by condition\" />
    </div>

    <div class=\"figure\">
      <h2>Center Selection vs Baseline</h2>
      <img src=\"center_vs_baseline.png\" alt=\"Center selection vs baseline\" />
    </div>

    <div class=\"figure\">
      <h2>Model Summary</h2>
      <pre>{model_summary_md}</pre>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    output_dir = Path(args.output_dir)

    copied = copy_required_assets(figures_dir, output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    model_summary_md = (output_dir / "model_summary.md").read_text(encoding="utf-8")
    html = build_html(summary, model_summary_md)

    (output_dir / "index.html").write_text(html, encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "copied_assets": copied,
                "index": str(output_dir / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
