import pandas as pd
import logging
import json
from configs.config import INCIDENTS_JSONL, OUTPUT_DIR, LOG_LEVEL
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article-level checks (articles_clean.parquet)
# ---------------------------------------------------------------------------

def check_missing_values(df: pd.DataFrame):
    log.info("Checking missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("  No missing values detected.")
    else:
        print("\n  Missing values:")
        print(missing.to_string())


def check_bias_distribution(df: pd.DataFrame):
    print("\n  Bias distribution:")
    counts = df["bias_text"].value_counts()
    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"    {label:<10} {count:>6,}  ({pct:.2f}%)")


def check_duplicates(df: pd.DataFrame):
    dup_titles = df["title"].duplicated().sum()
    print(f"\n  Duplicate titles : {dup_titles:,}")


def check_content_length(df: pd.DataFrame):
    df = df.copy()
    df["content_len"] = df["content_original"].str.len()
    print("\n  Content length (content_original):")
    print(f"    mean   : {int(df['content_len'].mean()):,}")
    print(f"    median : {int(df['content_len'].median()):,}")
    print(f"    min    : {int(df['content_len'].min()):,}")
    print(f"    max    : {int(df['content_len'].max()):,}")
    short = (df["content_len"] < 200).sum()
    print(f"\n  Articles < 200 chars : {short:,}")


def run_article_quality_checks(df: pd.DataFrame):
    print("\n===== ARTICLE QUALITY REPORT =====")
    print(f"  Dataset size    : {len(df):,}")
    print(f"  Unique sources  : {df['source'].nunique():,}")
    print(f"  Unique topics   : {df['topic'].nunique():,}")
    check_missing_values(df)
    check_bias_distribution(df)
    check_duplicates(df)
    check_content_length(df)


# ---------------------------------------------------------------------------
# Incident-level checks (prepared_incidents.jsonl)
# ---------------------------------------------------------------------------

def check_incidents_jsonl(path: Path):
    """
    Load prepared_incidents.jsonl and report quality metrics for the
    incident clustering output that feeds the API and Model pipeline.
    """
    if not path.exists():
        print(f"\n  [SKIP] {path} not found. Run build_incidents.py first.")
        return

    incidents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                incidents.append(json.loads(line))

    if not incidents:
        print("\n  [WARN] prepared_incidents.jsonl is empty.")
        return

    # Counts
    total = len(incidents)
    total_articles = sum(len(inc["articles"]) for inc in incidents)

    # Leaning coverage
    all_three = sum(
        1 for inc in incidents
        if len({a["leaning"] for a in inc["articles"]}) == 3
    )
    two_only = total - all_three

    # Articles per incident
    sizes = [len(inc["articles"]) for inc in incidents]
    avg_size = sum(sizes) / total

    # Topic distribution
    from collections import Counter
    topic_counts = Counter(inc.get("topic", "unknown") for inc in incidents)

    # Neutral summary presence
    missing_summary = sum(1 for inc in incidents if not inc.get(
        "neutral_summary", "").strip())

    print("\n===== INCIDENT QUALITY REPORT =====")
    print(f"  Total incidents          : {total:,}")
    print(f"  Total articles in output : {total_articles:,}")
    print(f"  Avg articles / incident  : {avg_size:.1f}")
    print(f"  Min articles / incident  : {min(sizes)}")
    print(f"  Max articles / incident  : {max(sizes)}")

    print(f"\n  Leaning coverage:")
    print(
        f"    All 3 leanings         : {all_three:,}  ({100 * all_three / total:.1f}%)")
    print(
        f"    2 leanings only        : {two_only:,}  ({100 * two_only / total:.1f}%)")

    print(f"\n  Missing neutral_summary  : {missing_summary}")

    print(f"\n  Top 10 topics:")
    for topic, count in topic_counts.most_common(10):
        pct = 100 * count / total
        print(f"    {topic:<30} : {count:>5,}  ({pct:.1f}%)")

    # Sample 3 incidents for manual inspection
    print("\n  --- Sample incidents (first 3) ---")
    for inc in incidents[:3]:
        leanings = [a["leaning"] for a in inc["articles"]]
        print(f"\n  ID      : {inc['incident_id']}")
        print(f"  Topic   : {inc.get('topic', '')}")
        print(f"  Summary : {inc.get('neutral_summary', '')[:120]}")
        print(f"  Articles: {len(inc['articles'])} ({', '.join(leanings)})")
        for a in inc["articles"]:
            print(
                f"    [{a['leaning']:^6}] {a['article_id']}  |  {a['headline'][:80]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path = Path(OUTPUT_DIR)

    # --- Article-level check ---
    articles_path = output_path / "articles_clean.parquet"
    if articles_path.exists():
        log.info(f"Loading {articles_path}")
        df = pd.read_parquet(articles_path)
        run_article_quality_checks(df)
    else:
        print(f"\n[SKIP] articles_clean.parquet not found at {articles_path}.")
        print("       Run load_articles.py first.")

    # --- Incident-level check ---
    incidents_path = output_path / INCIDENTS_JSONL
    check_incidents_jsonl(incidents_path)


if __name__ == "__main__":
    main()
