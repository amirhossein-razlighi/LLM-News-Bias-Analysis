from tqdm import tqdm
import pandas as pd
from typing import Optional
import os
import logging
import json
from configs.config import (
    BIAS_LABEL_MAP,
    BIAS_TEXT_MAP,
    JSONS_DIR,
    OUTPUT_DIR,
    LOG_LEVEL,
)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_single_article(path: Path) -> Optional[dict]:
    """
    Load and parse one JSON article file.
    Returns None on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"Failed to load {path.name}: {e}")
        return None


def validate_and_clean(record: dict) -> Optional[dict]:
    """
    Validate a single article record.
    Returns cleaned dict or None if the article should be dropped.
    """
    cleaned = {}

    article_id = record.get("ID") or record.get("id")
    if not article_id:
        return None
    cleaned["id"] = str(article_id).strip()

    cleaned["topic"] = str(record.get("topic", "")
                           ).strip().lower() or "unknown"

    cleaned["source"] = str(record.get("source", "")).strip()
    cleaned["source_url"] = str(record.get("source_url", "")).strip()
    cleaned["url"] = str(record.get("url", "")).strip()

    cleaned["date"] = str(record.get("date", "")).strip()

    cleaned["authors"] = str(record.get("authors", "")).strip()

    title = str(record.get("title", "")).strip()
    if not title:
        return None
    cleaned["title"] = title

    content_original = str(record.get("content_original", "")).strip()
    content = str(record.get("content", "")).strip()

    cleaned["content_original"] = content_original
    cleaned["content"] = content

    if not content_original and not content:
        return None

    bias_text = str(record.get("bias_text", "")).strip().lower()
    bias_num = record.get("bias")

    if bias_text in BIAS_TEXT_MAP:
        cleaned["bias_text"] = bias_text
        cleaned["bias"] = BIAS_TEXT_MAP[bias_text]
    elif bias_num is not None and int(bias_num) in BIAS_LABEL_MAP:
        cleaned["bias"] = int(bias_num)
        cleaned["bias_text"] = BIAS_LABEL_MAP[int(bias_num)]
    else:
        return None

    return cleaned


def load_all_articles(jsons_dir: str = JSONS_DIR) -> pd.DataFrame:
    """
    Load all JSON articles from jsons_dir.
    Returns a clean DataFrame indexed by article ID.
    """
    jsons_path = Path(jsons_dir)
    if not jsons_path.exists():
        raise FileNotFoundError(
            f"JSONs directory not found: {jsons_path}\n"
            "Make sure you've set DATASET_ROOT correctly (e.g. DATASET_ROOT=. python data_prep/load_articles.py)."
        )

    json_files = sorted(jsons_path.glob("*.json"))
    log.info(f"Found {len(json_files)} JSON files in {jsons_path}")

    records = []
    skipped = 0

    for fp in tqdm(json_files, desc="Loading articles"):
        raw = load_single_article(fp)
        if raw is None:
            skipped += 1
            continue

        cleaned = validate_and_clean(raw)
        if cleaned is None:
            skipped += 1
            continue

        records.append(cleaned)

    log.info(f"Loaded {len(records)} valid articles | Skipped {skipped}")

    df = pd.DataFrame(records)
    df = df.set_index("id")
    df = df[~df.index.duplicated(keep="first")]

    log.info(f"Final article count after deduplication: {len(df)}")
    return df


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print a summary of the loaded dataset."""
    bias_counts = df["bias_text"].value_counts()
    for label, count in bias_counts.items():
        pct = 100 * count / len(df)
        print(f"{label:<10}: {count:>6,} ({pct:.1f}%)")

    print(f"\nUnique sources : {df['source'].nunique()}")
    print(f"Unique topics  : {df['topic'].nunique()}")

    print(f"\nTop 10 topics:")
    for topic, count in df["topic"].value_counts().head(10).items():
        print(f"  {topic:<30}: {count}")

    df["content_len"] = df["content_original"].str.len()
    print(f"\nContent length (content_original):")
    print(f"  Mean   : {df['content_len'].mean():.0f}")
    print(f"  Median : {df['content_len'].median():.0f}")
    print(f"  Min    : {df['content_len'].min()}")
    print(f"  Max    : {df['content_len'].max()}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "articles_clean.parquet")

    log.info("Starting article loading")
    df = load_all_articles()

    print_dataset_stats(df)

    df.to_parquet(out_path, index=True)
    log.info(f"Saved clean articles to {out_path}")

    return df


if __name__ == "__main__":
    main()
