from configs.config import OUTPUT_DIR, LOG_LEVEL
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame):

    log.info("Checking missing values")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values detected")
    else:
        print("\nMissing values:")
        print(missing)


def check_bias_distribution(df: pd.DataFrame):

    print("\nBias distribution")

    counts = df["bias_text"].value_counts()

    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"{label:<10} {count:>6} ({pct:.2f}%)")


def check_duplicates(df: pd.DataFrame):

    dup_titles = df["title"].duplicated().sum()

    print("\nDuplicate titles:", dup_titles)


def check_content_length(df: pd.DataFrame):

    df["content_len"] = df["content_original"].str.len()

    print("\nContent length stats")
    print("mean   :", int(df["content_len"].mean()))
    print("median :", int(df["content_len"].median()))
    print("min    :", int(df["content_len"].min()))
    print("max    :", int(df["content_len"].max()))

    short = (df["content_len"] < 200).sum()

    print("\nArticles < 200 chars:", short)


def run_quality_checks(df: pd.DataFrame):

    print("\n===== DATASET QUALITY REPORT =====")

    print("\nDataset size:", len(df))
    print("Unique sources:", df["source"].nunique())
    print("Unique topics:", df["topic"].nunique())

    check_missing_values(df)
    check_bias_distribution(df)
    check_duplicates(df)
    check_content_length(df)


def main():

    input_path = Path(OUTPUT_DIR) / "articles_clean.parquet"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run load_articles.py first to generate articles_clean.parquet"
        )

    log.info("Loading dataset")

    df = pd.read_parquet(input_path)

    run_quality_checks(df)


if __name__ == "__main__":
    main()