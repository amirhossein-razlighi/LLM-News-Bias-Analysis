from configs.config import OUTPUT_DIR, LOG_LEVEL
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build_bias_balanced_bundles(
    df: pd.DataFrame,
    bundle_size: int = 12,
) -> List[pd.DataFrame]:
    """
    Create bias-balanced bundles.

    Each bundle contains roughly equal numbers of each bias class.
    """

    bundles = []

    grouped = df.groupby("bias")

    min_class_size = min(len(g) for _, g in grouped)

    per_class = bundle_size // len(grouped)

    log.info(f"Creating bundles with {per_class} samples per bias class")

    for start in range(0, min_class_size, per_class):

        parts = []

        for _, group in grouped:
            part = group.iloc[start:start + per_class]
            if len(part) == per_class:
                parts.append(part)

        if len(parts) == len(grouped):
            bundle = pd.concat(parts).sample(frac=1).reset_index()
            bundles.append(bundle)

    log.info(f"Built {len(bundles)} bundles")

    return bundles


def save_bundles(bundles: List[pd.DataFrame], out_dir: Path):

    os.makedirs(out_dir, exist_ok=True)

    for i, bundle in enumerate(bundles):
        path = out_dir / f"bundle_{i:04d}.parquet"
        bundle.to_parquet(path, index=False)

    log.info(f"Saved {len(bundles)} bundles to {out_dir}")


def main():

    input_path = Path(OUTPUT_DIR) / "articles_clean.parquet"
    bundle_dir = Path(OUTPUT_DIR) / "bundles"

    if not input_path.exists():
        raise FileNotFoundError(
            "Run load_articles.py first to generate articles_clean.parquet"
        )

    log.info("Loading cleaned articles")
    df = pd.read_parquet(input_path)

    bundles = build_bias_balanced_bundles(df)

    save_bundles(bundles, bundle_dir)


if __name__ == "__main__":
    main()