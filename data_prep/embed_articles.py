import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import (
        CONTENT_TRUNCATE_CHARS,
        EMBEDDING_MODEL,
        EMBEDDING_TEXT_FIELD,
        OUTPUT_DIR,
        LOG_LEVEL,
    )
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build_embedding_text(row: pd.Series, field: str = EMBEDDING_TEXT_FIELD) -> str:
    """
    Build the text string to embedding for a single article row.
    """
    title = str(row.get("title", "")).strip()
    content = str(row.get("content_original", "")
                  or row.get("content", "")).strip()
    content_snippet = content[:CONTENT_TRUNCATE_CHARS]

    if field == "title":
        return title
    elif field == "content":
        return content_snippet
    else:
        return f"[TITLE] {title} [CONTENT] {content_snippet}"


def prepare_texts(df: pd.DataFrame) -> list[str]:
    """
    Build a list of embedding texts, one per article (preserving df order).
    """
    texts = []
    for _, row in df.iterrows():
        texts.append(build_embedding_text(row))
    return texts


def embed_articles(
    df: pd.DataFrame,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed all articles using SentenceTransformers.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed."
        )

    log.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    log.info(f"Preparing texts for {len(df)} articles.")
    texts = prepare_texts(df)

    log.info(f"Encoding with batch_size={batch_size}.")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    log.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    articles_path = os.path.join(OUTPUT_DIR, "articles_clean.parquet")
    if not os.path.exists(articles_path):
        raise FileNotFoundError(
            f"Could not find {articles_path}."
        )

    log.info(f"Loading articles from {articles_path}")
    df = pd.read_parquet(articles_path)
    log.info(f"Loaded {len(df)} articles")

    embeddings = embed_articles(df)

    emb_path = os.path.join(OUTPUT_DIR, "article_embeddings.npy")
    np.save(emb_path, embeddings)
    log.info(f"Saved embeddings into {emb_path}")

    ids_path = os.path.join(OUTPUT_DIR, "article_ids_ordered.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(df.index.tolist()))
    log.info(f"Saved ordered article IDs into {ids_path}")

    return embeddings


if __name__ == "__main__":
    main()
