from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Optional
from collections import defaultdict
import os
import logging
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from configs.config import (
        COMMUNITY_DETECTION_THRESHOLD,
        INCIDENTS_JSONL,
        MAX_ARTICLES_PER_LEANING,
        MIN_BIAS_DIVERSITY,
        MIN_COMMUNITY_SIZE,
        OUTPUT_DIR,
        LOG_LEVEL,
    )
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(output_dir: str = OUTPUT_DIR):
    """
    Load the three artefacts produced by the earlier pipeline steps.
    Returns (df, embeddings, ordered_ids).
    """
    output_path = Path(output_dir)

    articles_path = output_path / "articles_clean.parquet"
    embeddings_path = output_path / "article_embeddings.npy"
    ids_path = output_path / "article_ids_ordered.txt"

    if not articles_path.exists():
        raise FileNotFoundError(
            f"Missing {articles_path}. Run load_articles.py first."
        )
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_path}. Run embed_articles.py first."
        )
    if not ids_path.exists():
        raise FileNotFoundError(
            f"Missing {ids_path}. Run embed_articles.py first."
        )

    log.info("Loading articles_clean.parquet …")
    df = pd.read_parquet(articles_path)

    log.info("Loading article_embeddings.npy …")
    embeddings = np.load(embeddings_path)

    log.info("Loading article_ids_ordered.txt …")
    with open(ids_path, "r", encoding="utf-8") as f:
        ordered_ids = [line.strip() for line in f if line.strip()]

    if len(ordered_ids) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(ordered_ids)} article IDs but {len(embeddings)} embeddings."
        )

    log.info(
        f"Loaded {len(df)} articles, {len(embeddings)} embeddings, {len(ordered_ids)} ordered IDs."
    )
    return df, embeddings, ordered_ids


# ---------------------------------------------------------------------------
# Community detection (per-topic for performance)
# ---------------------------------------------------------------------------

def _community_detection_for_group(
    indices: list[int],
    embeddings: np.ndarray,
    threshold: float,
    min_size: int,
) -> list[list[int]]:
    """
    Run sentence_transformers community detection on a subset of embeddings
    identified by `indices`. Returns clusters as lists of global indices.
    """
    try:
        from sentence_transformers.util import community_detection
        import torch
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. Install it with: pip install sentence-transformers"
        )

    if len(indices) < min_size:
        return []

    group_embeddings = embeddings[indices]
    tensor = torch.tensor(group_embeddings, dtype=torch.float32)

    clusters = community_detection(
        tensor, threshold=threshold, min_community_size=min_size)

    # Map local indices back to global embedding indices
    return [[indices[local_i] for local_i in cluster] for cluster in clusters]


def detect_incidents(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    ordered_ids: list[str],
    threshold: float = COMMUNITY_DETECTION_THRESHOLD,
    min_size: int = MIN_COMMUNITY_SIZE,
) -> list[list[str]]:
    """
    Detect incident clusters by running community detection within each topic
    group. Grouping by topic dramatically reduces the O(n²) similarity cost.

    Returns a list of clusters, each cluster being a list of article IDs.
    """
    # Build a mapping from article_id → position in the embeddings array
    id_to_pos = {aid: i for i, aid in enumerate(ordered_ids)}

    # Only keep articles present in both the DataFrame and the embeddings
    valid_ids = [aid for aid in ordered_ids if aid in df.index]
    log.info(
        f"{len(valid_ids)} articles are present in both DataFrame and embeddings.")

    # Group valid article IDs by topic
    topic_groups: dict[str, list[int]] = defaultdict(list)
    for aid in valid_ids:
        topic = df.at[aid, "topic"]
        topic_groups[topic].append(id_to_pos[aid])

    log.info(
        f"Running community detection across {len(topic_groups)} topic groups …")

    all_clusters: list[list[str]] = []
    for topic, indices in tqdm(topic_groups.items(), desc="Topics"):
        clusters = _community_detection_for_group(
            indices, embeddings, threshold, min_size)
        for cluster_indices in clusters:
            cluster_ids = [ordered_ids[i] for i in cluster_indices]
            all_clusters.append(cluster_ids)

    log.info(f"Found {len(all_clusters)} raw incident clusters.")
    return all_clusters


# ---------------------------------------------------------------------------
# Neutral summary (rule-based, no LLM needed)
# ---------------------------------------------------------------------------

def make_neutral_summary(incident_df: pd.DataFrame) -> str:
    """
    Build a brief neutral summary for an incident without an LLM.

    Strategy:
      1. Use the first center article's title (most likely to be descriptively neutral).
      2. Fall back to the shortest title across all articles in the incident.

    The result is capped at 300 characters.
    """
    center_rows = incident_df[incident_df["bias_text"] == "center"]
    if not center_rows.empty:
        summary = center_rows.iloc[0]["title"]
    else:
        summary = incident_df["title"].dropna().sort_values(
            key=lambda s: s.str.len()).iloc[0]

    return str(summary).strip()[:300]


# ---------------------------------------------------------------------------
# PreparedIncident builder
# ---------------------------------------------------------------------------

def build_prepared_incident(
    incident_idx: int,
    article_ids: list[str],
    df: pd.DataFrame,
    max_per_leaning: int = MAX_ARTICLES_PER_LEANING,
    min_bias_diversity: int = MIN_BIAS_DIVERSITY,
) -> Optional[dict]:
    """
    Build a PreparedIncident dict from a cluster of article IDs.

    Article IDs are generated as  inc_{N:04d}_{leaning}_{local_i}  to satisfy
    the  _bucket_from_article_id()  regex in engine_analytics.py:
        (^|_)left(_|$)  /  (^|_)center(_|$)  /  (^|_)right(_|$)

    Returns None when the cluster does not meet bias-diversity requirements.
    """
    # Filter to IDs actually present in the DataFrame
    valid_ids = [aid for aid in article_ids if aid in df.index]
    if not valid_ids:
        return None

    incident_df = df.loc[valid_ids].copy()

    # Determine topic from the most common topic in the cluster
    topic = incident_df["topic"].mode(
    ).iloc[0] if not incident_df.empty else "unknown"

    # Group by leaning
    leaning_groups: dict[str, list[str]] = defaultdict(list)
    for aid, row in incident_df.iterrows():
        leaning_groups[row["bias_text"]].append(str(aid))

    # Enforce bias-diversity filter
    if len(leaning_groups) < min_bias_diversity:
        return None

    incident_id = f"inc_{incident_idx:04d}"
    neutral_summary = make_neutral_summary(incident_df)

    articles = []
    original_ids: dict[str, str] = {}

    for leaning in ("left", "center", "right"):
        group = leaning_groups.get(leaning, [])
        for local_i, original_id in enumerate(group[:max_per_leaning]):
            new_article_id = f"{incident_id}_{leaning}_{local_i}"
            row = incident_df.loc[original_id]

            body = str(row.get("content_original")
                       or row.get("content") or "").strip()

            articles.append(
                {
                    "article_id": new_article_id,
                    "headline": str(row["title"]).strip(),
                    "body": body,
                    "outlet_name": str(row.get("source") or "").strip(),
                    "leaning": leaning,
                }
            )
            original_ids[new_article_id] = original_id

    if not articles:
        return None

    return {
        "incident_id": incident_id,
        "topic": topic,
        "neutral_summary": neutral_summary,
        "articles": articles,
        "metadata": {"original_ids": original_ids},
    }


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def save_incidents_jsonl(incidents: list[dict], path: Path) -> None:
    """Write one JSON object per line to a .jsonl file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for incident in incidents:
            f.write(json.dumps(incident, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(incidents)} incidents to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = Path(OUTPUT_DIR)

    # Step 1: load artefacts from previous pipeline steps
    df, embeddings, ordered_ids = load_data(OUTPUT_DIR)

    # Step 2: cluster articles into incidents
    clusters = detect_incidents(
        df,
        embeddings,
        ordered_ids,
        threshold=COMMUNITY_DETECTION_THRESHOLD,
        min_size=MIN_COMMUNITY_SIZE,
    )

    # Step 3: build PreparedIncident dicts
    log.info("Building PreparedIncident objects …")
    incidents: list[dict] = []
    skipped = 0
    for i, cluster in enumerate(tqdm(clusters, desc="Building incidents")):
        incident = build_prepared_incident(
            # use accepted count as sequential index
            incident_idx=len(incidents),
            article_ids=cluster,
            df=df,
            max_per_leaning=MAX_ARTICLES_PER_LEANING,
            min_bias_diversity=MIN_BIAS_DIVERSITY,
        )
        if incident is None:
            skipped += 1
            continue
        incidents.append(incident)

    log.info(
        f"Built {len(incidents)} valid incidents | Skipped {skipped} clusters "
        f"(failed bias-diversity filter)."
    )

    # Step 4: save
    out_path = output_path / INCIDENTS_JSONL
    save_incidents_jsonl(incidents, out_path)

    # Step 5: summary statistics
    if incidents:
        leaning_coverage = {"all_three": 0, "two_only": 0}
        total_articles = 0
        for inc in incidents:
            leanings = {a["leaning"] for a in inc["articles"]}
            if len(leanings) == 3:
                leaning_coverage["all_three"] += 1
            else:
                leaning_coverage["two_only"] += 1
            total_articles += len(inc["articles"])

        print("\n===== INCIDENT BUILD SUMMARY =====")
        print(f"Total incidents          : {len(incidents)}")
        print(f"  All 3 leanings         : {leaning_coverage['all_three']} "
              f"({100 * leaning_coverage['all_three'] / len(incidents):.1f}%)")
        print(f"  2 leanings only        : {leaning_coverage['two_only']} "
              f"({100 * leaning_coverage['two_only'] / len(incidents):.1f}%)")
        print(f"Total articles in output : {total_articles}")
        print(
            f"Avg articles / incident  : {total_articles / len(incidents):.1f}")
        print(f"\nOutput: {out_path}")

    return incidents


if __name__ == "__main__":
    main()
