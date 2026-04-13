from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from app.schemas.models import Article, PreparedIncident
from app.utils.io import write_jsonl


BIAS_LABEL_MAP = {
    0: "left",
    1: "center",
    2: "right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PreparedIncident JSONL from the real article JSON files under data/jsons."
    )
    parser.add_argument(
        "--json-dir",
        default="data/jsons",
        help="Directory containing article JSON files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output PreparedIncident JSONL file",
    )
    parser.add_argument(
        "--split-file",
        help="Optional TSV split file with an ID column used to filter articles before grouping",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Optional list of topics to keep",
    )
    parser.add_argument(
        "--min-per-leaning",
        type=int,
        default=3,
        help="Minimum number of left/center/right articles required per incident topic",
    )
    parser.add_argument(
        "--max-articles-per-leaning",
        type=int,
        default=8,
        help="Cap the number of articles kept per leaning within each incident topic",
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        help="Optional cap on the number of generated incidents after filtering",
    )
    return parser.parse_args()


def _load_split_ids(path: str | None) -> set[str] | None:
    if not path:
        return None

    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            article_id = (row.get("ID") or row.get("id") or "").strip()
            if article_id:
                ids.add(article_id)
    return ids


def _normalize_leaning(record: dict) -> str | None:
    bias_text = str(record.get("bias_text", "")).strip().lower()
    if bias_text in {"left", "center", "right"}:
        return bias_text

    bias_value = record.get("bias")
    if bias_value in BIAS_LABEL_MAP:
        return BIAS_LABEL_MAP[bias_value]

    return None


def _build_article(record: dict) -> Article | None:
    article_id = str(record.get("ID") or record.get("id") or "").strip()
    headline = str(record.get("title", "")).strip()
    leaning = _normalize_leaning(record)
    outlet_name = str(record.get("source", "")).strip()
    body = str(record.get("content_original") or record.get("content") or "").strip()

    if not article_id or not headline or not leaning or not outlet_name:
        return None

    return Article(
        article_id=article_id,
        headline=headline,
        body=body or None,
        outlet_name=outlet_name,
        leaning=leaning,
    )


def _topic_summary(topic: str) -> str:
    human_topic = topic.replace("_", " ")
    return (
        f"Articles about {human_topic} gathered from outlets with different ideological leanings. "
        "Choose the source that seems most useful for further reading."
    )


def main() -> None:
    args = parse_args()
    split_ids = _load_split_ids(args.split_file)
    allowed_topics = {topic.strip().lower() for topic in args.topics or []}

    grouped: dict[str, dict[str, list[Article]]] = defaultdict(lambda: defaultdict(list))
    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    for path in sorted(json_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        article_id = str(raw.get("ID") or raw.get("id") or "").strip()
        if split_ids is not None and article_id not in split_ids:
            continue

        topic = str(raw.get("topic", "")).strip().lower() or "unknown"
        if allowed_topics and topic not in allowed_topics:
            continue

        article = _build_article(raw)
        if article is None:
            continue

        grouped[topic][article.leaning].append(article)

    incidents: list[dict] = []
    for topic in sorted(grouped):
        by_leaning = grouped[topic]
        if any(len(by_leaning.get(leaning, [])) < args.min_per_leaning for leaning in ("left", "center", "right")):
            continue

        articles: list[Article] = []
        for leaning in ("left", "center", "right"):
            selected = by_leaning[leaning][: args.max_articles_per_leaning]
            articles.extend(selected)

        incident = PreparedIncident(
            incident_id=f"topic_{topic}",
            topic=topic,
            neutral_summary=_topic_summary(topic),
            articles=articles,
            metadata={
                "source": "real_topic_grouping",
                "json_dir": str(json_dir),
                "split_file": args.split_file,
                "article_counts": {
                    leaning: len(by_leaning.get(leaning, []))
                    for leaning in ("left", "center", "right")
                },
            },
        )
        incidents.append(incident.model_dump(mode="json"))

    if args.max_incidents is not None:
        incidents = incidents[: args.max_incidents]

    write_jsonl(args.output, incidents)

    print(
        json.dumps(
            {
                "output": args.output,
                "incident_count": len(incidents),
                "topics": [row["topic"] for row in incidents[:10]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
