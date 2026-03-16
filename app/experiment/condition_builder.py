from __future__ import annotations

import itertools
import random
from collections import defaultdict

from app.schemas.models import ConditionName, PreparedIncident, PresentedArticle


def _group_by_leaning(incident: PreparedIncident) -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for article in incident.articles:
        grouped[article.leaning].append(article)
    return grouped


def _sample_triplets(
    grouped: dict[str, list],
    max_combinations: int,
    seed: int,
) -> list[tuple]:
    left = grouped.get("left", [])
    center = grouped.get("center", [])
    right = grouped.get("right", [])

    if not left or not center or not right:
        return []

    all_triplets = list(itertools.product(left, center, right))
    if len(all_triplets) <= max_combinations:
        return all_triplets

    rng = random.Random(seed)
    rng.shuffle(all_triplets)
    return all_triplets[:max_combinations]


def _presented_articles_for_condition(
    triplet: tuple,
    condition: ConditionName,
) -> list[PresentedArticle]:
    left_article, center_article, right_article = triplet
    selected = [left_article, center_article, right_article]

    if condition == ConditionName.HEADLINES_ONLY:
        return [
            PresentedArticle(article_id=a.article_id, headline=a.headline)
            for a in selected
        ]

    if condition == ConditionName.HEADLINES_WITH_SOURCES:
        return [
            PresentedArticle(
                article_id=a.article_id,
                headline=a.headline,
                outlet_name=a.outlet_name,
            )
            for a in selected
        ]

    if condition == ConditionName.SOURCES_ONLY:
        return [
            PresentedArticle(article_id=a.article_id, outlet_name=a.outlet_name)
            for a in selected
        ]

    if condition == ConditionName.HEADLINES_WITH_MANIPULATED_SOURCES:
        rotated_outlets = [center_article.outlet_name, right_article.outlet_name, left_article.outlet_name]
        return [
            PresentedArticle(
                article_id=a.article_id,
                headline=a.headline,
                outlet_name=rotated_outlets[idx],
            )
            for idx, a in enumerate(selected)
        ]

    raise ValueError(f"Unsupported condition: {condition}")


def build_condition_bundles(
    incident: PreparedIncident,
    conditions: list[ConditionName],
    max_combinations: int = 3,
    seed: int = 42,
) -> dict[ConditionName, list[list[PresentedArticle]]]:
    grouped = _group_by_leaning(incident)
    triplets = _sample_triplets(grouped, max_combinations=max_combinations, seed=seed)

    result: dict[ConditionName, list[list[PresentedArticle]]] = {}
    for condition in conditions:
        result[condition] = [
            _presented_articles_for_condition(triplet, condition)
            for triplet in triplets
        ]
    return result
