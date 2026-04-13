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

    total_triplets = len(left) * len(center) * len(right)
    if total_triplets <= max_combinations:
        return list(itertools.product(left, center, right))

    rng = random.Random(seed)
    sampled_keys: set[tuple[int, int, int]] = set()
    sampled_triplets: list[tuple] = []

    while len(sampled_triplets) < max_combinations:
        triplet_key = (
            rng.randrange(len(left)),
            rng.randrange(len(center)),
            rng.randrange(len(right)),
        )
        if triplet_key in sampled_keys:
            continue
        sampled_keys.add(triplet_key)
        sampled_triplets.append(
            (
                left[triplet_key[0]],
                center[triplet_key[1]],
                right[triplet_key[2]],
            )
        )

    return sampled_triplets


def _presented_articles_for_condition(
    triplet: tuple,
    condition: ConditionName,
) -> list[PresentedArticle]:
    left_article, center_article, right_article = triplet
    selected = [left_article, center_article, right_article]

    if condition == ConditionName.HEADLINES_ONLY:
        return [
            PresentedArticle(article_id=a.article_id, headline=a.headline, leaning=a.leaning)
            for a in selected
        ]

    if condition == ConditionName.HEADLINES_WITH_SOURCES:
        return [
            PresentedArticle(
                article_id=a.article_id,
                headline=a.headline,
                outlet_name=a.outlet_name,
                leaning=a.leaning,
            )
            for a in selected
        ]

    if condition == ConditionName.SOURCES_ONLY:
        return [
            PresentedArticle(article_id=a.article_id, outlet_name=a.outlet_name, leaning=a.leaning)
            for a in selected
        ]

    if condition == ConditionName.HEADLINES_WITH_MANIPULATED_SOURCES:
        rotated_outlets = [center_article.outlet_name, right_article.outlet_name, left_article.outlet_name]
        return [
            PresentedArticle(
                article_id=a.article_id,
                headline=a.headline,
                outlet_name=rotated_outlets[idx],
                leaning=a.leaning,
            )
            for idx, a in enumerate(selected)
        ]

    raise ValueError(f"Unsupported condition: {condition}")


def build_condition_bundles(
    incident: PreparedIncident,
    conditions: list[ConditionName],
    max_combinations: int = 3,
    seed: int = 42,
    shuffle_candidates: bool = False,
) -> dict[ConditionName, list[list[PresentedArticle]]]:
    grouped = _group_by_leaning(incident)
    triplets = _sample_triplets(grouped, max_combinations=max_combinations, seed=seed)

    result: dict[ConditionName, list[list[PresentedArticle]]] = {}
    for condition_idx, condition in enumerate(conditions):
        condition_bundles: list[list[PresentedArticle]] = []
        for triplet_idx, triplet in enumerate(triplets):
            presented = _presented_articles_for_condition(triplet, condition)
            if shuffle_candidates and len(presented) > 1:
                # Make candidate order reproducible for a given seed/condition/triplet.
                order_rng = random.Random(seed + (condition_idx * 1009) + (triplet_idx * 9176))
                order_rng.shuffle(presented)
            condition_bundles.append(presented)
        result[condition] = condition_bundles
    return result
