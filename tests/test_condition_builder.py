from app.experiment.condition_builder import build_condition_bundles
from app.schemas.models import ConditionName, PreparedIncident


def _incident() -> PreparedIncident:
    return PreparedIncident.model_validate(
        {
            "incident_id": "inc_t1",
            "neutral_summary": "summary",
            "articles": [
                {"article_id": "l1", "headline": "L", "outlet_name": "LO", "leaning": "left"},
                {"article_id": "c1", "headline": "C", "outlet_name": "CO", "leaning": "center"},
                {"article_id": "r1", "headline": "R", "outlet_name": "RO", "leaning": "right"},
            ],
        }
    )


def test_builds_all_condition_bundles() -> None:
    bundles = build_condition_bundles(
        _incident(),
        conditions=[c for c in ConditionName],
        max_combinations=2,
        seed=123,
    )
    assert set(bundles.keys()) == set(ConditionName)
    assert len(bundles[ConditionName.HEADLINES_ONLY]) == 1


def test_manipulated_sources_rotates_outlets() -> None:
    bundles = build_condition_bundles(
        _incident(),
        conditions=[ConditionName.HEADLINES_WITH_MANIPULATED_SOURCES],
    )
    presented = bundles[ConditionName.HEADLINES_WITH_MANIPULATED_SOURCES][0]
    assert [p.outlet_name for p in presented] == ["CO", "RO", "LO"]


def test_shuffle_candidates_is_reproducible_for_seed() -> None:
    bundles_a = build_condition_bundles(
        _incident(),
        conditions=[ConditionName.HEADLINES_ONLY],
        seed=11,
        shuffle_candidates=True,
    )
    bundles_b = build_condition_bundles(
        _incident(),
        conditions=[ConditionName.HEADLINES_ONLY],
        seed=11,
        shuffle_candidates=True,
    )

    order_a = [p.article_id for p in bundles_a[ConditionName.HEADLINES_ONLY][0]]
    order_b = [p.article_id for p in bundles_b[ConditionName.HEADLINES_ONLY][0]]
    assert order_a == order_b


def test_shuffle_candidates_preserves_member_set() -> None:
    bundles = build_condition_bundles(
        _incident(),
        conditions=[ConditionName.HEADLINES_WITH_SOURCES],
        seed=17,
        shuffle_candidates=True,
    )
    presented = bundles[ConditionName.HEADLINES_WITH_SOURCES][0]
    assert sorted(p.article_id for p in presented) == ["c1", "l1", "r1"]
