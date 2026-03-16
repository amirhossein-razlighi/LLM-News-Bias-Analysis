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
