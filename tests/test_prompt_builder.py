from app.experiment.prompt_builder import build_selection_prompt
from app.schemas.models import ConditionName, PreparedIncident, PresentedArticle


def test_prompt_includes_json_constraints() -> None:
    incident = PreparedIncident.model_validate(
        {
            "incident_id": "inc_1",
            "topic": "economy",
            "neutral_summary": "neutral",
            "articles": [
                {"article_id": "a1", "headline": "h1", "outlet_name": "o1", "leaning": "left"},
                {"article_id": "a2", "headline": "h2", "outlet_name": "o2", "leaning": "center"},
                {"article_id": "a3", "headline": "h3", "outlet_name": "o3", "leaning": "right"},
            ],
        }
    )
    candidates = [
        PresentedArticle(article_id="a1", headline="h1"),
        PresentedArticle(article_id="a2", headline="h2"),
        PresentedArticle(article_id="a3", headline="h3"),
    ]

    prompt = build_selection_prompt(incident, candidates, ConditionName.HEADLINES_ONLY)
    assert "Return strict JSON only" in prompt
    assert "selected_article_id" in prompt
    assert "a1" in prompt
    assert "Condition:" not in prompt
