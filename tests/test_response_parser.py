from app.parsing.response_parser import parse_model_response
from app.schemas.models import ParseStatus


def test_parse_valid_json() -> None:
    text = '{"selected_article_id":"a_center_1","reason":"best"}'
    parsed = parse_model_response(text, {"a_left_1", "a_center_1", "a_right_1"})
    assert parsed.status == ParseStatus.SUCCESS
    assert parsed.selected_article_id == "a_center_1"


def test_parse_json_with_prefixed_article_id() -> None:
    text = '{"selected_article_id":"article_id=a_left_1","reason":"best"}'
    parsed = parse_model_response(text, {"a_left_1", "a_center_1", "a_right_1"})
    assert parsed.status == ParseStatus.SUCCESS
    assert parsed.selected_article_id == "a_left_1"


def test_parse_json_with_prefixed_quoted_article_id() -> None:
    text = '{"selected_article_id":"article_id: \"a_right_1\"","reason":"best"}'
    parsed = parse_model_response(text, {"a_left_1", "a_center_1", "a_right_1"})
    assert parsed.status == ParseStatus.SUCCESS
    assert parsed.selected_article_id == "a_right_1"


def test_fallback_parse_text_with_id() -> None:
    text = "I choose a_right_1 because it provides detailed context"
    parsed = parse_model_response(text, {"a_left_1", "a_center_1", "a_right_1"})
    assert parsed.status == ParseStatus.FALLBACK
    assert parsed.selected_article_id == "a_right_1"


def test_failed_parse_when_no_signal() -> None:
    text = "no idea"
    parsed = parse_model_response(text, {"a_left_1", "a_center_1", "a_right_1"})
    assert parsed.status == ParseStatus.FAILED
    assert parsed.selected_article_id is None
