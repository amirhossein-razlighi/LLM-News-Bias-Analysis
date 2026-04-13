from pathlib import Path

from app.utils.io import append_jsonl, read_jsonl


def test_append_jsonl_appends_multiple_rows(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"

    append_jsonl(target, {"id": 1})
    append_jsonl(target, {"id": 2})

    assert read_jsonl(target) == [{"id": 1}, {"id": 2}]
