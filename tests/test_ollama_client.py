import requests

from app.models.ollama_client import OllamaClient


class _Response:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("http error")

    def json(self):
        return self._payload

    @property
    def text(self):
        if isinstance(self._payload, dict):
            return str(self._payload)
        return str(self._payload)


def test_generate_calls_ollama(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert json["model"] == "llama3"
        assert json["format"]["type"] == "object"
        return _Response({"response": '{"selected_article_id":"a1","reason":"ok"}'})

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaClient()
    generation = client.generate(
        model="llama3",
        prompt="hello",
        response_schema={
            "type": "object",
            "properties": {"selected_article_id": {"type": "string"}, "reason": {"type": "string"}},
        },
    )
    assert "selected_article_id" in generation.text


def test_list_models(monkeypatch):
    def fake_get(url, timeout):
        assert url.endswith("/api/tags")
        return _Response({"models": [{"name": "llama3"}, {"name": "gemma3"}]})

    monkeypatch.setattr(requests, "get", fake_get)

    client = OllamaClient()
    models = client.list_models()
    assert models == ["llama3", "gemma3"]


def test_generate_404_model_not_found_has_helpful_hint(monkeypatch):
    def fake_post(url, json, timeout):
        return _Response({"error": "model 'qwen2.5:7b' not found"}, status=404)

    def fake_get(url, timeout):
        return _Response({"models": [{"name": "llama3"}]})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    client = OllamaClient()
    try:
        client.generate(model="qwen2.5:7b", prompt="hi", retries=0)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "ollama pull qwen2.5:7b" in str(exc)


def test_generate_falls_back_to_api_chat(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(url)
        if url.endswith("/api/generate"):
            return _Response({"error": "404 page not found"}, status=404)
        if url.endswith("/api/chat"):
            return _Response({"message": {"content": "hello from chat"}}, status=200)
        return _Response({"error": "should not hit"}, status=500)

    monkeypatch.setattr(requests, "post", fake_post)

    client = OllamaClient()
    generation = client.generate(model="llama3", prompt="hello", retries=0)
    assert generation.text == "hello from chat"
    assert any(c.endswith("/api/generate") for c in calls)
    assert any(c.endswith("/api/chat") for c in calls)
