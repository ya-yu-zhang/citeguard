"""OpenRouter payload defaults (no network)."""
from types import SimpleNamespace

from citeguard.ask import llm


def _settings(base, extra=""):
    return SimpleNamespace(llm_base_url=base, llm_model="m", llm_max_tokens=300, llm_extra_body=extra, llm_api_key="")


def test_openrouter_defaults_reasoning_off(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings("https://openrouter.ai/api/v1"))
    _, payload, _ = llm._request("q", [], None, stream=False)
    assert payload["reasoning"] == {"enabled": False}


def test_extra_body_overrides_reasoning(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings("https://openrouter.ai/api/v1", '{"reasoning":{"enabled":true}}'))
    _, payload, _ = llm._request("q", [], None, stream=False)
    assert payload["reasoning"] == {"enabled": True}


def test_ollama_payload_untouched(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings("http://localhost:11434/v1"))
    _, payload, _ = llm._request("q", [], None, stream=False)
    assert "reasoning" not in payload
