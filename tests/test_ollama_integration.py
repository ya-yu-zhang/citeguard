"""Opt-in live test against a local Ollama. Skips when Ollama is not reachable.

Run: CITEGUARD_OLLAMA_TEST=1 pytest -m ollama
"""
import os

import httpx
import pytest

from citeguard.ask.llm import generate_answer, self_check_answer

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("CITEGUARD_OLLAMA_MODEL", "qwen2.5:3b-instruct")


def _ollama_up() -> bool:
    try:
        r = httpx.get(BASE.rstrip("/") + "/models", timeout=2.0)
        return r.status_code == 200 and any(m.get("id") == MODEL for m in r.json().get("data", []))
    except Exception:
        return False


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.environ.get("CITEGUARD_OLLAMA_TEST") != "1" or not _ollama_up(),
        reason="set CITEGUARD_OLLAMA_TEST=1 and run Ollama with the model pulled",
    ),
]


def test_live_answer_cites_valid_snippets(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    citations = [
        {"chunk_id": "c1", "title_path": "FastAPI > Path Parameters", "excerpt": ""},
        {"chunk_id": "c2", "title_path": "FastAPI > Query Parameters", "excerpt": ""},
    ]
    contexts = [
        "You can declare path parameters with the same syntax used by Python format strings: "
        "@app.get('/items/{item_id}'). The value of item_id is passed to your function as the argument item_id.",
        "When you declare other function parameters that are not part of the path parameters, "
        "they are automatically interpreted as query parameters.",
    ]
    ans = generate_answer("How do I declare a path parameter in FastAPI?", citations, contexts)
    ok, reason = self_check_answer(ans, citations)
    assert ok, (reason, ans)
