"""API tests with a tiny in-memory fixture corpus (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citeguard.api import app as app_module
from citeguard.retrieval.bm25 import BM25Index
from citeguard.retrieval.hybrid import HybridRetriever


class _EmptyPoints:
    points = []


class FakeQdrant:
    """Minimal stand-in: vector search empty; hybrid still works via BM25 side."""

    def query_points(self, collection_name, query=None, limit=20, with_payload=True, **kwargs):
        return _EmptyPoints()

    def close(self):
        pass


FIXTURE_CHUNKS = [
    {
        "chunk_id": "c1",
        "doc_id": "fastapi:docs/en/docs/tutorial/cors.md",
        "text": "Use CORSMiddleware to enable Cross-Origin Resource Sharing in FastAPI.",
        "title_path": "CORS",
        "source_url": "https://example.com/cors.md",
        "source_repo": "fastapi/fastapi",
        "source_commit": "abc",
        "heading": "CORS",
        "token_count": 12,
    },
    {
        "chunk_id": "c2",
        "doc_id": "pydantic:docs/concepts/models.md",
        "text": "Pydantic BaseModel validates data and provides type hints for fields.",
        "title_path": "Models",
        "source_url": "https://example.com/models.md",
        "source_repo": "pydantic/pydantic",
        "source_commit": "abc",
        "heading": "Models",
        "token_count": 12,
    },
    {
        "chunk_id": "c3",
        "doc_id": "langgraph:src/oss/langgraph/streaming.mdx",
        "text": "LangGraph supports streaming tokens and state updates from a graph.",
        "title_path": "Streaming",
        "source_url": "https://example.com/streaming.mdx",
        "source_repo": "langchain-ai/docs",
        "source_commit": "abc",
        "heading": "Streaming",
        "token_count": 12,
    },
]


@pytest.fixture()
def client(monkeypatch):
    chunk_by_id = {c["chunk_id"]: c for c in FIXTURE_CHUNKS}
    bm25 = BM25Index.build([c["chunk_id"] for c in FIXTURE_CHUNKS], [c["text"] for c in FIXTURE_CHUNKS])
    retriever = HybridRetriever(bm25=bm25, qdrant_client=FakeQdrant(), chunk_by_id=chunk_by_id)

    def fake_build():
        return retriever

    monkeypatch.setattr(app_module, "build_retriever", fake_build)
    # reset lifespan state
    app_module._state.clear()
    with TestClient(app_module.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_chunks"] == 3
    assert "llm_configured" in body


def test_search_bm25(client):
    r = client.post("/search", json={"query": "CORSMiddleware FastAPI", "top_k": 2, "method": "bm25"})
    assert r.status_code == 200
    data = r.json()
    assert data["results"]
    assert data["results"][0]["doc_id"].endswith("cors.md")
    assert "source_url" in data["results"][0]
    assert "title_path" in data["results"][0]


def _ask_lines(caplog):
    out = []
    for rec in caplog.records:
        if rec.name == "citeguard.request":
            out.append(json.loads(rec.getMessage()))
    return out


def test_ask_exposes_request_id_and_logs(client, caplog, monkeypatch):
    monkeypatch.setenv("ASK_RATE_PER_MIN", "0")
    caplog.set_level("INFO", logger="citeguard.request")
    r = client.post("/ask", json={"question": "How do I enable CORS in FastAPI with CORSMiddleware?"})
    assert r.status_code == 200
    body = r.json()
    rid = body["request_id"]
    assert rid and r.headers["X-Request-ID"] == rid
    lines = [l for l in _ask_lines(caplog) if l["request_id"] == rid]
    assert len(lines) == 1
    line = lines[0]
    assert line["path"] == "/ask"
    assert line["outcome"] in ("answered", "refused")
    assert line["outcome"] == ("refused" if body["refused"] else "answered")
    assert line["n_citations"] == len(body["citations"])
    assert line["latency_ms"] >= 0
    assert {"refusal_reason", "model", "judge_mode"} <= line.keys()


def test_ask_honours_incoming_request_id(client):
    r = client.post("/ask", json={"question": "Pydantic BaseModel validation"}, headers={"X-Request-ID": "demo-123"})
    assert r.json()["request_id"] == "demo-123"
    assert r.headers["X-Request-ID"] == "demo-123"


def test_stream_events_carry_request_id(client, caplog):
    caplog.set_level("INFO", logger="citeguard.request")
    r = client.post("/ask/stream", json={"question": "LangGraph streaming tokens"})
    assert r.status_code == 200
    rid = r.headers["X-Request-ID"]
    datas = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    assert datas and all(d.get("request_id") == rid for d in datas)
    assert any(l["path"] == "/ask/stream" and l["request_id"] == rid for l in _ask_lines(caplog))


def test_health_has_version_and_judge_mode(client):
    body = client.get("/health").json()
    assert body["version"] and body["judge_mode"] in ("off", "veto")
