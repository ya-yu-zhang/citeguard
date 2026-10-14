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


def test_search_bm25(client):
    r = client.post("/search", json={"query": "CORSMiddleware FastAPI", "top_k": 2, "method": "bm25"})
    assert r.status_code == 200
    data = r.json()
    assert data["results"]
    assert data["results"][0]["doc_id"].endswith("cors.md")
    assert "source_url" in data["results"][0]
    assert "title_path" in data["results"][0]
