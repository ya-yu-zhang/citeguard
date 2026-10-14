"""Week-2 /ask tests — no network, stub LLM, mock retriever."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from citeguard.api import app as app_module
from citeguard.ask.evidence import evidence_gate
from citeguard.ask.graph import run_ask
from citeguard.ask.llm import self_check_answer, stub_generate
from citeguard.retrieval.bm25 import BM25Index
from citeguard.retrieval.hybrid import HybridRetriever, SearchHit


class _EmptyPoints:
    points = []


class FakeQdrant:
    def query_points(self, collection_name, query=None, limit=20, with_payload=True, **kwargs):
        return _EmptyPoints()

    def close(self):
        pass


def _hit(chunk_id: str, doc_id: str, text: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=score,
        title_path="Title",
        source_url="https://example.com/doc",
        method="hybrid",
    )


class ScriptedRetriever:
    """Returns pre-baked hits for any query (ignores vector/BM25)."""

    def __init__(self, hits: list[SearchHit]):
        self._hits = hits
        self.chunk_by_id = {h.chunk_id: h.to_dict() for h in hits}
        self.client = FakeQdrant()

    def search(self, query: str, top_k: int = 5, method: str = "hybrid") -> list[SearchHit]:
        return self._hits[:top_k]


FIXTURE = Path(__file__).parent / "fixtures" / "refuse_samples_week2.json"


def test_stub_generate_has_citations():
    cites = [
        {"chunk_id": "c1", "excerpt": "Use CORSMiddleware to enable CORS.", "title_path": "CORS", "source_url": "u", "score": 0.1},
        {"chunk_id": "c2", "excerpt": "allow_origins configures trusted origins.", "title_path": "CORS", "source_url": "u", "score": 0.09},
    ]
    ans = stub_generate("How do I enable CORS?", cites)
    ok, reason = self_check_answer(ans, cites)
    assert ok, reason


def test_self_check_rejects_missing_markers():
    cites = [{"chunk_id": "c1", "excerpt": "x" * 50}]
    ok, reason = self_check_answer("CORS is easy with no markers.", cites)
    assert ok is False
    assert reason == "missing_citation_markers"


def test_run_ask_refuses_empty_retrieval():
    r = ScriptedRetriever([])
    out = run_ask(r, "Anything?")  # type: ignore[arg-type]
    assert out["refused"] is True
    assert out["refusal_reason"] == "no_evidence:no_retrieval_hits"
    assert out["answer"] == ""
    assert out["method"] == "hybrid"


def test_run_ask_refuses_low_score():
    hits = [_hit("c1", "d1", "x" * 300, score=0.001)]
    r = ScriptedRetriever(hits)
    out = run_ask(r, "Anything?")  # type: ignore[arg-type]
    assert out["refused"] is True
    assert out["refusal_reason"] and "top_score_below_threshold" in out["refusal_reason"]


def test_run_ask_answers_with_stub_when_evidence_ok():
    text = "Use CORSMiddleware in FastAPI to configure Cross-Origin Resource Sharing. " * 3
    hits = [
        _hit("c1", "fastapi:cors.md", text, 0.05),
        _hit("c2", "fastapi:cors.md", text, 0.04),
        _hit("c3", "fastapi:cors.md", text, 0.03),
    ]
    r = ScriptedRetriever(hits)
    out = run_ask(r, "How do I enable CORS in FastAPI?")  # type: ignore[arg-type]
    assert out["refused"] is False
    assert out["answer"]
    assert "[1]" in out["answer"]
    assert len(out["citations"]) == 3
    assert out["method"] == "hybrid"


def test_refuse_samples_fixture_expect_refused_with_empty_hits():
    samples = json.loads(FIXTURE.read_text())
    assert len(samples) == 20
    r = ScriptedRetriever([])
    for s in samples:
        assert s["expected_refused"] is True
        out = run_ask(r, s["question"])  # type: ignore[arg-type]
        assert out["refused"] is True, s["question"]


def test_refuse_samples_low_evidence_gate():
    """Even with weak hits, evidence gate must refuse all 20 samples."""
    samples = json.loads(FIXTURE.read_text())
    weak = [_hit("w1", "noise.md", "zzz", score=0.005)]
    r = ScriptedRetriever(weak)
    for s in samples:
        out = run_ask(r, s["question"])  # type: ignore[arg-type]
        assert out["refused"] is True
        d = evidence_gate([h.to_dict() for h in weak])
        assert d.ok is False


@pytest.fixture()
def ask_client(monkeypatch):
    text = "CORSMiddleware enables CORS for FastAPI applications with allow_origins. " * 4
    hits = [
        _hit("c1", "fastapi:cors.md", text, 0.05),
        _hit("c2", "fastapi:cors.md", text, 0.04),
        _hit("c3", "fastapi:cors.md", text, 0.03),
    ]
    retriever = ScriptedRetriever(hits)

    def fake_build():
        return retriever

    monkeypatch.setattr(app_module, "build_retriever", fake_build)
    # also patch run path that uses _state retriever via lifespan
    app_module._state.clear()
    with TestClient(app_module.app) as c:
        yield c


def test_ask_endpoint_json(ask_client):
    r = ask_client.post("/ask", json={"question": "How do I enable CORS?"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) >= {"answer", "refused", "refusal_reason", "citations", "method"}
    assert body["refused"] is False
    assert body["method"] == "hybrid"
    assert body["citations"][0]["chunk_id"]
    assert "excerpt" in body["citations"][0]


def test_ask_stream_sse(ask_client):
    r = ask_client.post("/ask/stream", json={"question": "How do I enable CORS?"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    assert "event: final" in text
    assert "refused" in text


def test_ask_refuse_endpoint(monkeypatch):
    retriever = ScriptedRetriever([])

    def fake_build():
        return retriever

    monkeypatch.setattr(app_module, "build_retriever", fake_build)
    app_module._state.clear()
    with TestClient(app_module.app) as client:
        r = client.post("/ask", json={"question": "What is the capital of Mars?"})
        assert r.status_code == 200
        body = r.json()
        assert body["refused"] is True
        assert body["answer"] == ""


def test_normalize_multi_citations():
    from citeguard.ask.llm import normalize_citations

    assert normalize_citations("Foo [1, 2]. Bar [3].") == "Foo [1][2]. Bar [3]."
    ok, _ = self_check_answer("Foo [1, 2].", [{"chunk_id": "a"}, {"chunk_id": "b"}])
    assert ok


def test_self_check_grounding_rejects_hallucination():
    cites = [{"chunk_id": "a"}]
    ctx = ["Use model_dump() to convert a model instance into a Python dict."]
    ok, _ = self_check_answer(
        "Call model_dump() to convert the instance into a Python dict [1].",
        cites, question="How do I serialize a Pydantic model to a dict?", contexts=ctx,
    )
    assert ok
    ok, reason = self_check_answer(
        "Alice said hello in her private Slack message yesterday [1].",
        cites, question="What did Alice say on Slack?", contexts=ctx,
    )
    assert not ok and reason.startswith("low_grounding")


def test_self_check_rejects_not_in_evidence_phrasing():
    ok, reason = self_check_answer(
        "The password is not provided in the given evidence [1].", [{"chunk_id": "a"}]
    )
    assert not ok and reason == "model_declared_insufficient_evidence"


def test_unsupported_citation_reason_name(monkeypatch):
    from citeguard.ask import graph

    monkeypatch.setattr(graph, "llm_configured", lambda: True)
    hits = [{"text": "Use model_dump() to convert a model instance into a Python dictionary.", "chunk_id": "a"}]
    ans, ok, reason = graph._check(
        "Use the orjson serializer with custom encoders and streaming [1].",
        [{"chunk_id": "a"}], "How do I serialize?", hits,
    )
    assert not ok and reason == "unsupported_citation"
    assert graph.refusal_reason(reason) == "ungrounded_answer:unsupported_citation"
