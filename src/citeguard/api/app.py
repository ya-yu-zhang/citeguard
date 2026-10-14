"""FastAPI app: /health, /search, /ask (cite-or-abstain)."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from citeguard.api.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from citeguard.ask.graph import run_ask, stream_ask_events
from citeguard.ask.llm import llm_configured
from citeguard.config import BM25_PATH, CHUNKS_PATH, get_settings
from citeguard.index.store import make_client
from citeguard.retrieval.bm25 import BM25Index
from citeguard.retrieval.hybrid import HybridRetriever

_state: dict[str, Any] = {}


def _load_chunks(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.exists():
        return by_id
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            by_id[c["chunk_id"]] = c
    return by_id


def build_retriever() -> HybridRetriever:
    chunks = _load_chunks(CHUNKS_PATH)
    if not chunks:
        raise RuntimeError(f"No chunks at {CHUNKS_PATH}; run scripts/ingest_docs.py first")
    if BM25_PATH.exists():
        bm25 = BM25Index.load(BM25_PATH)
    else:
        ids = list(chunks.keys())
        texts = [chunks[i]["text"] for i in ids]
        bm25 = BM25Index.build(ids, texts)
    client = make_client()
    return HybridRetriever(bm25=bm25, qdrant_client=client, chunk_by_id=chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        _state["retriever"] = build_retriever()
        _state["ready"] = True
        _state["error"] = None
    except Exception as e:  # noqa: BLE001 — surface at /health
        _state["retriever"] = None
        _state["ready"] = False
        _state["error"] = str(e)
    _state["settings"] = settings
    yield
    client = getattr(_state.get("retriever"), "client", None)
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


app = FastAPI(title="CiteGuard", version="0.2.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = _state.get("settings") or get_settings()
    mode = "remote" if settings.qdrant_url else "embedded"
    n = len(getattr(_state.get("retriever"), "chunk_by_id", {}) or {})
    status = "ok" if _state.get("ready") else f"degraded: {_state.get('error')}"
    return HealthResponse(
        status=status,
        n_chunks=n,
        qdrant_mode=mode,
        llm_configured=llm_configured(),
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(503, detail=f"Retriever not ready: {_state.get('error')}")
    method = req.method
    allowed = {"bm25", "vector", "hybrid", "hybrid+rerank", "hybrid_rerank"}
    if method not in allowed:
        raise HTTPException(400, detail="method must be bm25|vector|hybrid|hybrid+rerank")
    if method == "hybrid_rerank":
        method = "hybrid+rerank"
    hits = retriever.search(req.query, top_k=req.top_k, method=method)
    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        method=method,
        results=[SearchResult(**h.to_dict()) for h in hits],
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Cite-or-abstain. Use stream=true for SSE (or POST /ask/stream)."""
    if req.stream:
        raise HTTPException(
            400,
            detail="For streaming use POST /ask/stream (text/event-stream), or set Accept accordingly",
        )
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(503, detail=f"Retriever not ready: {_state.get('error')}")
    result = run_ask(retriever, req.question)
    return AskResponse(
        answer=result["answer"],
        refused=result["refused"],
        refusal_reason=result["refusal_reason"],
        citations=[Citation(**c) for c in result["citations"]],
        method=result["method"],
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(503, detail=f"Retriever not ready: {_state.get('error')}")

    def event_gen():
        for ev in stream_ask_events(retriever, req.question):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def create_app() -> FastAPI:
    return app
