"""FastAPI app: /health and /search."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from citeguard.api.schemas import HealthResponse, SearchRequest, SearchResponse, SearchResult
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


app = FastAPI(title="CiteGuard", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = _state.get("settings") or get_settings()
    mode = "remote" if settings.qdrant_url else "embedded"
    n = len(getattr(_state.get("retriever"), "chunk_by_id", {}) or {})
    status = "ok" if _state.get("ready") else f"degraded: {_state.get('error')}"
    return HealthResponse(status=status, n_chunks=n, qdrant_mode=mode)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    retriever: HybridRetriever | None = _state.get("retriever")
    if retriever is None:
        raise HTTPException(503, detail=f"Retriever not ready: {_state.get('error')}")
    method = req.method
    if method not in {"bm25", "vector", "hybrid", "hybrid+rerank"}:
        raise HTTPException(400, detail="method must be bm25|vector|hybrid|hybrid+rerank")
    hits = retriever.search(req.query, top_k=req.top_k, method=method)
    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        method=method,
        results=[SearchResult(**h.to_dict()) for h in hits],
    )


def create_app() -> FastAPI:
    return app
