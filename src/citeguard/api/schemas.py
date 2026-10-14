from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    method: str = Field(
        "hybrid",
        description="bm25 | vector | hybrid (default) | hybrid+rerank (opt-in)",
    )


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title_path: str
    source_url: str
    method: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    method: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    n_chunks: int
    qdrant_mode: str
    llm_configured: bool = False
    version: str = ""
    judge_mode: str = "off"


class Citation(BaseModel):
    chunk_id: str
    title_path: str
    source_url: str
    score: float
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    stream: bool = False
    # method omitted on purpose for /ask — uses hybrid (or ASK_USE_RERANK config)


class AskResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[Citation]
    method: str
    request_id: str | None = None
