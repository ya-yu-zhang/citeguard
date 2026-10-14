"""Hybrid retrieval: BM25 + vector → RRF → optional rerank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from citeguard.config import DEFAULT_TOP_K, RETRIEVE_CANDIDATES, RRF_K
from citeguard.index.store import search_vectors
from citeguard.retrieval.bm25 import BM25Index
from citeguard.retrieval.embedder import embed_query
from citeguard.retrieval.rerank import rerank
from citeguard.retrieval.rrf import reciprocal_rank_fusion


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title_path: str
    source_url: str
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "title_path": self.title_path,
            "source_url": self.source_url,
            "method": self.method,
        }


class HybridRetriever:
    def __init__(
        self,
        *,
        bm25: BM25Index,
        qdrant_client,
        chunk_by_id: dict[str, dict],
    ):
        self.bm25 = bm25
        self.client = qdrant_client
        self.chunk_by_id = chunk_by_id

    def _payload_hit(self, chunk_id: str, score: float, method: str) -> SearchHit | None:
        c = self.chunk_by_id.get(chunk_id)
        if not c:
            return None
        return SearchHit(
            chunk_id=chunk_id,
            doc_id=c["doc_id"],
            text=c["text"],
            score=score,
            title_path=c.get("title_path", ""),
            source_url=c.get("source_url", ""),
            method=method,
        )

    def search_bm25(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        hits = []
        for cid, score in self.bm25.search(query, top_k=top_k):
            h = self._payload_hit(cid, score, "bm25")
            if h:
                hits.append(h)
        return hits

    def search_vector(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchHit]:
        qv = embed_query(query)
        hits = []
        for cid, score, _ in search_vectors(self.client, qv, top_k=top_k):
            h = self._payload_hit(cid, score, "vector")
            if h:
                hits.append(h)
        return hits

    def search_hybrid(self, query: str, top_k: int = DEFAULT_TOP_K, *, candidates: int = RETRIEVE_CANDIDATES) -> list[SearchHit]:
        bm25_ids = [cid for cid, _ in self.bm25.search(query, top_k=candidates)]
        qv = embed_query(query)
        vec_ids = [cid for cid, _, _ in search_vectors(self.client, qv, top_k=candidates)]
        fused = reciprocal_rank_fusion([bm25_ids, vec_ids], k=RRF_K)
        hits = []
        for cid, score in fused[:top_k]:
            h = self._payload_hit(cid, score, "hybrid")
            if h:
                hits.append(h)
        return hits

    def search_hybrid_rerank(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        candidates: int = RETRIEVE_CANDIDATES,
    ) -> list[SearchHit]:
        # RRF over larger pool, then cross-encoder rerank
        pool = self.search_hybrid(query, top_k=candidates, candidates=candidates)
        passages = [(h.chunk_id, h.text) for h in pool]
        ranked = rerank(query, passages, top_k=top_k)
        hits = []
        for cid, score in ranked:
            h = self._payload_hit(cid, score, "hybrid+rerank")
            if h:
                hits.append(h)
        return hits

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, method: str = "hybrid+rerank") -> list[SearchHit]:
        if method == "bm25":
            return self.search_bm25(query, top_k)
        if method == "vector":
            return self.search_vector(query, top_k)
        if method == "hybrid":
            return self.search_hybrid(query, top_k)
        return self.search_hybrid_rerank(query, top_k)
