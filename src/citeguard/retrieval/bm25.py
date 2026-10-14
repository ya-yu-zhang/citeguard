"""BM25 lexical index over chunks."""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunk_ids: list[str]
    texts: list[str]

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.chunk_ids[i], float(s)) for i, s in ranked if s > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "texts": self.texts}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls.build(data["chunk_ids"], data["texts"])

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> "BM25Index":
        corpus = [tokenize(t) for t in texts]
        return cls(bm25=BM25Okapi(corpus), chunk_ids=chunk_ids, texts=texts)
