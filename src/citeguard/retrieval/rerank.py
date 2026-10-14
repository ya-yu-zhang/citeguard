"""Cross-encoder reranker via fastembed TextCrossEncoder (bge-reranker-base)."""
from __future__ import annotations

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from citeguard.config import RERANK_MODEL


@lru_cache(maxsize=1)
def get_reranker(model_name: str = RERANK_MODEL) -> TextCrossEncoder:
    return TextCrossEncoder(model_name=model_name)


def rerank(
    query: str,
    passages: list[tuple[str, str]],
    *,
    top_k: int = 5,
    model_name: str = RERANK_MODEL,
) -> list[tuple[str, float]]:
    """passages: list of (id, text). Returns (id, score) sorted desc."""
    if not passages:
        return []
    model = get_reranker(model_name)
    texts = [t for _, t in passages]
    scores = list(model.rerank(query, texts))
    ranked = sorted(
        ((passages[i][0], float(scores[i])) for i in range(len(passages))),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]
