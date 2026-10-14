"""Local CPU embeddings via fastembed (BAAI/bge-small-en-v1.5)."""
from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from citeguard.config import EMBED_MODEL


@lru_cache(maxsize=1)
def get_embedder(model_name: str = EMBED_MODEL) -> TextEmbedding:
    return TextEmbedding(model_name=model_name)


def embed_texts(texts: list[str], *, model_name: str = EMBED_MODEL, batch_size: int = 32) -> list[list[float]]:
    model = get_embedder(model_name)
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for vec in model.embed(batch, batch_size=batch_size):
            out.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
    return out


def embed_query(query: str, *, model_name: str = EMBED_MODEL) -> list[float]:
    return embed_texts([query], model_name=model_name)[0]
