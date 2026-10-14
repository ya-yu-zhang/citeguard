"""Qdrant vector store — remote URL or local embedded path when QDRANT_URL unset."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from citeguard.config import COLLECTION_NAME, get_settings


def make_client(url: str | None = None, path: str | None = None) -> QdrantClient:
    settings = get_settings()
    resolved_url = url if url is not None else settings.qdrant_url
    resolved_path = path if path is not None else settings.qdrant_path
    if resolved_url:
        return QdrantClient(url=resolved_url, prefer_grpc=False)
    Path(resolved_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(resolved_path))


def ensure_collection(client: QdrantClient, dim: int, name: str = COLLECTION_NAME) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def upsert_chunks(
    client: QdrantClient,
    *,
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
    ids: list[str],
    name: str = COLLECTION_NAME,
    batch_size: int = 64,
) -> None:
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]
        points = [
            qm.PointStruct(id=_to_uuid(cid), vector=vec, payload={**pay, "chunk_id": cid})
            for cid, vec, pay in zip(batch_ids, batch_vecs, batch_payloads)
        ]
        client.upsert(collection_name=name, points=points)


def search_vectors(
    client: QdrantClient,
    query_vector: list[float],
    *,
    top_k: int = 20,
    name: str = COLLECTION_NAME,
) -> list[tuple[str, float, dict]]:
    # qdrant-client >=1.12: search() → query_points()
    response = client.query_points(
        collection_name=name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    out: list[tuple[str, float, dict]] = []
    for h in response.points:
        payload = h.payload or {}
        cid = str(payload.get("chunk_id", h.id))
        out.append((cid, float(h.score), payload))
    return out


def _to_uuid(chunk_id: str) -> str:
    h = hashlib.sha1(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))
