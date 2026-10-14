#!/usr/bin/env python3
"""Chunk raw docs → build BM25 + Qdrant indexes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citeguard.config import BM25_PATH, CHUNKS_PATH, INDEX_DIR, get_settings
from citeguard.index.store import ensure_collection, make_client, upsert_chunks
from citeguard.ingest.pipeline import run_ingest
from citeguard.retrieval.bm25 import BM25Index
from citeguard.retrieval.embedder import embed_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-embed", action="store_true", help="Only chunk + BM25")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print("==> chunking…")
    stats = run_ingest()
    print(f"    docs={stats['n_docs']} chunks={stats['n_chunks']}")

    # load chunks
    import json

    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    print("==> BM25…")
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    bm25 = BM25Index.build(ids, texts)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(BM25_PATH)
    print(f"    saved {BM25_PATH}")

    if args.skip_embed:
        print("skipped embeddings")
        return

    print("==> embeddings + Qdrant…")
    settings = get_settings()
    client = make_client()
    # probe dim
    sample = embed_texts([texts[0][:200] if texts else "hello"], batch_size=1)
    dim = len(sample[0])
    ensure_collection(client, dim)
    print(f"    dim={dim} mode={'remote' if settings.qdrant_url else 'embedded'}")

    vectors = embed_texts(texts, batch_size=args.batch_size)
    payloads = [
        {
            "doc_id": c["doc_id"],
            "title_path": c["title_path"],
            "source_url": c["source_url"],
            "text": c["text"][:2000],
        }
        for c in chunks
    ]
    upsert_chunks(client, vectors=vectors, payloads=payloads, ids=ids)
    print(f"    upserted {len(ids)} vectors")
    client.close()
    print("done")


if __name__ == "__main__":
    main()
