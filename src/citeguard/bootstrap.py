"""Container startup check: make sure chunks + indexes exist before uvicorn starts.

If data/processed/chunks.jsonl or data/indexes/ are missing:
  * CITEGUARD_INDEX_DATASET set (HF dataset repo id) → download them with huggingface_hub
    (HF_TOKEN only needed for private datasets);
  * otherwise exit(2) with instructions to run scripts/ingest_docs.py or mount data/.
"""
from __future__ import annotations

import os
import sys

from citeguard.config import BM25_PATH, CHUNKS_PATH, DATA_DIR, QDRANT_PATH

MSG = (
    "CiteGuard: indexes not found under {d} (need processed/chunks.jsonl, indexes/bm25.pkl, "
    "indexes/qdrant_local).\n"
    "Fix one of:\n"
    "  * mount prebuilt data:  docker run -v $PWD/data:/app/data ...\n"
    "  * build them in the container:  python scripts/ingest_docs.py\n"
    "  * set CITEGUARD_INDEX_DATASET=<hf user>/<dataset> to download a prebuilt snapshot"
)


def missing() -> list[str]:
    need = [CHUNKS_PATH, BM25_PATH]
    out = [str(p) for p in need if not p.exists()]
    if not os.environ.get("QDRANT_URL") and not QDRANT_PATH.exists():
        out.append(str(QDRANT_PATH))
    return out


def ensure_indexes() -> int:
    if not missing():
        print("CiteGuard: indexes present", flush=True)
        return 0
    repo = os.environ.get("CITEGUARD_INDEX_DATASET", "").strip()
    if repo:
        from huggingface_hub import snapshot_download
        print(f"CiteGuard: downloading indexes from HF dataset {repo}", flush=True)
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(DATA_DIR),
                          allow_patterns=["processed/*", "indexes/*", "indexes/**", "manifest.json"],
                          token=os.environ.get("HF_TOKEN") or None)
        if not missing():
            return 0
        print(f"CiteGuard: dataset {repo} did not provide: {missing()}", file=sys.stderr)
    print(MSG.format(d=DATA_DIR), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(ensure_indexes())
