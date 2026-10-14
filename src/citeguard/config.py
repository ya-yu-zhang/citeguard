"""Runtime configuration. QDRANT_URL unset → local embedded Qdrant (HF Spaces friendly)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
MANIFEST_PATH = DATA_DIR / "manifest.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
BM25_PATH = INDEX_DIR / "bm25.pkl"
QDRANT_PATH = INDEX_DIR / "qdrant_local"
COLLECTION_NAME = "citeguard_chunks"

# Retrieval defaults
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Lighter CPU cross-encoder (0.08 GB). bge-reranker-base (~1 GB) missed p95<1.5s (~2.7s).
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
DEFAULT_TOP_K = 5
RETRIEVE_CANDIDATES = 20  # BM25/vector pool before RRF
RERANK_CANDIDATES = 10  # only top-N after RRF go to cross-encoder
RERANK_MAX_CHARS = 512  # truncate passage text for faster rerank
DEFAULT_SEARCH_METHOD = "hybrid"  # hybrid+rerank is opt-in


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str = ""
    qdrant_url: str | None = None  # None / empty → embedded local path
    port: int = 8000
    embed_model: str = EMBED_MODEL
    rerank_model: str = RERANK_MODEL
    collection_name: str = COLLECTION_NAME
    qdrant_path: str = str(QDRANT_PATH)


def get_settings() -> Settings:
    return Settings()
