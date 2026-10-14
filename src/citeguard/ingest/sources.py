"""Pinned doc sources. LangGraph maintained docs are in langchain-ai/docs (not langgraph/docs)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from citeguard.config import RAW_DIR


@dataclass(frozen=True)
class DocSource:
    name: str
    repo: str  # owner/name
    path_in_repo: str
    local_subdir: str
    commit: str
    license: str
    blob_base_url: str  # for per-file source URLs

    def local_root(self) -> Path:
        return RAW_DIR / self.local_subdir


# Pinned to commits actually checked out during Week 1 ingest (2026-10-14).
SOURCES: list[DocSource] = [
    DocSource(
        name="fastapi",
        repo="fastapi/fastapi",
        path_in_repo="docs/en",
        local_subdir="fastapi",
        commit="f5c6e9b4f9cadc1bf0f9a1fd672e621206b63007",
        license="MIT",
        blob_base_url="https://github.com/fastapi/fastapi/blob/{commit}/{relpath}",
    ),
    DocSource(
        name="pydantic",
        repo="pydantic/pydantic",
        path_in_repo="docs",
        local_subdir="pydantic",
        commit="22c4317a5d51d71344275259b4e536c4a90fe53b",
        license="MIT",
        blob_base_url="https://github.com/pydantic/pydantic/blob/{commit}/{relpath}",
    ),
    DocSource(
        name="langgraph",
        repo="langchain-ai/docs",
        path_in_repo="src/oss/langgraph",
        local_subdir="langgraph",
        commit="a28c75411093fdc0e57b03c1d6486ba78f005f51",
        license="MIT",
        blob_base_url="https://github.com/langchain-ai/docs/blob/{commit}/{relpath}",
    ),
]


def sources_as_dicts() -> list[dict]:
    return [asdict(s) for s in SOURCES]
