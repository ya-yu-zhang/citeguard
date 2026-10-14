"""Scan raw docs → chunks.jsonl + manifest.json."""
from __future__ import annotations

import re

import json
from pathlib import Path

from citeguard.config import (
    CHUNKS_PATH,
    MANIFEST_PATH,
    PROCESSED_DIR,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)
from citeguard.ingest.chunk import chunk_markdown
from citeguard.ingest.sources import SOURCES


def _iter_doc_files(root: Path, path_in_repo: str) -> list[Path]:
    base = root / path_in_repo
    if not base.exists():
        # sparse checkout may already be rooted at path_in_repo content under local_subdir
        # but our clones keep full relative path
        raise FileNotFoundError(f"Missing docs path: {base}")
    files: list[Path] = []
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".md", ".mdx"}:
            files.append(p)
    return files


_INCLUDE_RE = re.compile(r"\{\*\s*(\S+?)\s*(?:[a-z]+\[[^\]]*\]\s*)*\*\}")
INCLUDE_MAX_CHARS = 6000


def resolve_includes(text: str, include_base: Path) -> tuple[str, int, int]:
    """Inline FastAPI `{* ../../docs_src/x.py hl[..] *}` placeholders with the referenced
    file as a fenced code block (hl/ln options are display-only and ignored).
    Paths are relative to `docs/en` (mkdocs root). Returns (text, resolved, missing)."""
    resolved = missing = 0

    def _sub(m: re.Match) -> str:
        nonlocal resolved, missing
        rel = m.group(1)
        fp = (include_base / rel).resolve()
        if not fp.is_file():
            missing += 1
            return m.group(0)
        resolved += 1
        code = fp.read_text(encoding="utf-8", errors="replace").rstrip()
        if len(code) > INCLUDE_MAX_CHARS:
            code = code[:INCLUDE_MAX_CHARS] + "\n# ... (truncated)"
        lang = "python" if fp.suffix == ".py" else fp.suffix.lstrip(".")
        name = rel.split("docs_src/", 1)[-1]
        return f"```{lang}\n# docs_src/{name}\n{code}\n```"

    return _INCLUDE_RE.sub(_sub, text), resolved, missing


def build_chunks(
    *,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> tuple[list[dict], dict]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    manifest_docs: list[dict] = []
    include_stats = {"resolved": 0, "missing": 0}

    for src in SOURCES:
        root = src.local_root()
        files = _iter_doc_files(root, src.path_in_repo)
        for fp in files:
            rel = fp.relative_to(root).as_posix()
            doc_id = f"{src.name}:{rel}"
            source_url = src.blob_base_url.format(commit=src.commit, relpath=rel)
            text = fp.read_text(encoding="utf-8", errors="replace")
            if src.name == "fastapi":
                text, n_ok, n_miss = resolve_includes(text, root / src.path_in_repo)
                include_stats["resolved"] += n_ok
                include_stats["missing"] += n_miss
            chunks = chunk_markdown(
                text,
                doc_id=doc_id,
                source_url=source_url,
                source_repo=src.repo,
                source_commit=src.commit,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
            for c in chunks:
                all_chunks.append(c.to_dict())
            manifest_docs.append(
                {
                    "doc_id": doc_id,
                    "relpath": rel,
                    "source_url": source_url,
                    "source_repo": src.repo,
                    "source_commit": src.commit,
                    "license": src.license,
                    "n_chunks": len(chunks),
                    "bytes": fp.stat().st_size,
                }
            )

    manifest = {
        "sources": [
            {
                "name": s.name,
                "repo": s.repo,
                "path_in_repo": s.path_in_repo,
                "commit": s.commit,
                "license": s.license,
                "note": (
                    "Maintained LangGraph docs live in langchain-ai/docs; "
                    "langchain-ai/langgraph/docs is not used."
                    if s.name == "langgraph"
                    else ""
                ),
            }
            for s in SOURCES
        ],
        "fastapi_includes": {**include_stats, "note": "docs_src fetched at the same pinned commit"},
        "n_docs": len(manifest_docs),
        "n_chunks": len(all_chunks),
        "chunk_target_tokens": target_tokens,
        "chunk_overlap_tokens": overlap_tokens,
        "docs": manifest_docs,
    }
    return all_chunks, manifest


def write_artifacts(chunks: list[dict], manifest: dict) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_ingest() -> dict:
    chunks, manifest = build_chunks()
    write_artifacts(chunks, manifest)
    return {"n_docs": manifest["n_docs"], "n_chunks": manifest["n_chunks"], "chunks_path": str(CHUNKS_PATH)}
