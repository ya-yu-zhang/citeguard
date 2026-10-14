"""Markdown-heading-aware chunking (~500 tokens, 50 overlap)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterator

import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_ANCHOR_RE = re.compile(r"\s*\{#[^}]+\}\s*")

def _clean_heading(title: str) -> str:
    return _ANCHOR_RE.sub("", title).strip()



def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    ids = _ENC.encode(text)
    if not ids:
        return []
    if len(ids) <= max_tokens:
        return [text]
    chunks: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        piece = ids[start : start + max_tokens]
        if not piece:
            break
        chunks.append(_ENC.decode(piece))
        if start + max_tokens >= len(ids):
            break
    return chunks


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title_path: str
    source_url: str
    source_repo: str
    source_commit: str
    heading: str
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_frontmatter(md: str) -> str:
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            return md[end + 4 :].lstrip("\n")
    return md


def iter_sections(md: str) -> Iterator[tuple[str, str]]:
    """Yield (title_path, section_body) using ATX headings."""
    md = _strip_frontmatter(md)
    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        yield ("", md.strip())
        return

    # preamble before first heading
    if matches[0].start() > 0:
        preamble = md[: matches[0].start()].strip()
        if preamble:
            yield ("", preamble)

    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = _clean_heading(m.group(2).strip())
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        title_path = " > ".join(t for _, t in stack)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[start:end].strip()
        if body:
            yield (title_path, body)


def chunk_markdown(
    md: str,
    *,
    doc_id: str,
    source_url: str,
    source_repo: str,
    source_commit: str,
    target_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for title_path, body in iter_sections(md):
        pieces = _split_tokens(body, target_tokens, overlap_tokens)
        for i, piece in enumerate(pieces):
            text = piece.strip()
            if not text:
                continue
            raw_id = f"{doc_id}::{title_path}::{i}::{text[:64]}"
            chunk_id = hashlib.sha1(raw_id.encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    title_path=title_path or "(root)",
                    source_url=source_url,
                    source_repo=source_repo,
                    source_commit=source_commit,
                    heading=title_path.split(" > ")[-1] if title_path else "(root)",
                    token_count=count_tokens(text),
                )
            )
    return chunks
