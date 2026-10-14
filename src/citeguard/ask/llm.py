"""Optional OpenAI-compatible LLM; deterministic stub when LLM_API_KEY is unset."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import httpx

from citeguard.config import get_settings

_CITE_RE = re.compile(r"\[(\d+)\]")


def llm_configured() -> bool:
    return bool((get_settings().llm_api_key or "").strip())


def _format_context(citations: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        blocks.append(
            f"[{i}] chunk_id={c['chunk_id']} title={c.get('title_path','')}\n"
            f"{c.get('excerpt') or ''}"
        )
    return "\n\n".join(blocks)


def stub_generate(question: str, citations: list[dict]) -> str:
    """Deterministic extractive answer — only quotes provided citations with [n] markers."""
    if not citations:
        return ""
    parts = [
        "Based on the retrieved documentation (stub generator; set LLM_API_KEY for a live model):"
    ]
    for i, c in enumerate(citations[:3], start=1):
        excerpt = (c.get("excerpt") or "").replace("\n", " ").strip()
        if len(excerpt) > 180:
            excerpt = excerpt[:177] + "..."
        parts.append(f"- {excerpt} [{i}]")
    parts.append(f"(Question restated: {question.strip()})")
    return "\n".join(parts)


def _chat_messages(question: str, citations: list[dict]) -> list[dict[str, str]]:
    system = (
        "You answer technical questions ONLY using the numbered evidence snippets. "
        "Every factual sentence must end with one or more citation markers like [1] or [2]. "
        "If the evidence is insufficient, reply exactly: INSUFFICIENT_EVIDENCE. "
        "Do not invent APIs, filenames, or behavior not present in the snippets."
    )
    user = f"Question:\n{question}\n\nEvidence:\n{_format_context(citations)}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_answer(question: str, citations: list[dict]) -> str:
    if not citations:
        return ""
    if not llm_configured():
        return stub_generate(question, citations)
    settings = get_settings()
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": _chat_messages(question, citations),
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def stream_generate(question: str, citations: list[dict]) -> Iterator[str]:
    """Yield answer text chunks. Stub yields once; live LLM uses SSE if available."""
    if not citations:
        return
        yield  # pragma: no cover
    if not llm_configured():
        yield stub_generate(question, citations)
        return
    settings = get_settings()
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": _chat_messages(question, citations),
        "temperature": 0.0,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta


def self_check_answer(answer: str, citations: list[dict]) -> tuple[bool, str | None]:
    """Verify answer cites only valid indices and is not an insufficiency flag."""
    if not answer or not answer.strip():
        return False, "empty_answer"
    if "INSUFFICIENT_EVIDENCE" in answer:
        return False, "model_declared_insufficient_evidence"
    n = len(citations)
    if n == 0:
        return False, "no_citations"
    markers = [int(m) for m in _CITE_RE.findall(answer)]
    if not markers:
        return False, "missing_citation_markers"
    bad = [m for m in markers if m < 1 or m > n]
    if bad:
        return False, f"citation_out_of_range:{bad}"
    return True, None
