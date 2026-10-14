"""Provider-agnostic OpenAI-compatible LLM client (Ollama / OpenRouter / ...).

LLM_MODEL unset → deterministic stub generator (no network). LLM_API_KEY is optional
(Ollama ignores it; OpenRouter requires it).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterator, Sequence

import httpx

from citeguard.config import LLM_CONTEXT_CHARS, get_settings

_CITE_RE = re.compile(r"\[(\d+)\]")
# small models sometimes emit [1, 2] or [1][2]; normalize "[1, 2]" → "[1][2]"
_MULTI_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)+)\]")


def llm_configured() -> bool:
    return bool((get_settings().llm_model or "").strip())


def _format_context(citations: list[dict], contexts: Sequence[str] | None = None) -> str:
    blocks = []
    for i, c in enumerate(citations, start=1):
        body = contexts[i - 1] if contexts and i - 1 < len(contexts) else (c.get("excerpt") or "")
        blocks.append(f"[{i}] {c.get('title_path','')}\n{body[:LLM_CONTEXT_CHARS]}")
    return "\n\n".join(blocks)


def stub_generate(question: str, citations: list[dict]) -> str:
    """Deterministic extractive answer — only quotes provided citations with [n] markers."""
    if not citations:
        return ""
    parts = [
        "Based on the retrieved documentation (stub generator; set LLM_MODEL for a live model):"
    ]
    for i, c in enumerate(citations[:3], start=1):
        excerpt = (c.get("excerpt") or "").replace("\n", " ").strip()
        if len(excerpt) > 180:
            excerpt = excerpt[:177] + "..."
        parts.append(f"- {excerpt} [{i}]")
    parts.append(f"(Question restated: {question.strip()})")
    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are a documentation assistant. Answer ONLY from the numbered evidence snippets.\n"
    "Rules:\n"
    "1. Answer in 1-4 short sentences in your own words; do not copy whole snippets.\n"
    "2. Put a citation like [1] or [2] at the END of EVERY sentence, naming the snippet "
    "that supports it.\n"
    "3. If the snippets do not answer the question, reply with exactly: INSUFFICIENT_EVIDENCE\n"
    "4. Never invent APIs, parameters, versions, people, or facts not in the snippets.\n\n"
    "Example\n"
    "Evidence:\n[1] Tutorial > Query Params\nFunction parameters that are not path parameters "
    "are interpreted as query parameters.\n[2] Tutorial > Body\nTo declare a request body, "
    "use Pydantic models.\n"
    "Question: How do I declare a request body?\n"
    "Answer: Declare a parameter whose type is a Pydantic model; FastAPI reads it from the "
    "request body [2].\n\n"
    "Example\n"
    "Evidence:\n[1] Tutorial > Body\nTo declare a request body, use Pydantic models.\n"
    "Question: What is the capital of France?\n"
    "Answer: INSUFFICIENT_EVIDENCE"
)


def _chat_messages(
    question: str, citations: list[dict], contexts: Sequence[str] | None = None
) -> list[dict[str, str]]:
    user = (
        f"Evidence:\n{_format_context(citations, contexts)}\n\n"
        f"Question: {question}\n\n"
        "Answer (end every sentence with [n]) or INSUFFICIENT_EVIDENCE:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _request(question, citations, contexts, stream: bool) -> tuple[str, dict, dict]:
    s = get_settings()
    url = s.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": s.llm_model,
        "messages": _chat_messages(question, citations, contexts),
        "temperature": 0.0,
        "max_tokens": s.llm_max_tokens,
        "stream": stream,
    }
    if "openrouter.ai" in s.llm_base_url:
        # Reasoning models (e.g. qwen3.7-flash) otherwise spend max_tokens on reasoning and
        # return empty content -> every answer refused as empty_answer. LLM_EXTRA_BODY can override.
        payload["reasoning"] = {"enabled": False}
    if (s.llm_extra_body or "").strip():
        payload.update(json.loads(s.llm_extra_body))
    headers = {"Content-Type": "application/json"}
    if (s.llm_api_key or "").strip():
        headers["Authorization"] = f"Bearer {s.llm_api_key}"
    return url, payload, headers


def normalize_citations(answer: str) -> str:
    def _sub(m: re.Match) -> str:
        return "".join(f"[{n.strip()}]" for n in m.group(1).split(","))

    return _MULTI_CITE_RE.sub(_sub, answer)


def _log_usage(model: str, usage: dict, latency_s: float) -> None:
    """Append token usage to $LLM_USAGE_LOG (JSONL) when set. Never logs keys or text."""
    path = os.environ.get("LLM_USAGE_LOG")
    if not path:
        return
    rec = {
        "ts": time.time(),
        "model": model,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cost": usage.get("cost"),
        "latency_s": round(latency_s, 3),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def generate_answer(
    question: str, citations: list[dict], contexts: Sequence[str] | None = None
) -> str:
    if not citations:
        return ""
    if not llm_configured():
        return stub_generate(question, citations)
    url, payload, headers = _request(question, citations, contexts, stream=False)
    t0 = time.perf_counter()
    with httpx.Client(timeout=get_settings().llm_timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    _log_usage(payload["model"], data.get("usage") or {}, time.perf_counter() - t0)
    return normalize_citations((data["choices"][0]["message"]["content"] or "").strip())


def stream_generate(
    question: str, citations: list[dict], contexts: Sequence[str] | None = None
) -> Iterator[str]:
    """Yield answer text chunks. Stub yields once; live LLM uses SSE."""
    if not citations:
        return
        yield  # pragma: no cover
    if not llm_configured():
        yield stub_generate(question, citations)
        return
    url, payload, headers = _request(question, citations, contexts, stream=True)
    with httpx.Client(timeout=get_settings().llm_timeout_s) as client:
        with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or [{}]
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta


_WORD_RE = re.compile(r"[a-z0-9_]{4,}")
_STOP = frozenset(
    "this that with from have will your which their there these those what when where "
    "about into also then than them they been were does used using use uses only just "
    "more some such other each like make makes made allows allow can could should would "
    "must need needs function functions method methods value values".split()
)
_NONANSWER_RE = re.compile(
    r"(not (?:provided|mentioned|included|available|specified|found) in the (?:given |provided )?"
    r"(?:evidence|snippets?|documentation|context))|(?:evidence|snippets?) (?:does|do) not",
    re.I,
)
GROUNDING_MIN_SUPPORT = 0.5


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOP}


def grounding_support(
    answer: str, question: str, contexts: Sequence[str]
) -> float:
    """Share of the answer's content words (minus words copied from the question) that
    appear in the evidence. Cheap lexical proxy for hallucination, not a semantic check."""
    ans = _tokens(_CITE_RE.sub(" ", answer)) - _tokens(question)
    if not ans:
        return 0.0
    ctx = set().union(*(_tokens(c) for c in contexts)) if contexts else set()
    return len(ans & ctx) / len(ans)


def self_check_answer(
    answer: str,
    citations: list[dict],
    *,
    question: str | None = None,
    contexts: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    """Verify citation markers are present/in range; optionally a lexical grounding check
    against the cited snippets (enabled when question+contexts are passed)."""
    if not answer or not answer.strip():
        return False, "empty_answer"
    if "INSUFFICIENT_EVIDENCE" in answer or _NONANSWER_RE.search(answer):
        return False, "model_declared_insufficient_evidence"
    n = len(citations)
    if n == 0:
        return False, "no_citations"
    markers = [int(m) for m in _CITE_RE.findall(normalize_citations(answer))]
    if not markers:
        return False, "missing_citation_markers"
    bad = [m for m in markers if m < 1 or m > n]
    if bad:
        return False, f"citation_out_of_range:{bad}"
    if question is not None and contexts:
        cited = [contexts[m - 1] for m in sorted(set(markers)) if m - 1 < len(contexts)]
        sup = grounding_support(answer, question, cited)
        if sup < GROUNDING_MIN_SUPPORT:
            return False, f"low_grounding:{sup:.2f}"
    return True, None
