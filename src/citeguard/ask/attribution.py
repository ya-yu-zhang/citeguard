"""Post-hoc citation repair + per-sentence verification for small LLMs.

Small local models often drop `[n]` markers, write a literal `[n]`, or cite the wrong
snippet. Instead of refusing the whole answer, each sentence is checked against the
retrieved chunks with a cheap lexical support score:

* sentence cites [k] and chunk k supports it        → keep as is
* cited chunk doesn't support it / no marker        → re-attribute to the best-supporting
  retrieved chunk if that one clears the threshold, else DROP the sentence
* no supported sentence left                         → caller refuses (no_supported_sentences)
* a sentence that CITES a snippet but is supported by no retrieved chunk counts as
  wrong_source; the caller refuses the whole answer (retrieval miss) rather than serve
  the model's own-knowledge claim

Support = share of the sentence's key terms (words of 4+ chars, minus stopwords and
words copied from the question) found in the chunk. Lexical proxy: it catches wrong-source
citations and invented vocabulary, NOT paraphrased-but-wrong meaning.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from citeguard.ask.llm import _CITE_RE, _tokens, normalize_citations

SENTENCE_MIN_SUPPORT = 0.5  # calibrated in docs/e2e_week2_ollama.md

_FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)
_TRAIL_MARK_RE = re.compile(r"([.!?])\s*((?:\[\d+\]\s*)+)")


@dataclass
class RepairResult:
    answer: str
    kept: int = 0
    reattributed: int = 0
    dropped: int = 0
    supported: int = 0  # kept sentences with non-trivial content
    wrong_source: int = 0  # cited a chunk, but NO retrieved chunk supports it
    log: list[dict] = field(default_factory=list)


def split_units(answer: str) -> list[str]:
    """Split into sentences; fenced code blocks are single units; trailing markers
    after a full stop ('foo. [1]') are attached to their sentence."""
    text = normalize_citations(answer).replace("[n]", " ").replace("[N]", " ")
    units: list[str] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        units.extend(_split_prose(text[pos : m.start()]))
        units.append(m.group(0).strip())
        pos = m.end()
    units.extend(_split_prose(text[pos:]))
    return [u for u in units if u.strip()]


def _split_prose(text: str) -> list[str]:
    text = _TRAIL_MARK_RE.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def support(sentence: str, chunk: str, question: str) -> float | None:
    """None when the sentence has no key terms of its own (pure restatement/filler)."""
    terms = _tokens(_CITE_RE.sub(" ", sentence)) - _tokens(question)
    if not terms:
        return None
    return len(terms & _tokens(chunk)) / len(terms)


def repair_citations(
    answer: str,
    question: str,
    contexts: Sequence[str],
    *,
    min_support: float = SENTENCE_MIN_SUPPORT,
) -> RepairResult:
    res = RepairResult(answer="")
    out: list[str] = []
    for unit in split_units(answer):
        marks = [int(m) for m in _CITE_RE.findall(unit)]
        body = _CITE_RE.sub("", unit).strip()
        body = re.sub(r"\s+([.!?,;:])", r"\1", body)
        if not body:
            continue
        scores = [support(body, c, question) for c in contexts]
        if all(s is None for s in scores):
            # filler with no own content: keep only if it carries a valid marker
            if marks and all(1 <= m <= len(contexts) for m in marks):
                out.append(_attach(body, sorted(set(marks))))
                res.kept += 1
            else:
                res.dropped += 1
                res.log.append({"unit": unit, "action": "drop_filler"})
            continue
        valid = [m for m in marks if 1 <= m <= len(contexts) and (scores[m - 1] or 0) >= min_support]
        if valid:
            out.append(_attach(body, sorted(set(valid))))
            res.kept += 1
            res.supported += 1
            continue
        best = max(range(len(contexts)), key=lambda i: scores[i] or 0.0)
        if (scores[best] or 0.0) >= min_support:
            out.append(_attach(body, [best + 1]))
            res.reattributed += 1
            res.supported += 1
            res.log.append({"unit": unit, "action": f"reattribute->{best + 1}", "support": scores[best]})
        else:
            res.dropped += 1
            if marks:
                res.wrong_source += 1
            res.log.append({"unit": unit, "action": "drop_unsupported", "cited": marks,
                            "best_support": scores[best]})
    res.answer = " ".join(out).strip()
    return res


def _attach(body: str, marks: list[int]) -> str:
    tag = "".join(f"[{m}]" for m in marks)
    if body.startswith("```"):
        return f"{body}\n{tag}"
    if body[-1] in ".!?":
        return f"{body[:-1]} {tag}{body[-1]}"
    return f"{body} {tag}"


_TASK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:write|compose|draft|generate|create|make|tell|sing|draw|invent)\b"
    r".{0,80}\b(?:poem|poems|song|lyrics|story|stories|essay|joke|jokes|haiku|limerick|"
    r"letter|novel|rap|sonnet|recipe|speech)\b",
    re.I,
)


def is_out_of_scope_task(question: str) -> bool:
    """Creative/generic tasks (not questions about the docs). Deliberately narrow:
    'Write a FastAPI test' is still allowed."""
    return bool(_TASK_RE.search(question))
