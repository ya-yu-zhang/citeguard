"""Sentence-level semantic faithfulness judge (LLM-as-judge). NOT wired into /ask yet.

For each sentence of an answer, the judge sees the question, the sentence, and the full
text of the chunk(s) the sentence cites, and returns a JSON verdict:
  supported   — the cited text entails the sentence (paraphrase OK; no new facts)
  unsupported — the sentence adds facts the cited text doesn't state, contradicts it, or
                presents cited facts as answering the question when they don't
plus one answer-level flag, `answers_question`.

Configured separately from the generator (any OpenAI-compatible endpoint):
  JUDGE_BASE_URL (default https://openrouter.ai/api/v1), JUDGE_MODEL
  (default deepseek/deepseek-v4-flash), JUDGE_API_KEY (falls back to OPENROUTER_API_KEY).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import httpx

from citeguard.ask.attribution import split_units
from citeguard.ask.llm import _CITE_RE

JUDGE_SYSTEM = (
    "You are a strict fact-checker for a documentation Q&A system. You receive a user "
    "question and an answer split into numbered sentences; each sentence comes with the "
    "exact documentation text it cites.\n"
    "For EACH sentence decide:\n"
    "- supported: the cited text states or directly implies everything the sentence claims "
    "(paraphrase is fine).\n"
    "- unsupported: the sentence adds a fact, API, number or recommendation the cited text "
    "does not state; contradicts it; or presents the cited facts as the way to do what the "
    "question asks when the cited text is about something else.\n"
    "Do NOT use your own knowledge to fill gaps: a true fact missing from the cited text is "
    "unsupported. Purely connective phrasing (e.g. 'Additionally,') is fine.\n"
    "Also decide answers_question: does the answer, as a whole, address what was asked?\n"
    'Reply with JSON only: {"verdicts":[{"i":1,"supported":true,"reason":"<=15 words"}],'
    '"answers_question":true}'
)


# v2 (one revision, informed by the 3B Week 2 Q1 miss, so the gate set is NOT held-out):
# consistency with the cited text is not enough; the cited text must answer the asked question.
JUDGE_SYSTEM_V2 = JUDGE_SYSTEM.replace(
    "Also decide answers_question: does the answer, as a whole, address what was asked?\n",
    "Also decide answers_question: does the cited text, as used, actually answer the specific "
    "question asked? Being consistent with the cited text is NOT enough. If the answer "
    "describes a related but different feature, a special case, or a different mechanism "
    "than the one asked about, answers_question is false, and the sentences that present it "
    "as the answer are unsupported.\n",
)
assert JUDGE_SYSTEM_V2 != JUDGE_SYSTEM


@dataclass
class JudgeResult:
    verdicts: list[dict]
    answers_question: bool | None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    error: str | None = None

    @property
    def flagged(self) -> bool:
        """True = answer should be refused (any unsupported sentence, or off-question)."""
        if self.error:
            return True  # fail closed
        return any(not v.get("supported") for v in self.verdicts) or self.answers_question is False


def build_items(answer: str, chunk_texts: Sequence[str]) -> list[dict]:
    """Sentences with their cited chunk texts. Uncited sentences get no evidence."""
    items = []
    for unit in split_units(answer):
        marks = sorted({int(m) for m in _CITE_RE.findall(unit)})
        body = _CITE_RE.sub("", unit).strip()
        if len(body) < 3:
            continue
        ev = [chunk_texts[m - 1] for m in marks if 1 <= m <= len(chunk_texts)]
        items.append({"sentence": body, "cites": marks, "evidence": ev})
    return items


def _user_prompt(question: str, items: list[dict], max_chars: int) -> str:
    parts = [f"Question: {question}\n"]
    for i, it in enumerate(items, start=1):
        ev = "\n---\n".join(e[:max_chars] for e in it["evidence"]) or "(no citation)"
        parts.append(f"Sentence {i}: {it['sentence']}\nCited text for sentence {i}:\n{ev}\n")
    return "\n".join(parts)


def judge_answer(
    question: str,
    answer: str,
    chunk_texts: Sequence[str],
    *,
    max_chunk_chars: int = 2000,
    max_tokens: int = 800,
    retries: int = 1,
    timeout_s: float = 60.0,
) -> JudgeResult:
    """Retries once (by default) on bad_json / verdict_count_mismatch; usage is summed."""
    res = _judge_once(question, answer, chunk_texts, max_chunk_chars, max_tokens, timeout_s)
    tries = 0
    while res.error in ("bad_json", "verdict_count_mismatch") and tries < retries:
        tries += 1
        prev = res
        res = _judge_once(question, answer, chunk_texts, max_chunk_chars, max_tokens, timeout_s)
        res.usage = {"cost": (prev.usage.get("cost") or 0) + (res.usage.get("cost") or 0),
                     "retried_after": prev.error}
        res.latency_s += prev.latency_s
    return res


def judge_mode() -> str:
    """JUDGE_MODE=off (default) | veto: after the overlap rule passes, the judge may veto."""
    m = os.environ.get("JUDGE_MODE", "off").strip().lower()
    return m if m in ("off", "veto") else "off"


def _judge_once(question, answer, chunk_texts, max_chunk_chars, max_tokens, timeout_s) -> JudgeResult:
    items = build_items(answer, chunk_texts)
    if not items:
        return JudgeResult([], None, error="no_sentences")
    base = os.environ.get("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    model = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v4-flash")
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "usage": {"include": True},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_V2 if os.environ.get("JUDGE_PROMPT") == "v2" else JUDGE_SYSTEM},
            {"role": "user", "content": _user_prompt(question, items, max_chunk_chars)},
        ],
    }
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{base}/chat/completions", json=payload, timeout=timeout_s,
                       headers={"Authorization": f"Bearer {key}"} if key else {})
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # noqa: BLE001
        return JudgeResult([], None, error=f"request_failed:{type(e).__name__}",
                           latency_s=time.perf_counter() - t0)
    lat = time.perf_counter() - t0
    usage = data.get("usage") or {}
    text = (data["choices"][0]["message"].get("content") or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        obj = json.loads(m.group(0) if m else text)
    except json.JSONDecodeError:
        return JudgeResult([], None, usage=usage, latency_s=lat, error="bad_json")
    verdicts = obj.get("verdicts") or []
    for v, it in zip(verdicts, items):
        v["sentence"] = it["sentence"]
        v["cites"] = it["cites"]
    if len(verdicts) != len(items):
        return JudgeResult(verdicts, obj.get("answers_question"), usage, lat, error="verdict_count_mismatch")
    return JudgeResult(verdicts, obj.get("answers_question"), usage, lat)
