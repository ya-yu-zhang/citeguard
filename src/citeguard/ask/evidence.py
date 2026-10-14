"""Evidence gate for cite-or-abstain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from citeguard.config import EVIDENCE_MIN_TOP3_CHARS, EVIDENCE_MIN_TOP_SCORE


@dataclass(frozen=True)
class EvidenceDecision:
    ok: bool
    reason: str | None
    top_score: float
    top3_chars: int


def evidence_gate(
    hits: Sequence[dict],
    *,
    min_top_score: float = EVIDENCE_MIN_TOP_SCORE,
    min_top3_chars: int = EVIDENCE_MIN_TOP3_CHARS,
) -> EvidenceDecision:
    """Refuse when retrieval is empty, top score too low, or top-3 text too short.

    Hits are dicts with at least ``score`` and ``text`` (SearchHit.to_dict shape).
    """
    if not hits:
        return EvidenceDecision(
            ok=False,
            reason="no_retrieval_hits",
            top_score=0.0,
            top3_chars=0,
        )
    top_score = float(hits[0].get("score") or 0.0)
    top3_chars = sum(len(str(h.get("text") or "")) for h in hits[:3])
    if top_score < min_top_score:
        return EvidenceDecision(
            ok=False,
            reason=f"top_score_below_threshold:{top_score:.4f}<{min_top_score}",
            top_score=top_score,
            top3_chars=top3_chars,
        )
    if top3_chars < min_top3_chars:
        return EvidenceDecision(
            ok=False,
            reason=f"top3_chars_below_threshold:{top3_chars}<{min_top3_chars}",
            top_score=top_score,
            top3_chars=top3_chars,
        )
    return EvidenceDecision(ok=True, reason=None, top_score=top_score, top3_chars=top3_chars)


def hits_to_citations(hits: Sequence[dict], *, excerpt_chars: int = 240) -> list[dict]:
    out = []
    for h in hits:
        text = str(h.get("text") or "")
        out.append(
            {
                "chunk_id": h.get("chunk_id", ""),
                "title_path": h.get("title_path", ""),
                "source_url": h.get("source_url", ""),
                "score": float(h.get("score") or 0.0),
                "excerpt": text[:excerpt_chars],
            }
        )
    return out
