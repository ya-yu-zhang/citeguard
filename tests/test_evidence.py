from citeguard.ask.evidence import evidence_gate, hits_to_citations
from citeguard.config import EVIDENCE_MIN_TOP3_CHARS, EVIDENCE_MIN_TOP_SCORE


def test_empty_hits_refuse():
    d = evidence_gate([])
    assert d.ok is False
    assert d.reason == "no_retrieval_hits"


def test_low_score_refuse():
    hits = [{"score": EVIDENCE_MIN_TOP_SCORE - 0.001, "text": "x" * 500, "chunk_id": "c1"}]
    d = evidence_gate(hits)
    assert d.ok is False
    assert d.reason and d.reason.startswith("top_score_below_threshold")


def test_short_top3_refuse():
    hits = [
        {"score": 0.5, "text": "a" * 50, "chunk_id": "c1"},
        {"score": 0.4, "text": "b" * 50, "chunk_id": "c2"},
        {"score": 0.3, "text": "c" * 50, "chunk_id": "c3"},
    ]
    assert sum(len(h["text"]) for h in hits) < EVIDENCE_MIN_TOP3_CHARS
    d = evidence_gate(hits)
    assert d.ok is False
    assert d.reason and d.reason.startswith("top3_chars_below_threshold")


def test_sufficient_evidence_ok():
    hits = [
        {"score": 0.05, "text": "e" * 100, "chunk_id": "c1", "title_path": "T", "source_url": "u"},
        {"score": 0.04, "text": "e" * 100, "chunk_id": "c2"},
        {"score": 0.03, "text": "e" * 100, "chunk_id": "c3"},
    ]
    d = evidence_gate(hits)
    assert d.ok is True
    assert d.reason is None
    cites = hits_to_citations(hits)
    assert cites[0]["chunk_id"] == "c1"
    assert len(cites[0]["excerpt"]) <= 240
