"""Measure /ask prompt length (tokens) for the same questions on two indexes.

Rebuilds exactly what /ask sends (top-5 hybrid hits → up to 1200 chars each →
_chat_messages) and counts tokens with the Qwen2.5 tokenizer (+ chat-template overhead
of ~3 tokens/message + 3). Only questions that pass the evidence gate (i.e. reach the
LLM) are counted. No LLM calls.

Usage: python scripts/measure_prompt_tokens.py <data_dir> [<data_dir> ...]
  data_dir contains processed/chunks.jsonl, indexes/bm25.pkl, indexes/qdrant_local
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citeguard.api.app import _load_chunks  # noqa: E402
from citeguard.ask.evidence import evidence_gate, hits_to_citations  # noqa: E402
from citeguard.ask.llm import _chat_messages  # noqa: E402
from citeguard.config import ASK_TOP_K, LLM_CONTEXT_CHARS  # noqa: E402
from citeguard.index.store import make_client  # noqa: E402
from citeguard.retrieval.bm25 import BM25Index  # noqa: E402
from citeguard.retrieval.hybrid import HybridRetriever  # noqa: E402

TOK = Tokenizer.from_file("/workspace/models/qwen2.5_tokenizer.json")


def questions() -> list[str]:
    qs = [q["question"] for q in json.loads((ROOT / "docs/answerable_samples_week2.json").read_text())]
    qs += [q["question"] for q in json.loads((ROOT / "docs/refuse_samples_week2.json").read_text())]
    return qs


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def measure(data_dir: Path) -> dict:
    chunks = _load_chunks(data_dir / "processed/chunks.jsonl")
    r = HybridRetriever(bm25=BM25Index.load(data_dir / "indexes/bm25.pkl"),
                        qdrant_client=make_client(url="", path=str(data_dir / "indexes/qdrant_local")),
                        chunk_by_id=chunks)
    toks, ctx_chars = [], []
    for q in questions():
        hits = [h.to_dict() for h in r.search(q, top_k=ASK_TOP_K, method="hybrid")]
        if not evidence_gate(hits).ok:
            continue
        ctx = [str(h.get("text") or "")[:LLM_CONTEXT_CHARS] for h in hits]
        msgs = _chat_messages(q, hits_to_citations(hits), ctx)
        n = sum(len(TOK.encode(m["content"]).ids) + 3 for m in msgs) + 3
        toks.append(n)
        ctx_chars.append(sum(len(c) for c in ctx))
    return {"data_dir": str(data_dir), "n_reaching_llm": len(toks), "p50_tokens": pct(toks, .5),
            "p95_tokens": pct(toks, .95), "mean_tokens": round(statistics.mean(toks), 1),
            "p50_context_chars": pct(ctx_chars, .5)}


if __name__ == "__main__":
    print(json.dumps([measure(Path(p)) for p in sys.argv[1:]], indent=1))
