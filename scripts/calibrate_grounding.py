"""Calibrate the per-sentence support threshold on answerable vs refuse sets.

Step 1 (--generate): for every question that passes the retrieval gate, get ONE raw LLM
answer (same prompt/contexts as /ask) and cache it. Step 2: sweep thresholds offline over
the cached answers, applying the exact repair + self-check pipeline used by /ask.

Usage: python scripts/calibrate_grounding.py --generate   (needs server on :8000 + Ollama)
       python scripts/calibrate_grounding.py               (sweep only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citeguard.ask import attribution, llm  # noqa: E402
from citeguard.ask.attribution import is_out_of_scope_task, repair_citations  # noqa: E402
from citeguard.ask.evidence import evidence_gate, hits_to_citations  # noqa: E402
from citeguard.config import LLM_CONTEXT_CHARS  # noqa: E402

CACHE = ROOT / "docs/calibration_raw_week2.json"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def generate(base: str) -> None:
    rows = []
    sets = [("answerable", json.loads((ROOT / "docs/answerable_samples_week2.json").read_text())),
            ("refuse", json.loads((ROOT / "docs/refuse_samples_week2.json").read_text()))]
    with httpx.Client(timeout=300) as c:
        for kind, items in sets:
            for it in items:
                q = it["question"]
                hits = c.post(f"{base}/search", json={"query": q, "top_k": 5, "method": "hybrid"}).json()["results"]
                row = {"kind": kind, "id": it["id"], "question": q, "gate": None, "raw": None, "contexts": []}
                if is_out_of_scope_task(q):
                    row["gate"] = "out_of_scope_task"
                elif not (d := evidence_gate(hits)).ok:
                    row["gate"] = d.reason
                else:
                    ctx = [h["text"][:LLM_CONTEXT_CHARS] for h in hits]
                    row["contexts"] = ctx
                    row["raw"] = llm.generate_answer(q, hits_to_citations(hits), ctx)
                rows.append(row)
                print(kind, it["id"], row["gate"] or "LLM", flush=True)
    CACHE.write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def decide(row: dict, thr: float) -> bool:
    """True = answered (mirrors graph._check without regeneration)."""
    if row["gate"] or not row["raw"]:
        return False
    cites = [{"chunk_id": str(i)} for i in range(len(row["contexts"]))]
    ok0, r0 = llm.self_check_answer(row["raw"], cites)
    if not ok0 and r0 in {"model_declared_insufficient_evidence", "empty_answer", "no_citations"}:
        return False
    rep = repair_citations(row["raw"], row["question"], row["contexts"], min_support=thr)
    if rep.wrong_source or rep.supported == 0:
        return False
    old = llm.GROUNDING_MIN_SUPPORT
    llm.GROUNDING_MIN_SUPPORT = thr
    try:
        ok, _ = llm.self_check_answer(rep.answer, cites, question=row["question"], contexts=row["contexts"])
    finally:
        llm.GROUNDING_MIN_SUPPORT = old
    return ok


def sweep() -> list[dict]:
    rows = json.loads(CACHE.read_text())
    ans = [r for r in rows if r["kind"] == "answerable"]
    ref = [r for r in rows if r["kind"] == "refuse"]
    out = []
    for t in THRESHOLDS:
        a = sum(decide(r, t) for r in ans) / len(ans)
        fa = [r["id"] for r in ref if decide(r, t)]
        out.append({"threshold": t, "answer_rate": a, "refusal_accuracy": 1 - len(fa) / len(ref), "false_answers": fa})
    print("| threshold | answer rate | refusal accuracy | refuse ids answered |\n|---|---|---|---|")
    for o in out:
        print(f"| {o['threshold']:.1f} | {o['answer_rate']:.2f} | {o['refusal_accuracy']:.2f} | {o['false_answers'] or '—'} |")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--base", default="http://localhost:8000")
    a = ap.parse_args()
    if a.generate:
        generate(a.base)
    sweep()
