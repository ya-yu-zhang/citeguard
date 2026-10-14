"""Live end-to-end run of /ask and /ask/stream against a running server.

Usage: python scripts/e2e_ask.py [--base http://localhost:8000] [--out docs/e2e_week2_ollama.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r"\[(\d+)\]")


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def retrieved_chunks(c: httpx.Client, base: str, q: str) -> dict[str, str]:
    r = c.post(f"{base}/search", json={"query": q, "top_k": 5, "method": "hybrid"})
    r.raise_for_status()
    return {h["chunk_id"]: h.get("text") or "" for h in r.json()["results"]}


SENT_RE = re.compile(r"[^.!?\n]*?(?:\[\d+\]\s*)+[.!?]?|[^.!?\n]+[.!?]")
TERM_MIN = 0.5  # share of a sentence's key terms that must appear in its cited chunk(s)


def sentence_support(res: dict, chunks: dict[str, str], question: str) -> dict:
    """Programmatic citation validity, per cited sentence:
    (a) every [n] maps to a returned citation, (b) that citation's chunk_id was retrieved,
    (c) the cited chunk text contains >= TERM_MIN of the sentence's key terms
    (content words >=4 chars, stopwords and question words removed)."""
    from citeguard.ask.llm import _tokens

    cits = res.get("citations") or []
    qtok = _tokens(question)
    text = re.sub(r"```.*?(```|$)", " ", res.get("answer") or "", flags=re.S)  # ignore code
    text = text.replace("[n]", " ")
    # "sentence. [1]" → "sentence [1]." so a trailing marker belongs to its sentence
    text = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", r" \2\1 ", text)
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if len(_tokens(x)) >= 2]
    n_cited = n_ok = n_uncited = 0
    fails = []
    for snt in sents:
        ms = [int(m) for m in CITE.findall(snt)]
        if not ms:
            n_uncited += 1
            continue
        n_cited += 1
        if any(not (1 <= m <= len(cits)) for m in ms):
            fails.append({"sentence": snt, "why": "marker_out_of_range"})
            continue
        ids = [cits[m - 1]["chunk_id"] for m in ms]
        if any(i not in chunks for i in ids):
            fails.append({"sentence": snt, "why": "chunk_not_retrieved"})
            continue
        terms = _tokens(CITE.sub(" ", snt)) - qtok
        ctx = set().union(*(_tokens(chunks[i]) for i in ids))
        share = len(terms & ctx) / len(terms) if terms else 1.0
        if share >= TERM_MIN:
            n_ok += 1
        else:
            fails.append({"sentence": snt, "why": f"key_terms_{share:.2f}"})
    return {"cited_sentences": n_cited, "valid_sentences": n_ok,
            "uncited_sentences": n_uncited, "failures": fails}


def ask(c, base, q):
    t = time.perf_counter()
    r = c.post(f"{base}/ask", json={"question": q})
    r.raise_for_status()
    return r.json(), time.perf_counter() - t


def ask_stream(c, base, q):
    t = time.perf_counter()
    first_tok = None
    final = None
    n_tok = 0
    with c.stream("POST", f"{base}/ask/stream", json={"question": q}) as r:
        r.raise_for_status()
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if ev == "token":
                    n_tok += 1
                    if first_tok is None:
                        first_tok = time.perf_counter() - t
                elif ev == "final":
                    final = data
    return final, time.perf_counter() - t, first_tok, n_tok


def check_citations(res: dict, retrieved: set[str]) -> tuple[bool, list[int], list[str]]:
    cits = res.get("citations") or []
    markers = sorted({int(m) for m in CITE.findall(res.get("answer") or "")})
    used = [cits[m - 1]["chunk_id"] for m in markers if 1 <= m <= len(cits)]
    in_range = all(1 <= m <= len(cits) for m in markers)
    ids_ok = all(c["chunk_id"] in retrieved for c in cits)
    return bool(markers) and in_range and ids_ok, markers, used


def summarize(rows):
    summary = {}
    for tag in ("ask", "stream"):
        ref = [r for r in rows if r["kind"] == "refuse"]
        ans = [r for r in rows if r["kind"] == "answerable"]
        answered = [r for r in ans if not r[tag]["refused"]]
        lats = [r[tag]["latency_s"] for r in rows]
        ans_lats = [r[tag]["latency_s"] for r in answered]
        summary[tag] = {
            "refusal_accuracy": sum(r[tag]["refused"] for r in ref) / len(ref),
            "answer_rate": len(answered) / len(ans),
            "citation_id_validity": (sum(bool(r[tag]["citation_valid"]) for r in answered) / len(answered)) if answered else None,
            "sentence_citation_validity": (
                sum(r[tag]["sentence_check"]["valid_sentences"] for r in answered)
                / max(1, sum(r[tag]["sentence_check"]["cited_sentences"] for r in answered))
            ) if answered else None,
            "answers_fully_valid": (
                sum(1 for r in answered if r[tag]["sentence_check"]["cited_sentences"]
                    and r[tag]["sentence_check"]["valid_sentences"] == r[tag]["sentence_check"]["cited_sentences"]
                    and r[tag]["sentence_check"]["uncited_sentences"] == 0) / len(answered)
            ) if answered else None,
            "p50_all_s": pct(lats, .5), "p95_all_s": pct(lats, .95),
            "p50_answered_s": pct(ans_lats, .5), "p95_answered_s": pct(ans_lats, .95),
        }
    ttfts = [r["stream"]["ttft_s"] for r in rows if r["stream"]["ttft_s"]]
    summary["stream"]["ttft_p50_s"] = pct(ttfts, .5)
    summary["stream"]["ttft_p95_s"] = pct(ttfts, .95)
    return summary


def rescore(path: str, base: str) -> None:
    """Recompute sentence checks + summary for a saved run (re-fetches chunk text)."""
    d = json.loads(Path(path).read_text())
    with httpx.Client(timeout=120) as c:
        for row in d["rows"]:
            chunks = retrieved_chunks(c, base, row["question"])
            for tag in ("ask", "stream"):
                x = row[tag]
                if not x["refused"]:
                    rr = {"answer": x["answer"], "citations": [{"chunk_id": i} for i in x["all_chunk_ids"]]}
                    x["sentence_check"] = sentence_support(rr, chunks, row["question"])
    d["summary"] = summarize(d["rows"])
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(json.dumps(d["summary"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--out", default=str(ROOT / "docs/e2e_week2_ollama.json"))
    ap.add_argument("--rescore", help="recompute metrics for an existing JSON")
    a = ap.parse_args()
    if a.rescore:
        return rescore(a.rescore, a.base)
    answerable = json.loads((ROOT / "docs/answerable_samples_week2.json").read_text())
    refuse = json.loads((ROOT / "docs/refuse_samples_week2.json").read_text())
    rows = []
    with httpx.Client(timeout=600) as c:
        health = c.get(f"{a.base}/health").json()
        for kind, items in (("answerable", answerable), ("refuse", refuse)):
            for it in items:
                q = it["question"]
                chunks = retrieved_chunks(c, a.base, q)
                retrieved = set(chunks)
                res, lat = ask(c, a.base, q)
                sres, slat, ttft, ntok = ask_stream(c, a.base, q)
                row = {"kind": kind, "id": it["id"], "question": q}
                for tag, rr, l in (("ask", res, lat), ("stream", sres, slat)):
                    ok, markers, used = check_citations(rr, retrieved) if not rr["refused"] else (None, [], [])
                    row[tag] = {
                        "refused": rr["refused"], "refusal_reason": rr["refusal_reason"],
                        "answer": rr["answer"], "markers": markers, "cited_chunk_ids": used,
                        "citation_valid": ok, "latency_s": round(l, 3),
                        "all_chunk_ids": [x["chunk_id"] for x in rr["citations"]],
                        "sentence_check": None if rr["refused"] else sentence_support(rr, chunks, q),
                    }
                row["stream"]["ttft_s"] = round(ttft, 3) if ttft else None
                row["stream"]["n_token_events"] = ntok
                rows.append(row)
                print(kind, it["id"], "ask:", "REF" if res["refused"] else "ANS", f"{lat:.1f}s",
                      "| stream:", "REF" if sres["refused"] else "ANS", f"{slat:.1f}s",
                      res["refusal_reason"] or "", flush=True)
    summary = summarize(rows)
    Path(a.out).write_text(json.dumps({"health": health, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
