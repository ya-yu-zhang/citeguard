"""Validate the semantic faithfulness judge against hand labels (answer level).

Positive = hand-labelled WRONG answer (judge should flag it).
Cases:
  * 3B post-inline Week 2 answers (/ask + stream)
  * 3B Week 3 hard answers (false answer Q3) + converted Q5/Q6
  * Qwen Week 2 answers (/ask + stream)
  * Qwen's 14 regenerated blocked drafts
Identical (question, answer) pairs are judged once and the verdict is reused.
Budget guard: aborts when the summed reported cost exceeds --cap.

Usage: python scripts/eval_judge.py [--limit N] [--cap 0.05]   (server on :8000 for /search)
Cache: docs/judge_eval_week3.json (re-runs only judge missing pairs)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from citeguard.ask.judge import judge_answer  # noqa: E402

GOLD = ROOT / "docs/judge_gold_labels.json"
WRONG_W2_3B = {"ask": {1, 5, 6}, "stream": {1, 5, 6, 9}}
WRONG_HARD_3B = {3}
CHUNKS = {}
for line in (ROOT / "data/processed/chunks.jsonl").open():
    c = json.loads(line)
    CHUNKS[c["chunk_id"]] = c["text"]


def cases(base: str) -> list[dict]:
    out = []
    d = json.loads((ROOT / "docs/e2e_week2_ollama.json").read_text())
    for r in d["rows"]:
        for t in ("ask", "stream"):
            x = r[t]
            if x["refused"]:
                continue
            wrong = (r["kind"] == "answerable" and r["id"] in WRONG_W2_3B[t]) or r["kind"] == "refuse"
            out.append(dict(case=f"3B/w2/{r['kind']}/Q{r['id']}/{t}", model="3B", question=r["question"],
                            answer=x["answer"], chunk_ids=x["all_chunk_ids"], wrong=wrong))
    for r in json.loads((ROOT / "docs/refuse_hard_week3_results.json").read_text()):
        for t in ("ask", "stream"):
            x = r[t]
            if x["refused"]:
                continue
            wrong = r["kind"] == "refuse" and r["id"] in WRONG_HARD_3B or r["kind"] == "refuse"
            out.append(dict(case=f"3B/hard/{r['kind']}/Q{r['id']}/{t}", model="3B", question=r["question"],
                            answer=x["answer"], chunk_ids=x["citations"], wrong=wrong))
    d = json.loads((ROOT / "docs/e2e_openrouter_qwen.json").read_text())
    for r in d["rows"]:
        for t in ("ask", "stream"):
            x = r[t]
            if not x["refused"]:
                out.append(dict(case=f"Qwen/w2/{r['kind']}/Q{r['id']}/{t}", model="Qwen", question=r["question"],
                                answer=x["answer"], chunk_ids=x["all_chunk_ids"], wrong=r["kind"] == "refuse"))
    for dr in json.loads((ROOT / "docs/openrouter_qwen_blocked_drafts.json").read_text()):
        hits = httpx.post(f"{base}/search", json={"query": dr["question"], "top_k": 5, "method": "hybrid"},
                          timeout=60).json()["results"]
        out.append(dict(case=f"Qwen/draft/Q{dr['id']}", model="Qwen-draft", question=dr["question"],
                        answer=dr["raw"], chunk_ids=[h["chunk_id"] for h in hits], wrong=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="judge at most N new pairs")
    ap.add_argument("--cap", type=float, default=0.05)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--tag", default="", help="output suffix (default = prompt)")
    a = ap.parse_args()
    import os
    os.environ["JUDGE_PROMPT"] = a.prompt
    global OUT
    OUT = ROOT / f"docs/judge_eval_week3_{a.tag or a.prompt}.json"
    cache = json.loads(OUT.read_text()) if OUT.exists() else {"judged": {}, "cases": []}
    judged = cache["judged"]
    spent = sum((j.get("usage") or {}).get("cost") or 0 for j in judged.values())
    cs = json.loads(GOLD.read_text())["cases"]
    for c in cs:
        c["wrong"] = c["answer_label"] == "wrong"
    new = 0
    for c in cs:
        key = c["question"] + "\x00" + c["answer"]
        c["key"] = key
        if key in judged:
            continue
        if a.limit and new >= a.limit:
            continue
        if spent > a.cap:
            print("ABORT: cap reached", spent)
            break
        res = judge_answer(c["question"], c["answer"], [CHUNKS.get(i, "") for i in c["chunk_ids"]])
        judged[key] = {"verdicts": res.verdicts, "answers_question": res.answers_question,
                       "flagged": res.flagged, "error": res.error, "usage": res.usage,
                       "latency_s": round(res.latency_s, 2)}
        spent += (res.usage or {}).get("cost") or 0
        new += 1
        OUT.write_text(json.dumps({"judged": judged, "cases": cache.get("cases", [])}, indent=1, ensure_ascii=False))
        print(f"{c['case']:40s} wrong={c['wrong']!s:5s} flagged={res.flagged!s:5s} err={res.error} cost={spent:.5f}", flush=True)
    for c in cs:
        j = judged.get(c["key"])
        c["flagged"] = j["flagged"] if j else None
    cache["cases"] = [{"case": c["case"], "wrong": c["wrong"], "flagged": c["flagged"], "key_q": c["question"]} for c in cs]
    cache["prompt"] = a.prompt
    cache["spent_reported"] = spent
    OUT.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    done = [c for c in cs if c["flagged"] is not None]
    tp = sum(c["wrong"] and c["flagged"] for c in done); fn = sum(c["wrong"] and not c["flagged"] for c in done)
    fp = sum(not c["wrong"] and c["flagged"] for c in done); tn = sum(not c["wrong"] and not c["flagged"] for c in done)
    sa = sn = 0
    for c in done:
        j = judged[c["key"]]
        for g, v in zip(c["sentences"], j["verdicts"] if not j["error"] else []):
            sn += 1; sa += bool(v.get("supported")) == g["supported"]
        if j["error"]:
            sn += len(c["sentences"])
    print(f"per-answer agreement {tp+tn}/{len(done)}; per-sentence agreement {sa}/{sn} (errors count as disagreement)")
    print(f"cases judged {len(done)}/{len(cs)}  TP={tp} FN={fn} FP={fp} TN={tn}  reported_cost={spent:.5f}")


if __name__ == "__main__":
    main()
