"""Week 3: run the hard should-refuse set through /ask and /ask/stream (no tuning).

Usage: python scripts/run_hard_refuse.py  (server on :8000)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e2e_ask import ask, ask_stream  # noqa: E402

BASE = "http://localhost:8000"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=str(ROOT / "docs/refuse_hard_week3_results.json"))
    a = ap.parse_args()
    qs = json.loads((ROOT / "docs/refuse_hard_week3.json").read_text())
    rows = []
    with httpx.Client(timeout=600) as c:
        for q in qs:
            if q.get("screening") == "keep_refuse":
                kind, question = "refuse", q["question"]
            elif q.get("converted_answerable"):
                kind, question = "converted_answerable", q["converted_answerable"]["question"]
            else:
                continue
            res, lat = ask(c, a.base, question)
            sres, slat, _, _ = ask_stream(c, a.base, question)
            row = {"id": q["id"], "kind": kind, "question": question}
            for tag, r, l in (("ask", res, lat), ("stream", sres, slat)):
                row[tag] = {"refused": r["refused"], "refusal_reason": r["refusal_reason"],
                            "answer": r["answer"], "citations": [x["chunk_id"] for x in r["citations"]],
                            "latency_s": round(l, 2)}
            rows.append(row)
            print(q["id"], kind, "ask:", res["refusal_reason"] or "ANSWERED", "| stream:", sres["refusal_reason"] or "ANSWERED", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
