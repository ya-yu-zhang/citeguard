"""Agreement report for a judge results file, standalone and veto mode, two label views.

Label views (gold file is NOT edited):
  original   — docs/judge_gold_labels.json as is
  incomplete — 3B Week 2 Q1 ask/stream relabelled 'incomplete' (not wrong: true but off-target);
               decided AFTER seeing judge results, so it is post-hoc, not an independent label.
Veto mode: the overlap rule (unchanged) runs first. Served answers passed it; Qwen's 14 drafts were
blocked by it, so in veto mode they are refused regardless of the judge.
Usage: python scripts/judge_report.py docs/judge_eval_week3_v1r.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCOMPLETE = {"3B/w2/answerable/Q1/ask", "3B/w2/answerable/Q1/stream"}
# post-review (teammate review 2026-10-14 + my check of Q15 s3 against its cited chunk; post-hoc):
POST_INCOMPLETE = INCOMPLETE | {"3B/w2/answerable/Q20/ask"}
POST_WRONG = {"Qwen/draft/Q6": None,  # no BackgroundTasks; middleware claim wrong (sentence labels unknown)
              "Qwen/draft/Q15": [True, True, False]}  # s3 version claims not in cited chunk [3]


def main(path: str) -> None:
    res = json.loads(Path(path).read_text())["judged"]
    gold = json.loads((ROOT / "docs/judge_gold_labels.json").read_text())["cases"]
    for view in ("original", "incomplete", "post-review"):
        for mode in ("standalone", "veto"):
            tp = fn = fp = tn = sa = sn = 0
            fps = []
            for c in gold:
                j = res[c["question"] + "\x00" + c["answer"]]
                inc = INCOMPLETE if view == "incomplete" else POST_INCOMPLETE if view == "post-review" else set()
                bad = (c["answer_label"] == "wrong" and c["case"] not in inc) or (view == "post-review" and c["case"] in POST_WRONG)
                draft = c["model"] == "Qwen-draft"
                refused = (draft or j["flagged"]) if mode == "veto" else j["flagged"]
                if bad and refused: tp += 1
                elif bad: fn += 1
                elif refused: fp += 1; fps.append(c["case"])
                else: tn += 1
                if mode == "standalone":
                    gs = [s["supported"] for s in c["sentences"]]
                    if c["case"] in inc and c["answer_label"] == "wrong":
                        gs = [True] * len(gs)
                    if view == "post-review" and c["case"] in POST_WRONG:
                        if POST_WRONG[c["case"]] is None:
                            continue  # sentence labels not reviewed; excluded from per-sentence
                        gs = POST_WRONG[c["case"]]
                    vs = [] if j["error"] else j["verdicts"]
                    sn += len(gs); sa += sum(bool(v.get("supported")) == g for v, g in zip(vs, gs))
            n = tp + fn + fp + tn
            s = f" | per-sentence {sa}/{sn}" if mode == "standalone" else ""
            print(f"[{view:10s} {mode:10s}] per-answer {tp+tn}/{n}  TP={tp} FN={fn} FP={fp} TN={tn}{s}")
            if mode == "veto":
                print("    false refusals:", ", ".join(fps))
    errs = [j["error"] for j in res.values() if j["error"]]
    retried = sum(1 for j in res.values() if (j.get("usage") or {}).get("retried_after"))
    cost = sum((j.get("usage") or {}).get("cost") or 0 for j in res.values())
    print(f"calls={len(res)} retried={retried} final_errors={errs} reported_cost={cost:.5f}")


if __name__ == "__main__":
    main(sys.argv[1])
