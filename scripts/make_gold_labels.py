"""Build docs/judge_gold_labels.json (answer + sentence labels) from eval cases.
Sentence labels for wrong answers come from SENTENCE_OVERRIDES (hand-labelled)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from eval_judge import cases  # noqa
from citeguard.ask.judge import build_items  # noqa
OVR = json.loads((ROOT / "docs/_gold_sentence_overrides.json").read_text()) if (ROOT / "docs/_gold_sentence_overrides.json").exists() else {}
out, seen = [], set()
for c in cases("http://localhost:8000"):
    items = build_items(c["answer"], ["x"] * 10)
    sl = OVR.get(c["case"], [True] * len(items) if not c["wrong"] else [None] * len(items))
    out.append(dict(case=c["case"], model=c["model"], question=c["question"], answer=c["answer"],
                    chunk_ids=c["chunk_ids"], answer_label="wrong" if c["wrong"] else "correct",
                    sentences=[dict(i=i + 1, sentence=it["sentence"], supported=s) for i, (it, s) in enumerate(zip(items, sl))]))
json.dump({"status": "pending Thursday human review", "labeler": "agent draft of zhang yu's answer-level hand labels; sentence labels for wrong answers are agent-proposed",
           "note": "answer_label wrong = should be flagged; sentence supported=false marks the offending sentence(s)",
           "cases": out}, open(ROOT / "docs/judge_gold_labels.json", "w"), indent=1, ensure_ascii=False)
for c in out:
    if c["answer_label"] == "wrong":
        print("##", c["case"], "|", c["question"])
        for s in c["sentences"]: print("  ", s["i"], s["sentence"])
print(len(out), "cases")
