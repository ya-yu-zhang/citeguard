# Week 3: semantic faithfulness judge (deepseek/deepseek-v4-flash via OpenRouter)

Status: **experimental; NOT wired into /ask.** Module `src/citeguard/ask/judge.py`, eval `scripts/eval_judge.py`,
gold labels `docs/judge_gold_labels.json` (**pending Thursday human review**), raw results
`docs/judge_eval_week3_v1.json`, `docs/judge_eval_week3_v2.json`.

Reported as **agreement with human labels X/N**, not as a faithfulness score.
Labeled set: 70 answered cases (61 unique question/answer pairs), 9 hand-labelled wrong
(3B Week 2 Q1/Q5/Q6 ask+stream, Q9 stream, hard Q3 ask+stream), 61 correct (other 3B answers,
Qwen Week 2 answers, Qwen's 14 regenerated drafts). 164 sentences. Sentence labels inside the wrong
answers are agent-proposed; all sentences in correct answers are labelled supported.

Settings: temperature 0, max_tokens 300, JSON mode, reasoning disabled. Errors fail closed (= flagged).

## Order of work (to avoid fitting the judge to the gate)
1. Gold labels written first. 2. Full run with the original prompt (v1). 3. One prompt revision (v2: the
cited text must actually answer the asked question; consistency is not enough), full rerun.
The v2 revision was motivated by the Q1 miss, so the gate set is **not held-out**, and it is small (9 wrong answers).

## Results

| | v1 (baseline) | v2 (revised) |
|---|---|---|
| Per-answer agreement | **61/70** | **58/70** |
| Per-sentence agreement (errors = disagree) | 151/164 | 148/164 |
| Wrong flagged (TP) / missed (FN) | 6 / 3 | 7 / 2 |
| Correct flagged (FP) / passed (TN) | 6 / 55 | 10 / 51 |
| Gate (a): flag all 9 wrong | FAIL (missed Q1 ask, Q1 stream, Q9 stream) | FAIL (missed Q1 ask, Q1 stream) |
| Gate (b): pass Qwen's 14 drafts | FAIL 11/14 (Q6, Q15 flagged; Q11 bad_json) | FAIL 9/14 (Q5 count mismatch; Q6, Q11, Q15, QH5 flagged) |
| 3B correct answers passed | 3B converted Q5 ask+stream flagged | + 3B Q12 stream, Q20 ask flagged |
| JSON/format errors | 2 bad_json (likely truncated at max_tokens 300) | 1 verdict_count_mismatch |

v2 caught Q9-stream but **still missed Q1** (the case it was written for) and added 4 new false rejections
of correct answers (3B Q12 stream, 3B Q20 ask, Qwen Q4 stream, Qwen draft QH5). Net: v2 is worse than v1.
Note that 3B Q12 stream was already borderline in the hand labels.

## Cost
Credit (OpenRouter /api/v1/key usage): 0.005049 before the task → 0.016686 after; **≈$0.0116 spent** of the $0.05 cap
(includes ~$0.0004 from the interrupted/first-5 runs). ≈$0.00009 per judged answer.

## Wiring proposal (not implemented)
Neither prompt meets the gates, so **do not wire it as-is.** If wiring is pursued after more work:
- **Combine, don't replace:** keep the lexical overlap rule (thresholds unchanged) and use the judge as an
  *extra veto* only on answers the overlap rule already passes. Expected effect, from this set: 3B would lose
  the 6 true-wrong answers it caught (good) plus 2 of its 33 correct answered cases (converted Q5 ask+stream) with v1;
  Qwen would lose 1 of its 14 Week 2 answered cases (Q7 ask, a JSON error). Answer rate drops by a few points for both;
  latency +~1–3 s per answer and ~$0.0001/answer.
- **Replace (judge rescues overlap-blocked drafts):** for Qwen, v1 would have passed 11/14 blocked drafts, raising Qwen's
  answer rate substantially, but the judge would then be the only guard while still missing Q1-type off-target answers
  (2/9 wrong answers pass in both rounds). Not recommended until gate (a) passes on a held-out set.
- Before either: raise max_tokens / retry once on bad JSON (format errors: 2 in v1, 1 in v2, of 61 calls each; all fail closed and became
  false rejections), and build a held-out wrong-answer set beyond these 9.

## Latency note
3B vs Qwen latency is **not comparable** until 3B is rerun on an idle machine (3B 1.4→7.1 s jump is unexplained, likely box load).
