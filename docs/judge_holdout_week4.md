# Week 4 judge holdout set

Independent test for the semantic faithfulness judge. **Do not tune the prompt on this file.**

| Split | N | Purpose |
|---|---|---|
| wrong | 20 | Must be flagged |
| correct | 20 | Must not be flagged |

Error patterns in the wrong split: wrong API name, concept confusion, wrong parameter location, invented numeric claim, fabricated version, false mechanism, outdated API, overclaim, wrong scope, unsupported citation.

Suggested pass bar (agreed before looking at scores): catch ≥ 18/20 wrong and mis-refuse ≤ 2/20 correct on the frozen prompt.

Source file: `docs/judge_holdout_week4.json`.
