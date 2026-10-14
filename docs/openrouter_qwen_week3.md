# Week 3 — 3B local (Ollama) vs `qwen/qwen3.7-flash` (OpenRouter)

Run date: 2026-10-14. Both models use the same post-inline index (3,903 chunks), the same prompt and pipeline, and the same thresholds: the evidence gate, per-sentence support of 0.5, the task filter, and buffer-then-stream. **No tuning.**
- **3B:** `qwen2.5:3b-instruct` Q4_K_M on Ollama, CPU. Results: `docs/e2e_week2_ollama.json`, `docs/refuse_hard_week3_results.json`.
- **Qwen:** `qwen/qwen3.7-flash` via `LLM_BASE_URL=https://openrouter.ai/api/v1`, temperature 0, `max_tokens` 300, `LLM_EXTRA_BODY={"reasoning":{"enabled":false},"usage":{"include":true}}`. It ran on a separate uvicorn (port 8001) with env overrides; the default `.env` model is still the 3B. The fallback model was not needed (0 errors). Results: `docs/e2e_openrouter_qwen.json`, `docs/refuse_hard_openrouter_qwen.json`, per-call usage `docs/usage_openrouter_qwen.jsonl`.

Correctness was **hand-checked for every answered response** on both paths.

## Side by side

| | 3B `/ask` | 3B stream | Qwen `/ask` | Qwen stream |
|---|---|---|---|---|
| **Week 2 answerable (20)**: answered | 18 (0.90) | 18 (0.90) | 7 (0.35) | 7 (0.35) |
| ↳ correct / wrong (hand-checked) | 15 / **3** | 14 / **4** | **7 / 0** | **7 / 0** |
| ↳ refusals by reason | `unsupported_citation` 1, model said not in docs 1 | `low_grounding` 1, model 1 | `unsupported_citation` **12**, model 1 | `unsupported_citation` **12**, model 1 |
| **Week 2 refuse (20)**: refusal accuracy | 1.00 | 1.00 | 1.00 | 1.00 |
| ↳ breakdown (gate / task / model / ungrounded) | 8 / 1 / 8 / 3 | 8 / 1 / 8 / 3 | 8 / 1 / 11 / 0 | 8 / 1 / 11 / 0 |
| **Week 3 hard refuse (18)**: refused | 17 | 17 | **18** | **18** |
| ↳ false answers | Q3 | Q3 | 0 | 0 |
| ↳ breakdown (gate / model / ungrounded) | 2 / 12 / 3 | 2 / 14 / 1 | 2 / 16 / 0 | 2 / 16 / 0 |
| **Week 3 converted (2)**: answered (correct) | 2 (2) | 2 (2) | 0 | 0 |
| ↳ refusal reason | — | — | `unsupported_citation` ×2 | `unsupported_citation` ×2 |
| Latency p50 / p95, all 40 Week 2 questions | 7.12 / 9.97 s | 0.70 / 5.37 s¹ | 2.08 / 3.12 s | 2.16 / 3.23 s |
| Latency p50, answered only | 8.70 s | 3.88 s¹ | 2.27 s | 2.61 s |
| Time to first token p50 (stream, buffered) | — | 3.88 s | — | 2.60 s |
| Prompt / completion tokens per LLM call, p50 (p95) | ≈1,396 (1,829)² / n/a | | 1,470 (1,754) / 68 (105) | |

¹ The 3B stream call runs right after `/ask` for the same question, so Ollama's prompt cache is warm.
² Counted offline with the Qwen2.5 tokenizer (`scripts/measure_prompt_tokens.py`); see the token section.

### Wrong answers (all from the 3B; Qwen had none)
- **Q1, path parameter (both paths):** describes the `{file_path:path}` converter instead of the basic `{item_id}` syntax. Off-target; the gold chunk is at hybrid rank 6.
- **Q5, Depends (both paths):** "Dependencies are used to declare function parameters that are not path parameters, such as query parameters, request bodies…" Wrong.
- **Q6, run after response (both paths):** "use the `response_class` parameter…" Wrong; it's `BackgroundTasks`.
- **Q9, response_model (stream only):** "declare the path operation function with the `response_model` parameter". Imprecise and wrong; it's a decorator argument. `/ask` refused this one.
- **Hard Q3, workers × cores (both paths):** "the number of Uvicorn workers should be set to the number of CPU cores…" Not stated in the docs.
- **Borderline but counted correct:** 3B stream Q12 says to "specify the mode as 'json'" for `model_dump`. The mode is optional, so this is misleading but not false.

### Why Qwen answers so few: verifier false rejections, not model errors
`/ask` refused 12 of 20 answerable questions as `ungrounded_answer:unsupported_citation`, and both converted questions the same way.
- **I regenerated the blocked drafts** (14 extra calls, included in the spend; `docs/openrouter_qwen_blocked_drafts.json`). They are correct and well cited. Examples:
  - **Q1:** "include it in the route path string, such as `/items/{item_id}` [4]". Correct, but its key-term overlap with the chunk is 0.40, below 0.5.
  - **Q15:** `{"extra": "forbid"}` [1]. Correct.
  - **H6:** "caches the result … for the scope of a single request [1] … `use_cache=False` [1]". Correct, overlap 0.33.
- **Cause:** Qwen paraphrases more than the 3B (e.g. "extracts and validates" vs the docs' wording), so lexical support often lands at 0.3–0.45. Under the current rule, **any** cited sentence below 0.5 refuses the whole answer.
- **Effect:** the lexical verifier blocks hallucinations from the small model but **penalises a better paraphrasing model**. It needs a semantic check (NLI or an LLM judge) before a stronger model can show its value. No threshold was changed here.
- **Nondeterminism:** some drafts regenerated with no failing sentence, so the drafts that were actually blocked differed slightly. Qwen's `/ask` and stream also disagreed on Q2 and Q16 (each answered on one path only).

## Spend
| | USD |
|---|---|
| Key usage before this task | 0.000138 |
| After first 5 calls | 0.000590 (≈ $0.00005 per call; projected ≈ $0.006 for the full run, under the $0.05 cap, so I continued) |
| After all runs | **0.005049** |
| **Spent by this task** | **$0.00491** (5 probe calls + 103 run calls + 14 draft regenerations; the logged per-call costs of the 117 logged calls sum to $0.00461) |
| **Remaining credit** | **$0.494951** of $0.50 |

## Prompt tokens and the 3B latency jump (1.4 s → 7.1 s p50)
I measured the exact `/ask` prompt (top-5 hybrid, ≤ 1,200 chars per chunk) for the 40 Week 2 questions on both indexes, counting only questions that reach the LLM:

| Index | Questions reaching LLM | Prompt tokens p50 | p95 | Mean | Context chars p50 |
|---|---|---|---|---|---|
| pre-inline (`/workspace/citeguard_bak_pre_inline`) | 27 | 1,316 | 1,820 | 1,327 | 3,876 |
| post-inline (current) | 32 | 1,396 | 1,829 | 1,377 | 4,424 |

Answer length barely changed either: median answer 224 → 243 chars.

**So prompt length is not the main cause** (+6 % at the median). What explains the jump:
1. **Fewer gate refusals:** 14 → 9 of 40 questions were refused instantly at the retrieval gate. These take about 0.1 s, so fewer of them raises the all-question p50. This is a *mix* effect.
2. **Slower LLM calls themselves:** median per-call latency of LLM-reaching `/ask` calls was 3.0 s pre-inline vs 8.1 s post-inline, with near-identical prompt and output sizes. A controlled probe just now (8 calls, 3B, prompts 1,158–1,573 tokens, 23–99 output tokens, logged in `docs/usage_ollama_3b_probe.jsonl`) took **9.5–12.7 s per call**. The box's load average was 2–7 at that time; other agents share this CPU.

The ~3 s calls in the pre-inline run were the outlier, not the post-inline ones. The likely cause is box load or Ollama cache state rather than the index, but I can't confirm it without an alternating A/B in one session. **Not resolved; no truncation added.**

Token usage is now logged per request when `LLM_USAGE_LOG` is set (`src/citeguard/ask/llm.py::_log_usage`; model, prompt/completion tokens, cost, latency, no text or keys).
