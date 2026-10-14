# Week 3 — inline FastAPI `docs_src` code into the index

Run date: 2026-10-14. Local only, same model (`qwen2.5:3b-instruct`, Ollama CPU). **No thresholds or prompts were changed.**

## What changed in ingest
1. **Include resolution.** FastAPI pages reference their example code as `{* ../../docs_src/<path>.py hl[..] title[..] *}`. `docs_src/` was fetched at the **same pinned commit** (`fastapi/fastapi@f5c6e9b4`) and each placeholder is replaced by a fenced ```python block (prefixed `# docs_src/<path>`). `hl`/`ln`/`title` options are display-only and ignored (whole file inlined, capped at 6,000 chars).
   - Resolved: **442**. Unresolved: **1** (`how-to/configure-swagger-ui.md` includes `../../fastapi/openapi/docs.py`, which is library source rather than `docs_src`).
   - Code: `src/citeguard/ingest/pipeline.py::resolve_includes`.
2. **Chunker bug fix (needed for 1, and it affects all sources).** `# comment` lines inside fenced code blocks were being treated as Markdown headings. That split sections at code comments and produced bogus title paths like `CMD ["fastapi", "run", …]`, `(1)!` or `Create store with semantic search enabled`. Inlining Python code would have made this much worse. Headings inside ``` / ~~~ fences are now ignored (`chunk.py::_fence_spans`).

### Chunk counts (old → new = 4,100 → **3,903**)
Each change in isolation (chunking only, same corpus):

| | no fence fix | fence fix |
|---|---|---|
| **no inlining** | 4,100 (old index) | 3,787 |
| **inlining** | 4,638 | **3,903 (new index)** |

Per source, old → new: FastAPI 2,159 → 2,257 · Pydantic 867 → 788 · LangGraph 1,074 → 858. All chunk_ids are unique (3,903/3,903). The total *drops* because the fence fix merges the fake "comment-heading" fragments; inlining alone adds 538 chunks. **Both changes ship together, so the before/after numbers below reflect both, not inlining alone.** Pydantic and LangGraph chunk_ids shifted too.

## Week 1 retrieval eval (30 queries, doc-level gold)
| Method | R@5 before → after | MRR before → after | p50 / p95 latency before → after |
|---|---|---|---|
| bm25 | 0.733 → **0.733** | 0.524 → **0.575** | 0.008/0.011 s → 0.006/0.007 s |
| vector | 0.867 → **0.867** | 0.676 → **0.687** | 0.040/0.049 s → 0.032/0.040 s |
| hybrid | 0.900 → **0.967** | 0.723 → **0.761** | 0.054/0.068 s → 0.048/0.056 s |
| hybrid+rerank | 0.933 → **0.933** | 0.669 → **0.721** | 0.198/0.227 s → 0.193/0.224 s |

Hits that flipped:
- hybrid: +"How do I define a Pydantic BaseModel?" and +"How to configure model_config in Pydantic v2?" (no losses).
- bm25: +"upload a file", −"Pydantic field validation".

Default hybrid now beats hybrid+rerank on R@5 (0.967 vs 0.933). Files: `docs/eval_week1.{md,json}` (after), `docs/eval_week1_pre_inline.{md,json}` (before). The README retrieval table was **not** touched (小助手2 owns it).

## Week 2 samples (20 answerable + 20 refuse) via `/ask` and `/ask/stream`
| Metric | before `/ask` | before stream | **after `/ask`** | **after stream** |
|---|---|---|---|---|
| Refusal accuracy | 1.00 | 1.00 | 1.00 | 1.00 |
| Answer rate | 0.75 | 0.75 | 0.90 | 0.90 |
| Citation ID validity | 1.00 | 1.00 | 1.00 | 1.00 |
| Sentence citation validity (lexical) | 1.00 | 1.00 | 1.00 | 1.00 |
| p50 latency, all (s) | 1.41 | 1.23 | 7.12 | 0.70 |
| p95 latency, all (s) | 6.33 | 6.13 | 9.97 | 5.37 |
| p50 latency, answered (s) | 4.04 | 4.09 | 8.70 | 3.88 |
| p95 latency, answered (s) | 7.29 | 7.05 | 11.13 | 5.75 |
| Stream TTFT p50 (s) | — | 4.08 | — | 3.88 |

Refusal breakdown (`kind:stage` → count):

- before `ask`: answerable:ANS=15, answerable:model=1, answerable:ungrounded=4, refuse:gate=13, refuse:model=4, refuse:task=1, refuse:ungrounded=2
- before `stream`: answerable:ANS=15, answerable:model=1, answerable:ungrounded=4, refuse:gate=13, refuse:model=4, refuse:task=1, refuse:ungrounded=2
- after `ask`: answerable:ANS=18, answerable:model=1, answerable:ungrounded=1, refuse:gate=8, refuse:model=7, refuse:task=1, refuse:ungrounded=4
- after `stream`: answerable:ANS=18, answerable:model=1, answerable:ungrounded=1, refuse:gate=8, refuse:model=7, refuse:task=1, refuse:ungrounded=4

On the refuse set, the retrieval gate now stops 8 questions instead of 13. The new index changed RRF scores, so more refuse questions reach the LLM; the model and the verifier still refused all of them.

**Manual correctness check of the 18 answered (after).** 15 correct and grounded. **3 wrong or off-target:**
- **Q1 (declare a path parameter):** answers with the `{file_path:path}` converter. The gold chunk is now at hybrid rank 6 (was >50), just outside the top 5, so the model used the nearest path-params section.
- **Q5 (Depends):** "Dependencies are used to declare function parameters that are not path parameters, such as query parameters, request bodies…" Wrong.
- **Q6 (run after response):** says use `response_class`. Wrong; it's `BackgroundTasks`.

Q9 (`response_model`), which was answered wrongly before, is now refused (`ungrounded_answer:unsupported_citation`). Q4, Q7 and Q15, refused before, are now answered correctly.

**Net:** correct answers 13 → 15, wrong answers 1 → 3, answer rate 0.75 → 0.90. The "sentence citation validity 1.00" is by construction (same lexical rule as the repair step), so it doesn't detect these errors.

## Week 3 hard set (18 should-refuse + 2 converted)
| | before `/ask` | before stream | **after `/ask`** | **after stream** |
|---|---|---|---|---|
| Refused / 18 | 18 | 18 | 17 | 17 |
| no_evidence: retrieval gate | 1 | 1 | 2 | 2 |
| no_evidence: model INSUFFICIENT_EVIDENCE | 14 | 15 | 12 | 14 |
| ungrounded_answer: no_supported_sentences | 1 | 1 | 3 | 1 |
| ungrounded_answer: unsupported_citation | 2 | 1 | 0 | 0 |
| False answers | 0 | 0 | [3] | [3] |

**New false answer, Q3** ("how many times the CPU core count should uvicorn workers be"), on both paths:
> "When deploying, the number of Uvicorn workers should be set to the number of CPU cores to parallelize the execution of the application and take advantage of multiple cores in the CPU [3]." (cites `3e0058788245-0005`)

The docs never give that ratio. The sentence reuses the chunk's words ("parallelize", "multiple cores"), so the lexical check passes. Q3 was already marked borderline in screening.

Converted questions (answerable): **Q5** "How do I use GraphQL with FastAPI?" → answered on both paths, citing the GraphQL page (`81142cbac393-0002`). It now includes the inlined Strawberry example code, which is correct. **Q6** "Is the result of a FastAPI dependency cached, and for what scope?" → "cached per request [1]" is correct. Its second sentence about `Depends(scope="function")` is about dependency teardown timing, not the cache scope, so it is off-topic but grounded.

## Gold-chunk ranks for Week 2 Q1 / Q4 / Q7 / Q15 (top_k = 50)
| Q | Gold (answer chunk) | hybrid before → after | BM25 before → after | vector before → after |
|---|---|---|---|---|
| 1 | path-params `c3fbe41174be-0000` ("same syntax used by Python format strings"; now with inlined code) | >50 → **6** | >50 → >50 | >50 → **2** |
| 4 | handling-errors chunk stating/raising `HTTPException` (any of 7 chunks after inlining; before: only `-0001`) | >50 → **9** (`-0009`) | >50 → 21 | >50 → **2** |
| 4 (strict) | exactly `b05842751c15-0001` (prose only, unchanged) | >50 → >50 | >50 → >50 | >50 → 34 |
| 7 | cors.md `0e60b3828d2a-0004` | 1 → 1 | 3 → 3 | 1 → 1 |
| 15 | `8555df178cda-0037` (`extra='forbid'`) | 3 → 3 | 5 → 6 | 4 → 4 |

Q4 is now answered from `b05842751c15-0004` (hybrid rank 4). That page section now contains inlined `raise HTTPException(...)` code, which wasn't counted by the "after" gold predicate because `-0004` covers a later subsection. Q1 and Q4 moved from ">50" into vector top-2 but not hybrid top-5. BM25 is what holds them back.

## Files
- `docs/e2e_week2_ollama.json` (after); `docs/e2e_week2_ollama_pre_inline.json` (before).
- `docs/refuse_hard_week3_results.json` (after, now including the 2 converted); `docs/refuse_hard_week3_results_pre_inline.json` (before).
- `docs/eval_week1*.{md,json}`.
- Pre-inline index backup on the box (not in git): `/workspace/citeguard_bak_pre_inline/`.
