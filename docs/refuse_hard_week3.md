# Week 3 — hard should-refuse set (项目小助手1's Week 3 list)

Setup: the same stack as Week 2 v2.
- Model: `qwen2.5:3b-instruct` on Ollama, CPU.
- Retrieval: hybrid top 5. Evidence gate: RRF ≥ 0.02 and top-3 ≥ 200 chars.
- Per-sentence support threshold 0.5, plus the task filter.

**No thresholds, prompts or questions were changed for this run.** Run date: 2026-10-14.

Source: the 20-question list 项目小助手1 posted for Week 3 (FastAPI 7 / Pydantic 6 / LangGraph 5 / cross-doc 2), translated faithfully into English (the corpus is English). The Chinese originals are in `docs/refuse_hard_week3.json` (`question_zh`).
Raw `/ask` and `/ask/stream` outputs are in `docs/refuse_hard_week3_results.json`; the runner is `scripts/run_hard_refuse.py`.
An earlier run used the wrong list (the Week 2 应拒样例). It is kept as `docs/refuse_hard_week3_oldlist.{md,json}` and `docs/refuse_hard_week3_oldlist_results.json`.

## 1. Screening: do the docs actually answer it?

Method:
- `/search` (hybrid, top 5) for each question, and I read the chunks.
- A regex keyword scan over all 4,100 chunks for each question's key term, e.g. `requests per second`, `mongo`, `times faster`, `unsafe`, `check digit`, `v3`, `kafka`, `crewai`, `SLA`, `enterprise`.

| # | Area | Question (EN) | 原文 | Screening | Evidence / note |
|---|---|---|---|---|---|
| 1 | FastAPI | How many requests per second can FastAPI handle on a 4-core machine? | FastAPI 在 4 核机器上每秒能处理多少个请求？ | keep (refuse) | Top-5: concurrency/async (f4c940cf9e97-0012), `fastapi run`; keyword scan for 'requests per second'/'req/s' = 0 hits. Benchmarks page has no hardware-specific throughput number. |
| 2 | FastAPI | What is the maximum number of tasks BackgroundTasks can queue, and what happens when that limit is exceeded? | BackgroundTasks 最多能排多少个任务，超出以后会怎样？ | keep (refuse) | Background Tasks docs retrieved (de4625b764d1-0000, 8027ae2f3842-0005/-0006 'Caveat': use Celery for heavy work). No queue limit or overflow behaviour is stated anywhere. |
| 3 | FastAPI | When deploying, how many times the number of CPU cores should the number of uvicorn workers be set to? | 部署时 uvicorn 的 worker 数应该设成 CPU 核数的几倍？ | keep (borderline) | BORDERLINE, kept. Workers/cores are discussed (de9cd06037c6-0018 'multiple cores … multiple processes … workers'; 3e0058788245-0005; 9262c2155592-0039 example `--workers 4`), but no multiplier of CPU cores is recommended. Keyword scan 'number of cores' = 0. A grounded answer could only say 'docs give no ratio'. |
| 4 | FastAPI | Which has better performance: FastAPI's dependency injection or NestJS's? | FastAPI 的依赖注入和 NestJS 比，哪个性能更好？ | keep (refuse) | NestJS is mentioned only as an inspiration in alternatives.md (a4b5765b1e1b-0012); no DI performance comparison. |
| 5 | FastAPI | How do I make FastAPI automatically generate a GraphQL schema? | 怎么让 FastAPI 自动生成 GraphQL schema？ | **convert → answerable** | PARTIALLY ANSWERABLE (premise false). Docs say FastAPI integrates any ASGI GraphQL library and recommend Strawberry (81142cbac393-0000, 81142cbac393-0002); nothing says FastAPI auto-generates a GraphQL schema. The corrective answer is grounded, so this is not a clean should-refuse question. |
| 6 | FastAPI | Is the Depends cache shared across multiple worker processes? | Depends 的缓存在多个 worker 进程之间共享吗？ | **convert → answerable** | ANSWERABLE (by explicit scope). The dependency cache is per request: 'call that sub-dependency only once per request … cache … in that specific request' (667e40148410-0004). Worker processes are separate processes, each loading its own variables in RAM (de9cd06037c6-0022). A grounded answer is 'No, it is per-request'. |
| 7 | FastAPI | Which ORM does FastAPI officially recommend for connecting to MongoDB? | FastAPI 官方推荐用哪个 ORM 连 MongoDB？ | keep (refuse) | MongoDB appears only in a Motor test tip (083a313d635d-0006), Docker image list, alternatives, sponsor notes. No officially recommended Mongo ORM. |
| 8 | Pydantic | How many times faster is Pydantic v2 than v1? Please give concrete benchmark data. | Pydantic v2 比 v1 快多少倍？请给出具体基准数据。 | keep (refuse) | Performance chunks (2c2675a3b653-0002/-0003) give Pydantic vs dedicated code (3.45x) and TypedDict vs models (~2.5x, 8484900589c8-0007); no v2-vs-v1 benchmark. |
| 9 | Pydantic | How long does model_validate take to validate 100,000 records? | model_validate 校验 10 万条数据需要多长时间？ | keep (refuse) | Performance tips (8484900589c8-0001) give no timing for 100k records. |
| 10 | Pydantic | What unsafe code does Pydantic's Rust core use? | Pydantic 的 Rust 核心用了哪些 unsafe 代码？ | keep (refuse) | pydantic-core is Rust (2c2675a3b653-0002); 'unsafe' hits are only FastAPI CI and LangGraph error-handler defaults. Nothing about Rust unsafe code. |
| 11 | Pydantic | How does Pydantic validate the check digit of a Chinese national ID number? | Pydantic 怎么校验中国身份证号的校验位？ | keep (refuse) | Keyword scan for ID card / check digit = 0. Only generic field validators retrieved. |
| 12 | Pydantic | Can the result of a computed_field be persisted to a database? | computed_field 的结果能持久化到数据库吗？ | keep (borderline) | BORDERLINE, kept. computed_field docs (bc9c8dfb40a5-0024) say it is included in serialization and JSON Schema, and Pydantic does no extra logic; nothing about persisting to a database (Pydantic has no DB layer in these docs). |
| 13 | Pydantic | Which APIs will be removed in Pydantic v3? | Pydantic v3 会删掉哪些 API？ | keep (borderline) | BORDERLINE, kept. Version policy (6f621e822874-0003) says V3 breaking changes will be careful and pydantic-core will be merged; `__validators__` 'will be deprecated in v3' (13b45477775a-0042); an alias default may change (daaf3f27a982-0011). No list of APIs to be removed. |
| 14 | LangGraph | What is the maximum number of thread states a LangGraph checkpointer can store? | LangGraph 的 checkpointer 最多能存多少个线程状态？ | keep (refuse) | Persistence/threads docs (cc0bd87d2064-*, 4a50ddd9115d-0002) define threads/checkpoints; no maximum. |
| 15 | LangGraph | In multi-agent scenarios, which has lower latency: LangGraph or CrewAI? | LangGraph 和 CrewAI 在多 agent 场景下谁的延迟更低？ | keep (refuse) | Keyword scan 'crewai' = 0. |
| 16 | LangGraph | How is the enterprise edition of LangGraph Platform priced? | LangGraph Platform 的企业版怎么收费？ | keep (refuse) | No LangGraph Platform pricing; 'enterprise' hits are Pydantic support (6b1295503ceb-*), no price. |
| 17 | LangGraph | What is the maximum number of nodes a StateGraph can hold, and will it crash if that is exceeded? | 一个 StateGraph 最多能放多少个节点，超过会崩吗？ | keep (refuse) | Recursion limit (max super-steps, 8c9c196dbe78-0059) and GRAPH_RECURSION_LIMIT error (6bc259477f79-0000) retrieved; nothing about a maximum node count. Risk: model may conflate with recursion limit. |
| 18 | LangGraph | How does LangGraph natively integrate with Kafka for event-driven processing? | LangGraph 怎么原生接入 Kafka 做事件驱动？ | keep (refuse) | Kafka only in FastAPI release-note external links and a TODO comment (a2a846c50c74-0004). No native LangGraph Kafka integration. |
| 19 | 跨文档 | For a service built with FastAPI, Pydantic and LangGraph, how many nines of SLA can it achieve in production? | 用 FastAPI、Pydantic 和 LangGraph 搭的服务，生产环境 SLA 能达到几个 9？ | keep (refuse) | Only SLA mention is Pydantic enterprise support 'an SLA matched to your operating model' (6b1295503ceb-0000); no nines figure. |
| 20 | 跨文档 | Among the maintainers of these three libraries, who has contributed the most to async support? | 这三个库的维护者里，谁对异步支持贡献最多？ | keep (refuse) | Pydantic maintainers list retrieved (3a5385ca18c9-0001); no attribution of async contributions. |

**Result: 18 kept (3 borderline: Q3, Q12, Q13), 2 converted (Q5, Q6).**
- **Q5 (GraphQL):** the premise is false, but a corrective answer is fully grounded. FastAPI integrates any ASGI GraphQL library; Strawberry is recommended (`81142cbac393-0000`, `-0002`). Converted to *"How do I use GraphQL with FastAPI?"*. An informational `/ask` run answered: "…you can integrate Strawberry, which is recommended and has a design closest to FastAPI's [1]." Correct and grounded.
- **Q6 (Depends cache across workers):** the docs state that the cache is per request (`667e40148410-0004`), and that workers are separate processes with their own RAM (`de9cd06037c6-0022`). Converted to *"Is the result of a FastAPI dependency (Depends) cached, and for what scope?"*. An informational `/ask` run answered "…cached for a specific request… [1]". Correct and grounded.
- **Q3 (workers × cores):** kept. The docs discuss workers and cores, and give an example with `--workers 4`, but never give a ratio to CPU cores.
- **Q12 (computed_field → DB):** kept. The docs cover serialization and JSON Schema only.
- **Q13 (v3 removals):** kept. The docs have v3 intentions (careful breaking changes, `pydantic-core` merged, `__validators__` to be deprecated) but no list of removed APIs.

These three are borderline: a cautious grounded answer, "the docs don't specify X, but say Y", would be acceptable to a human reviewer. Counting them as should-refuse is the stricter reading.

## 2. Results on the 18 kept questions

| Metric | `/ask` | `/ask/stream` |
|---|---|---|
| Refused | **18 / 18** | **18 / 18** |
| False answers | 0 | 0 |
| `no_evidence` total | 15 | 16 |
| ↳ retrieval gate (`top_score_below_threshold`) | 1 (Q9) | 1 (Q9) |
| ↳ model said `INSUFFICIENT_EVIDENCE` | 14 | 15 |
| `ungrounded_answer` total | 3 (Q3, Q11, Q12) | 2 (Q3, Q12) |
| ↳ `no_supported_sentences` | 1 (Q3) | 1 (Q3) |
| ↳ `retrieval_miss` (cited sentence unsupported by any retrieved chunk) | 2 (Q11, Q12) | 1 (Q12) |

Observations:
- **All 18 were refused on both paths; zero false answers.** 17 of 18 got past the retrieval gate and reached the LLM, so this is a real test of the generation and verification stages.
- The model's own `INSUFFICIENT_EVIDENCE` did most of the work (14 on `/ask`). The citation verifier caught the rest:
  - Q3: `no_supported_sentences`.
  - Q11 and Q12: `retrieval_miss`. The model drafted an answer citing chunks that don't support it, and the verifier blocked it.
- **Reasons differ between the paths on Q11, decisions don't.** `/ask` gave `ungrounded_answer:retrieval_miss`; `/ask/stream` gave `no_evidence:model_declared_insufficient_evidence`. Both refused. The stream path is a separate generation call, and at temperature 0 the model still produced different drafts, so per-reason counts can differ by one or two between runs.
- Refused answers are not stored (the API returns `answer: ""`), so the text of the drafts blocked on Q3, Q11 and Q12 isn't available. Logging blocked drafts is a suggested addition.

Per question:

| # | `/ask` refusal_reason | `/ask/stream` refusal_reason |
|---|---|---|
| 1 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 2 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 3 | `ungrounded_answer:no_supported_sentences` | `ungrounded_answer:no_supported_sentences` |
| 4 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 7 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 8 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 9 | `no_evidence:top_score_below_threshold:0.0164<0.02` | `no_evidence:top_score_below_threshold:0.0164<0.02` |
| 10 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 11 | `ungrounded_answer:retrieval_miss` | `no_evidence:model_declared_insufficient_evidence` |
| 12 | `ungrounded_answer:retrieval_miss` | `ungrounded_answer:retrieval_miss` |
| 13 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 14 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 15 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 16 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 17 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 18 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 19 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 20 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |

## 3. Converted questions
See section 1. Q5 and Q6 should move to the answerable set; both were answered correctly with grounded citations in a one-off `/ask` run.

## 4. Gold-source retrieval ranks for the Week 2 v2 `retrieval_miss` refusals (Q1, Q4, Q7, Q15)

`/search` with `top_k=50` (API max). Rank = first position of the gold chunk.
- **Answer chunk:** the chunk whose text actually states the answer.
- **Page:** any chunk of the gold doc page.

| Q | Question | Answer chunk (gold) | hybrid | BM25 | vector | Best rank of gold page (hybrid) |
|---|---|---|---|---|---|---|
| 1 | Declare a path parameter | `c3fbe41174be-0000` (path-params.md, "same syntax used by Python format strings") | **>50** | >50 | >50 | 1 (`-0017`, "path parameters containing paths", wrong section) |
| 4 | Custom HTTP error | `b05842751c15-0001` (handling-errors.md, "To return HTTP responses with errors … use `HTTPException`") | **>50** | >50 | >50 | 4 (`-0004`); `-0003` "raise an HTTPException" at 34 |
| 7 | Add CORS | `0e60b3828d2a-0004` (cors.md, "Use `CORSMiddleware`") | **1** | 3 | 1 | 1 |
| 15 | Forbid extra fields | `8555df178cda-0037` (validation_errors.md `extra_forbidden`, only chunk with `extra='forbid'`) | **3** | 5 | 4 | — |

**Finding (correction to Week 2).** The Week 2 label `ungrounded_answer:retrieval_miss` is only accurate for **Q1 and Q4**: their answer chunks are not in the top 50. For **Q7 and Q15** the gold chunk was retrieved (ranks 1 and 3), so those refusals are **verifier false negatives**.
- Q7: the model's sentence ("by importing `CORSMiddleware` and configuring it …") didn't reach 50 % lexical overlap with its cited chunk.
- Q15: the answer's key terms (`model_config`, "option", "definition") aren't in the cited chunk. "extra" and "forbid" don't count because they are question words.

The reason name overstates retrieval as the cause. A more accurate name would be `unsupported_citation`. I didn't rename it or change thresholds in this run (out of scope; would shift the stage).

Contributing factor in the corpus: FastAPI's tutorial code lives in `docs_src/*.py` and appears in the ingested Markdown only as `{* ../../docs_src/... *}` include markers. So chunks such as path-params `-0000` and handling-errors `-0002` carry little text, and their actual code (e.g. `raise HTTPException(...)`) is missing from the index.


## Suggested next steps (not done)
1. Rename `retrieval_miss` → `unsupported_citation` (see section 4: two of the four Week 2 cases were verifier false negatives, not retrieval misses).
2. Log blocked drafts (refused-after-generation) server-side, for analysis.
3. Move converted Q5/Q6 into the answerable set. Decide team policy for borderline Q3/Q12/Q13 ("docs don't specify" answers).
4. Inline the `docs_src` includes during ingest (Q1/Q4 code examples are not indexed).
