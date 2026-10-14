# Week 3 — hard should-refuse set (项目小助手1's 20 questions)

> Note (2026-10-14): refusal reason `retrieval_miss` was renamed to `unsupported_citation`; this doc was updated to the new name. Raw JSON from runs before the rename still contains `retrieval_miss`.

> **Superseded:** this run used the wrong list (项目小助手1’s Week 2 应拒样例, not the Week 3 hard list). Kept for reference; the current results are in `refuse_hard_week3.md`. File references below to `refuse_hard_week3*.json` now correspond to `refuse_hard_week3_oldlist*.json`.

Setup: the same stack as Week 2 v2 (`qwen2.5:3b-instruct` on Ollama, CPU; hybrid top-5; evidence gate RRF ≥ 0.02 and top-3 ≥ 200 chars; per-sentence support threshold 0.5; task filter). **No thresholds, prompts or questions were changed for this run.** Run date: 2026-10-14.

Source: 项目小助手1's group-chat list of 20 应拒样例 (Week 2), translated faithfully into English (the corpus is English). The Chinese originals and annotations are kept in `docs/refuse_hard_week3_oldlist.json` (`question_zh`). Raw `/ask` and `/ask/stream` outputs: `docs/refuse_hard_week3_oldlist_results.json`. Runner: `scripts/run_hard_refuse.py`.

## 1. Screening: do the docs actually answer it?

For each question: `/search` (hybrid, top 5), read the chunks, plus a keyword search over all 4,100 chunks for the key term.

| # | Question (EN) | 原文 | Screening | Evidence / note |
|---|---|---|---|---|
| 1 | What are CiteGuard's own training hyperparameters? | CiteGuard 自己的训练超参是多少？（元问题，库里没有） | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 2 | How does FastAPI natively support WebSocket cluster leader election? | FastAPI 怎么原生支持 WebSocket 集群选举？ | keep (refuse) | WebSockets docs are retrieved (84ca5edcf8ca-*, 218e2a04fe08-0000) but say nothing about cluster/leader election. |
| 3 | What is Pydantic v1's `orm_mode` called in v2? | Pydantic v1 的 `orm_mode` 在 v2 里还叫什么？（诱导过时说法，应拒或明确说需查迁移指南且无依据则拒） | **answerable → convert** | Docs answer it: pydantic docs/migration.md 'Changes to config' lists `orm_mode` → `from_attributes` (chunk c28fef63e4cf-0019, NOT in hybrid top-5; top-5 has the adjacent chunk c28fef63e4cf-0018). Also docs/concepts/models.md 13b45477775a-0019. |
| 4 | How does LangGraph natively render UI on iOS? | LangGraph 怎么在 iOS 上原生渲染 UI？ | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 5 | Write complete example code for a Kubernetes operator using FastAPI. | 用 FastAPI 写一个完整的 Kubernetes operator 示例代码 | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 6 | How does Pydantic connect directly to Postgres to run ORM migrations? | Pydantic 如何直接连接 Postgres 做 ORM 迁移？ | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 7 | What stock trading strategies does LangGraph officially recommend? | LangGraph 官方推荐的股票交易策略有哪些？ | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 8 | What is the default prefix of `APIRouter` in FastAPI version 0.50? | FastAPI 0.50 版本的 `APIRouter` 默认前缀是什么？（过旧版本细节） | keep (refuse) | Release notes for 0.32/0.62 and bigger-applications docs retrieved; none states a 0.50-specific APIRouter default prefix. |
| 9 | What are the book borrowing rules of the NUIST (Nanjing University of Information Science and Technology) library? | 南信大图书馆的借书规则是什么？（领域外） | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 10 | Compare the routing performance benchmarks of Django and Rails. | 比较 Django 和 Rails 的路由性能基准（库外） | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 11 | How do I call LangGraph's undocumented internal debug API? | LangGraph 如何调用未公开的内部 debug API？ | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 12 | What is the price of a Pydantic commercial license? | Pydantic 的商业授权价格是多少？ | keep (refuse) | Enterprise support page retrieved (6b1295503ceb-0003/-0004): license is open source, support via contact form; no price stated. |
| 13 | What features will the FastAPI author add in the next version? | FastAPI 作者下个版本会加什么功能？（预测类） | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 14 | Use FastAPI to automatically generate all the CRUD for this SQL. | 把这段 SQL 用 FastAPI 自动生成全部 CRUD（无具体 schema，证据不足） | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 15 | Does LangGraph support quantum computing backends? | LangGraph 是否支持量子计算后端？ | keep (refuse) | Only hit for 'quantum' is a sample input string ('quantum computing' topic) in cfdf5cebb128-0005; no backend support statement. |
| 16 | How do I use the Bitcoin wallet validation described in the Pydantic docs? | Pydantic 文档里写的比特币钱包校验怎么用？（不存在的功能） | keep (refuse) | No 'bitcoin' anywhere in corpus. |
| 17 | What are the installation steps for FastAPI on Windows XP? | FastAPI 在 Windows XP 上的安装步骤 | keep (refuse) | Install docs retrieved (35fe24971a67-0013, 473a8b43ed74-0003) say Python 3.10–3.14 via uv; nothing about Windows XP. |
| 18 | Which raised more funding, LangGraph or AutoGPT? | LangGraph 和 AutoGPT 谁融资更多？ | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 19 | Based on the docs, summarize your company's internal compliance process. | 根据文档总结你们公司的内部合规流程 | keep (refuse) | No retrieved top-5 chunk (and no corpus text found by keyword search) answers this. |
| 20 | Use the docs to prove that FastAPI is 100 times faster than Node. | 用文档证明 FastAPI 比 Node 快 100 倍（文档无此结论则拒） | keep (refuse) | Benchmarks page retrieved (6cf2d306e9ae-0001) compares Uvicorn/Starlette/FastAPI; no 100x-vs-Node claim. |

**Result: 19 kept, 1 converted.** Q3 is answerable from the corpus: Pydantic `migration.md` lists `orm_mode` → `from_attributes` (chunk `c28fef63e4cf-0019`). The converted question is *"What is Pydantic v1's `orm_mode` config option called in Pydantic v2?"* (expect `from_attributes`). It should move to the answerable set, so it is excluded from the refusal numbers below. One informational `/ask` run of the converted question **refused** it (`ungrounded_answer:no_supported_sentences`): hybrid top-5 has the adjacent chunk `-0018` but not `-0019`, so it is a retrieval miss.

## 2. Results on the 19 kept questions

`/ask` and `/ask/stream` gave identical decisions and reasons on all 19 questions.

| Metric | `/ask` | `/ask/stream` |
|---|---|---|
| Refused | **19 / 19** | **19 / 19** |
| False answers | 0 | 0 |
| `no_evidence` | 17 | 17 |
| ↳ retrieval gate (`top_score_below_threshold`) | 3 (Q1, Q15, Q18) | 3 |
| ↳ model said `INSUFFICIENT_EVIDENCE` | 14 | 14 |
| ↳ `out_of_scope_task` | 0 | 0 |
| `ungrounded_answer` (`no_supported_sentences`) | 2 (Q7, Q9) | 2 |

Unlike the Week 2 refuse set, where 13 of 20 were stopped by the retrieval gate, 16 of these 19 questions got past the gate and **reached the LLM**. That makes this a much better test of the generation and verification stages. Here, refusals relied mainly on the model saying `INSUFFICIENT_EVIDENCE` (14), with the citation verifier as a backstop (2).
Q5 ("Write complete example code for a Kubernetes operator…") did *not* trip the creative-task filter, as intended. The model refused it.

Per question:

| # | `/ask` refusal_reason | `/ask/stream` refusal_reason |
|---|---|---|
| 1 | `no_evidence:top_score_below_threshold:0.0164<0.02` | `no_evidence:top_score_below_threshold:0.0164<0.02` |
| 2 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 4 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 5 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 6 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 7 | `ungrounded_answer:no_supported_sentences` | `ungrounded_answer:no_supported_sentences` |
| 8 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 9 | `ungrounded_answer:no_supported_sentences` | `ungrounded_answer:no_supported_sentences` |
| 10 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 11 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 12 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 13 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 14 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 15 | `no_evidence:top_score_below_threshold:0.0164<0.02` | `no_evidence:top_score_below_threshold:0.0164<0.02` |
| 16 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 17 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 18 | `no_evidence:top_score_below_threshold:0.0164<0.02` | `no_evidence:top_score_below_threshold:0.0164<0.02` |
| 19 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |
| 20 | `no_evidence:model_declared_insufficient_evidence` | `no_evidence:model_declared_insufficient_evidence` |

## 3. Gold-source retrieval ranks for the Week 2 v2 `unsupported_citation` (then `retrieval_miss`) refusals (Q1, Q4, Q7, Q15)

`/search` with `top_k=50` (API max). Rank = first position of the gold chunk.
- **Answer chunk:** the chunk whose text actually states the answer.
- **Page:** any chunk of the gold doc page.

| Q | Question | Answer chunk (gold) | hybrid | BM25 | vector | Best rank of gold page (hybrid) |
|---|---|---|---|---|---|---|
| 1 | Declare a path parameter | `c3fbe41174be-0000` (path-params.md, "same syntax used by Python format strings") | **>50** | >50 | >50 | 1 (`-0017`, "path parameters containing paths", wrong section) |
| 4 | Custom HTTP error | `b05842751c15-0001` (handling-errors.md, "To return HTTP responses with errors … use `HTTPException`") | **>50** | >50 | >50 | 4 (`-0004`); `-0003` "raise an HTTPException" at 34 |
| 7 | Add CORS | `0e60b3828d2a-0004` (cors.md, "Use `CORSMiddleware`") | **1** | 3 | 1 | 1 |
| 15 | Forbid extra fields | `8555df178cda-0037` (validation_errors.md `extra_forbidden`, only chunk with `extra='forbid'`) | **3** | 5 | 4 | — |

**Finding (correction to Week 2).** The Week 2 label (then named `retrieval_miss`) is only accurate for **Q1 and Q4**: their answer chunks are not in the top 50. For **Q7 and Q15** the gold chunk was retrieved (ranks 1 and 3), so those refusals are **verifier false negatives**.
- Q7: the model's sentence ("by importing `CORSMiddleware` and configuring it …") didn't reach 50 % lexical overlap with its cited chunk.
- Q15: the answer's key terms (`model_config`, "option", "definition") aren't in the cited chunk. "extra" and "forbid" don't count because they are question words.

The old name (`retrieval_miss`) overstated retrieval as the cause; it has since been renamed to `unsupported_citation`.

Contributing factor in the corpus: FastAPI's tutorial code lives in `docs_src/*.py` and appears in the ingested Markdown only as `{* ../../docs_src/... *}` include markers. So chunks such as path-params `-0000` and handling-errors `-0002` carry little text, and their actual code (e.g. `raise HTTPException(...)`) is missing from the index.

## Suggested next steps (not done)
1. ~~Rename `retrieval_miss` → `unsupported_citation`~~ (done), and add a retrieval-diagnostic field (did the top-5 contain the gold page?).
2. Inline the `docs_src` includes during ingest, so the code examples become searchable and citable (Q1, Q4).
3. Move converted Q3 to the answerable set. The hard refuse set stays at 19.
4. To check Q7/Q15-style false negatives, use a semantic support check (NLI or LLM judge) rather than lowering the lexical threshold.
