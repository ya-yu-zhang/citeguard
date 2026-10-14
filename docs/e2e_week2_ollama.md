# Week 2 — live end-to-end `/ask` with a free local LLM (Ollama, CPU)

Run date: 2026-10-14. Box: 8 vCPU, 16 GB RAM (~4–5 GB free while the API, Chrome etc. run), no GPU.
Ollama 0.40.4, Qwen2.5 Instruct GGUF Q4_K_M imported from Hugging Face (`ollama pull` was blocked by the box's
network proxy, so `ollama create` from the official Qwen GGUF was used; same weights).
Retrieval: default hybrid (BM25 + bge-small, RRF), top-5, evidence gate unchanged (RRF top < 0.02 or top-3 < 200 chars → refuse).
Temperature 0, `max_tokens` 400, up to 1200 chars per chunk sent to the LLM.

Reproduce: `ollama serve` + `LLM_MODEL=qwen2.5:3b-instruct uvicorn citeguard.api.app:app --app-dir src`, then
`python scripts/e2e_ask.py --out docs/e2e_week2_ollama.json`.
Sets: 20 answerable questions (`docs/answerable_samples_week2.json`, written against the FastAPI / Pydantic / LangGraph docs in the corpus) + the 20 refuse samples (`docs/refuse_samples_week2.json`). Each question is sent once to `POST /ask` and once to `POST /ask/stream`.
Raw per-question results: `docs/e2e_week2_ollama.json` (3B, primary) and `docs/e2e_week2_ollama_1.5b.json`.

## Metrics

| Metric | 3B `/ask` | 3B `/ask/stream` | 1.5B `/ask` | 1.5B `/ask/stream` |
|---|---|---|---|---|
| Refusal accuracy (20 refuse) | **1.00** | 0.95 | 1.00 | 1.00 |
| Answer rate (20 answerable) | **0.55** (11/20) | 0.45 | 0.15 (3/20) | 0.15 |
| Citation ID validity¹ | 1.00 | 1.00 | 1.00 | 1.00 |
| Sentence citation validity² | **0.76** | 0.65 | 0.67 | 1.00 (n=3) |
| Answers with every sentence validly cited³ | 0.45 | 0.56 | 0.00 | 0.00 |
| Latency p50 / p95, all 40 questions | 6.97 s / 12.9 s | 1.66 s / 7.46 s⁴ | 4.23 s / 13.2 s | 0.60 s / 10.4 s |
| Latency p50 / p95, answered only | 8.16 s / 13.8 s | 3.19 s / 8.38 s⁴ | 4.56 s / 5.96 s | 1.94 s / 2.75 s |
| Time to first token p50 / p95 | — | 0.25 s / 0.30 s | — | 0.21 s / 0.25 s |

1. Every `[n]` in the answer is in range of the returned `citations`, and every returned `chunk_id` was in the top-5 that `/search` retrieved for the same question.
2. Programmatic, per sentence. A sentence is valid only if (a) each `[n]` maps to a returned citation, (b) that chunk was retrieved, and (c) the cited chunk text contains ≥ 50 % of the sentence's key terms (words of 4 or more chars, minus stopwords and words copied from the question). It is a lexical proxy, so it catches "cited the wrong chunk" and invented facts that use new vocabulary. It does **not** catch a sentence that reuses the chunk's words but gets the meaning wrong. Implemented in `scripts/e2e_ask.py::sentence_support`, and comparable across providers (rerun the script with OpenRouter).
3. Answered questions where all cited sentences are valid and there are no uncited sentences.
4. Each stream call ran right after the `/ask` call for the same question, so Ollama's prompt cache was warm. That is why stream totals are lower; compare TTFT, not totals, for streaming UX.

How the 3B `/ask` refusals break down: on the refuse set, 13/20 were stopped by the retrieval gate (no LLM call), 4 by the model replying `INSUFFICIENT_EVIDENCE`, 2 by the new lexical grounding check, and 1 by missing markers. On the answerable set, the 9 refusals were 4 missing markers, 4 low grounding, and 1 where the model declared insufficient evidence.

Before prompt tuning (first 1.5B run, old prompt, no grounding check): refusal accuracy **0.75** (5 refuse questions were "answered", e.g. *"Your coworker Alice said 'What's your name?' in her private Slack DM yesterday [1]"*). Answer rate 0.30.

## Changes made for small models
- Provider-agnostic OpenAI-compatible client: `LLM_BASE_URL` (default `http://localhost:11434/v1`), `LLM_MODEL` (empty → offline stub), optional `LLM_API_KEY` (Bearer header sent only if set), `LLM_MAX_TOKENS`, `LLM_TIMEOUT_S`. OpenRouter is documented in `.env.example` and README, but was not called.
- The LLM now sees up to 1200 chars of each chunk; before, it saw only the 240-char response excerpt.
- Prompt: short-answer rules plus two few-shot examples (one cited answer, one `INSUFFICIENT_EVIDENCE`).
- `[1, 2]` is normalized to `[1][2]`.
- Self-check now also rejects "not provided in the given evidence"-style non-answers, and (live LLM only) answers whose content words are < 50 % supported by the cited chunks (`low_grounding`). Neither of these triggers a regeneration (at T=0 it would repeat itself). The stream path now uses the same no-regenerate rules as the graph.

## Notable failures (3B)
- **Correct but uncited / wrongly cited.** Q4 (HTTPException) and Q5 (Depends): the facts are right, but retrieval's top-5 didn't include the "Handling Errors" page. The model answered from its own knowledge and cited a status-code chunk instead. Answer-level self-check passed, while the sentence metric flags these (key-term share 0.00 / 0.20). This is a retrieval-recall issue, not a generation issue.
- **Factually wrong but passes lexical checks.** Q9: "declare a parameter of type `response_model` in your path operation function [1]". In fact `response_model` is a decorator argument. All checks passed because the words are in the chunk. A semantic (NLI/LLM-judge) check is needed (week 3, RAGAS faithfulness).
- **Prompt echo in streaming.** Refuse #7 (stream only): "Write a romantic poem about two toasters…" retrieved a LangGraph example that contains that literal prompt string. The model "answered" with the code snippet and cited [1], and grounding passed because the code was copied. This is the only false answer in 80 calls.
- **Missing markers.** Still the main cause of over-refusal (4/20 at 3B; 13/20 at 1.5B before tuning, 11/20 after). The model sometimes writes `[n]` literally.
- **/ask vs /ask/stream disagree** on 2/20 answerable questions (Q5, Q7). Even at T=0, Ollama's streamed and non-streamed generations can diverge slightly.

## Model choice
**Primary: `qwen2.5:3b-instruct` (Q4_K_M, ~2.4 GB resident).** It fits the box's free RAM next to the API process. Compared with 1.5B, it more than triples the answer rate at the same refusal accuracy, for about 2–4 s more latency. 1.5B is kept as a fallback for low-RAM machines, but its answer rate (15 %) is not demo-worthy. Both are free/local. A 7B+ model via OpenRouter is the next comparison, using the same script and metrics.
