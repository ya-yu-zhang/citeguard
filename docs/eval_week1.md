# CiteGuard Week 1 Retrieval Eval

Corpus queries: **30** hand-written (FastAPI / Pydantic / LangGraph).
Metric: Recall@5 = fraction of queries where any expected `doc_id` appears in top-5.
MRR = mean reciprocal rank of first relevant doc.

| Method | Recall@5 | MRR | p50 latency (s) | p95 latency (s) |
|---|---:|---:|---:|---:|
| `bm25` | 0.733 | 0.524 | 0.008 | 0.008 |
| `vector` | 0.867 | 0.676 | — | 0.040 |
| `hybrid` | 0.900 | 0.723 | — | 0.049 |
| `hybrid+rerank` | 0.900 | 0.637 | — | 2.853 |

## HTTP `/search` p95 (uvicorn, CPU, embedded Qdrant, 20 queries, warm)

| Method | p50 (s) | p95 (s) | mean (s) |
|---|---:|---:|---:|
| `hybrid` | 0.035 | 0.049 | 0.037 |
| `hybrid+rerank` | 2.584 | 2.676 | 2.576 |

## Notes

- Numbers above are from an actual run on this box (CPU, local `BAAI/bge-small-en-v1.5` + `BAAI/bge-reranker-base` via fastembed, Qdrant embedded). No paid APIs.
- Hybrid (RRF) beats BM25-only and vector-only on Recall@5 and MRR.
- Cross-encoder rerank did **not** improve Recall@5 on this 30-query set and slightly lowered MRR; it also misses the Week-1 p95&lt;1.5s target on CPU (~2.7s). Default API method remains `hybrid+rerank` for quality demos; use `method=hybrid` when latency matters.
- Expected doc paths were chosen from the pinned corpus; see `docs/eval_week1.json` for per-query hits.

Raw details: `docs/eval_week1.json`
