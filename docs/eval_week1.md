# CiteGuard Week 1 Retrieval Eval

Corpus queries: **30** hand-written (FastAPI / Pydantic / LangGraph).
Metric: Recall@5 = fraction of queries where any expected `doc_id` appears in top-5.
MRR = mean reciprocal rank of first relevant doc.

| Method | Recall@5 | MRR | p50 latency (s) | p95 latency (s) |
|---|---:|---:|---:|---:|
| `bm25` | 0.733 | 0.575 | 0.006 | 0.007 |
| `vector` | 0.867 | 0.687 | 0.032 | 0.040 |
| `hybrid` | 0.967 | 0.761 | 0.048 | 0.056 |
| `hybrid+rerank` | 0.933 | 0.721 | 0.193 | 0.224 |

## Notes

- Numbers above are from an actual run on this box (CPU, local `bge-small-en-v1.5` + `bge-reranker-base` via fastembed, Qdrant embedded).
- Expected doc paths were chosen from the pinned corpus; some queries may miss if headings/chunk boundaries differ — see JSON details.

Raw details: `docs/eval_week1.json`
