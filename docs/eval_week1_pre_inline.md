# CiteGuard Week 1 Retrieval Eval

Corpus queries: **30** hand-written (FastAPI / Pydantic / LangGraph).
Metric: Recall@5 = fraction of queries where any expected `doc_id` appears in top-5.
MRR = mean reciprocal rank of first relevant doc.

| Method | Recall@5 | MRR | p50 latency (s) | p95 latency (s) |
|---|---:|---:|---:|---:|
| `bm25` | 0.733 | 0.524 | 0.008 | 0.011 |
| `vector` | 0.867 | 0.676 | 0.040 | 0.049 |
| `hybrid` | 0.900 | 0.723 | 0.054 | 0.068 |
| `hybrid+rerank` | 0.933 | 0.669 | 0.198 | 0.227 |

## Notes

- Numbers above are from an actual run on this box (CPU, local `bge-small-en-v1.5` + `bge-reranker-base` via fastembed, Qdrant embedded).
- Expected doc paths were chosen from the pinned corpus; some queries may miss if headings/chunk boundaries differ — see JSON details.

Raw details: `docs/eval_week1.json`
