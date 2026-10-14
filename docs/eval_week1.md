# CiteGuard Week 1 Retrieval Eval

Corpus queries: **30** hand-written (FastAPI / Pydantic / LangGraph).
Metric: Recall@5 = fraction of queries where any expected `doc_id` appears in top-5.
MRR = mean reciprocal rank of first relevant doc.

**Rerun after gap fixes** (unique `chunk_id`, default method=`hybrid`, reranker=`Xenova/ms-marco-MiniLM-L-6-v2`, RRF top-10 → cross-encoder, passages truncated to 512 chars).

| Method | Recall@5 | MRR | p50 latency (s) | p95 latency (s) |
|---|---:|---:|---:|---:|
| `bm25` | 0.733 | 0.524 | 0.007 | 0.007 |
| `vector` | 0.867 | 0.676 | — | 0.031 |
| `hybrid` | 0.900 | **0.723** | — | 0.047 |
| `hybrid+rerank` | **0.933** | 0.669 | — | 0.216 |

## HTTP `/search` p95 (uvicorn, CPU, embedded Qdrant, 20 queries, warm)

| Method | p50 (s) | p95 (s) | mean (s) |
|---|---:|---:|---:|
| `hybrid` (API default) | 0.026 | **0.036** | 0.027 |
| `hybrid+rerank` (opt-in) | 0.193 | **0.214** | 0.194 |

Both methods meet the Week-1 **p95 &lt; 1.5s** target on this CPU box.

## Comparison vs previous run (bge-reranker-base, default hybrid+rerank)

| | Prior hybrid+rerank | Now hybrid+rerank (MiniLM-L-6) |
|---|---:|---:|
| Recall@5 | 0.900 | **0.933** |
| MRR | 0.637 | 0.669 |
| HTTP p95 | ~2.68s (missed target) | **0.214s** |

## Honest takeaways

- **Default API method is `hybrid`** — best MRR on this set, and fastest.
- **`hybrid+rerank` is opt-in** (`method=hybrid+rerank` or `hybrid_rerank`). With MiniLM-L-6 it **beats hybrid on Recall@5** (0.933 vs 0.900) but **still trails on MRR** (0.669 vs 0.723). Keep both; do not treat rerank as universally better.
- Switched from `BAAI/bge-reranker-base` (~2.7s p95 on CPU) to `Xenova/ms-marco-MiniLM-L-6-v2` so the opt-in path also clears 1.5s. Documented in config.
- Numbers are from actual runs on this box; no paid APIs. Per-query details: `docs/eval_week1.json` (local artifact, gitignored).

## chunk_id collisions

Previously ~24 collisions (4100 written → 4076 loaded) because ids hashed `doc_id::title::i::text[:64]`.  
Now: `sha1(doc_id)[:12]-{index:04d}` per doc — **4100 unique / 4100 loaded**.
