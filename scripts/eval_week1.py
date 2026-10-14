#!/usr/bin/env python3
"""Week-1 retrieval eval: Recall@5 and MRR for bm25 / vector / hybrid / hybrid+rerank."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from citeguard.api.app import build_retriever

# Hand-written queries with expected doc_id prefixes (source:relpath).
# Expected match = any retrieved hit whose doc_id starts with or equals expected.
EVAL_QUERIES: list[dict] = [
    # FastAPI
    {"q": "How do I enable CORS in FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/cors.md"]},
    {"q": "How to define a path parameter in FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/path-params.md"]},
    {"q": "How to use query parameters in FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/query-params.md"]},
    {"q": "How do I upload a file with FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/request-files.md"]},
    {"q": "How to return a JSON response in FastAPI?", "expect": ["fastapi:docs/en/docs/advanced/response-directly.md", "fastapi:docs/en/docs/tutorial/response-model.md"]},
    {"q": "How to add middleware in FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/middleware.md", "fastapi:docs/en/docs/advanced/middleware.md"]},
    {"q": "How does FastAPI dependency injection work?", "expect": ["fastapi:docs/en/docs/tutorial/dependencies/index.md", "fastapi:docs/en/docs/tutorial/dependencies/classes-as-dependencies.md"]},
    {"q": "How to use background tasks in FastAPI?", "expect": ["fastapi:docs/en/docs/tutorial/background-tasks.md"]},
    {"q": "How to handle WebSockets in FastAPI?", "expect": ["fastapi:docs/en/docs/advanced/websockets.md"]},
    {"q": "How to secure FastAPI with OAuth2 password flow?", "expect": ["fastapi:docs/en/docs/tutorial/security/oauth2-jwt.md", "fastapi:docs/en/docs/tutorial/security/simple-oauth2.md"]},
    {"q": "How to configure CORSMiddleware allow_origins?", "expect": ["fastapi:docs/en/docs/tutorial/cors.md"]},
    {"q": "What is FastAPI lifespan event handler?", "expect": ["fastapi:docs/en/docs/advanced/events.md"]},
    # Pydantic
    {"q": "How do I define a Pydantic BaseModel?", "expect": ["pydantic:docs/concepts/models.md"]},
    {"q": "How does Pydantic field validation work?", "expect": ["pydantic:docs/concepts/validators.md", "pydantic:docs/concepts/fields.md"]},
    {"q": "What are Pydantic computed fields?", "expect": ["pydantic:docs/concepts/fields.md"]},
    {"q": "How to use Pydantic Settings for environment variables?", "expect": ["pydantic:docs/concepts/pydantic_settings.md"]},
    {"q": "How does Pydantic serialize models to JSON?", "expect": ["pydantic:docs/concepts/serialization.md"]},
    {"q": "What is TypeAdapter in Pydantic v2?", "expect": ["pydantic:docs/concepts/type_adapter.md"]},
    {"q": "How to configure model_config in Pydantic v2?", "expect": ["pydantic:docs/concepts/config.md"]},
    {"q": "How do Pydantic discriminated unions work?", "expect": ["pydantic:docs/concepts/unions.md"]},
    # LangGraph
    {"q": "What is a LangGraph StateGraph?", "expect": ["langgraph:src/oss/langgraph/graph-api.mdx", "langgraph:src/oss/langgraph/workflows-agents.mdx"]},
    {"q": "How do I add memory to a LangGraph agent?", "expect": ["langgraph:src/oss/langgraph/add-memory.mdx", "langgraph:src/oss/langgraph/persistence.mdx"]},
    {"q": "How does LangGraph streaming work?", "expect": ["langgraph:src/oss/langgraph/streaming.mdx"]},
    {"q": "How to use human-in-the-loop interrupts in LangGraph?", "expect": ["langgraph:src/oss/langgraph/interrupts.mdx"]},
    {"q": "What is the LangGraph functional API?", "expect": ["langgraph:src/oss/langgraph/functional-api.mdx"]},
    {"q": "How to build agentic RAG with LangGraph?", "expect": ["langgraph:src/oss/langgraph/agentic-rag.mdx"]},
    {"q": "How does LangGraph persistence / checkpointer work?", "expect": ["langgraph:src/oss/langgraph/persistence.mdx", "langgraph:src/oss/langgraph/fault-tolerance.mdx"]},
    {"q": "How to use tools and tool calling in LangGraph?", "expect": ["langgraph:src/oss/langgraph/workflows-agents.mdx", "langgraph:src/oss/langgraph/sql-agent.mdx"]},
    {"q": "What is LangGraph Command for graph control flow?", "expect": ["langgraph:src/oss/langgraph/graph-api.mdx", "langgraph:src/oss/langgraph/use-graph-api.mdx"]},
    {"q": "How to deploy a LangGraph application?", "expect": ["langgraph:src/oss/langgraph/application-structure.mdx", "langgraph:src/oss/langgraph/deploy.mdx"]},
]


def _hit(doc_ids: list[str], expect: list[str]) -> bool:
    for e in expect:
        for d in doc_ids:
            if d == e or d.startswith(e) or e in d:
                return True
    return False


def _mrr(doc_ids: list[str], expect: list[str]) -> float:
    for i, d in enumerate(doc_ids, start=1):
        for e in expect:
            if d == e or d.startswith(e) or e in d:
                return 1.0 / i
    return 0.0


def evaluate(method: str, retriever, top_k: int = 5) -> dict:
    recalls = []
    mrrs = []
    latencies = []
    details = []
    for item in EVAL_QUERIES:
        t0 = time.perf_counter()
        hits = retriever.search(item["q"], top_k=top_k, method=method)
        latencies.append(time.perf_counter() - t0)
        doc_ids = [h.doc_id for h in hits]
        r = 1.0 if _hit(doc_ids, item["expect"]) else 0.0
        m = _mrr(doc_ids, item["expect"])
        recalls.append(r)
        mrrs.append(m)
        details.append({"q": item["q"], "hit": bool(r), "mrr": m, "got": doc_ids[:3]})
    return {
        "method": method,
        "n": len(EVAL_QUERIES),
        "recall_at_5": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "mean_latency_s": statistics.mean(latencies),
        "details": details,
    }


def main() -> None:
    out_path = ROOT / "docs" / "eval_week1.md"
    json_path = ROOT / "docs" / "eval_week1.json"
    print("loading retriever…")
    retriever = build_retriever()
    methods = ["bm25", "vector", "hybrid", "hybrid+rerank"]
    results = []
    for m in methods:
        print(f"evaluating {m}…")
        results.append(evaluate(m, retriever))
        r = results[-1]
        print(f"  R@5={r['recall_at_5']:.3f} MRR={r['mrr']:.3f} p95={r['p95_latency_s']:.3f}s")

    lines = [
        "# CiteGuard Week 1 Retrieval Eval",
        "",
        f"Corpus queries: **{len(EVAL_QUERIES)}** hand-written (FastAPI / Pydantic / LangGraph).",
        "Metric: Recall@5 = fraction of queries where any expected `doc_id` appears in top-5.",
        "MRR = mean reciprocal rank of first relevant doc.",
        "",
        "| Method | Recall@5 | MRR | p50 latency (s) | p95 latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| `{r['method']}` | {r['recall_at_5']:.3f} | {r['mrr']:.3f} | "
            f"{r['p50_latency_s']:.3f} | {r['p95_latency_s']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Numbers above are from an actual run on this box (CPU, local `bge-small-en-v1.5` + `bge-reranker-base` via fastembed, Qdrant embedded).",
        "- Expected doc paths were chosen from the pinned corpus; some queries may miss if headings/chunk boundaries differ — see JSON details.",
        "",
        f"Raw details: `docs/eval_week1.json`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # strip details for size? keep them
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
