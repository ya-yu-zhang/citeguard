"""LangGraph: retrieve → evidence_judge → (refuse | generate) → self_check."""
from __future__ import annotations

from typing import Any, Iterator, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from citeguard.ask.evidence import evidence_gate, hits_to_citations
from citeguard.ask.llm import generate_answer, llm_configured, normalize_citations, self_check_answer, stream_generate
from citeguard.config import ASK_TOP_K, LLM_CONTEXT_CHARS, get_settings
from citeguard.retrieval.hybrid import HybridRetriever


def _contexts(hits: list[dict[str, Any]]) -> list[str]:
    return [str(h.get("text") or "")[:LLM_CONTEXT_CHARS] for h in hits]


def _check(answer: str, citations: list, question: str, hits: list) -> tuple[bool, str | None]:
    """Citation self-check; lexical grounding only for live LLMs (stub is extractive)."""
    if not llm_configured():
        return self_check_answer(answer, citations)
    return self_check_answer(answer, citations, question=question, contexts=_contexts(hits))


_NO_REGEN = {"model_declared_insufficient_evidence", "empty_answer", "no_citations"}
# low_grounding:* is also final (prefix-matched below): regenerating at T=0 gives the same text


class AskState(TypedDict, total=False):
    question: str
    method: str
    hits: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    answer: str
    refused: bool
    refusal_reason: str | None
    regenerate_count: int
    evidence_ok: bool
    self_check_ok: bool
    self_check_reason: str | None


def _ask_method() -> str:
    settings = get_settings()
    return "hybrid+rerank" if settings.ask_use_rerank else "hybrid"


def build_ask_graph(retriever: HybridRetriever):
    def retrieve(state: AskState) -> dict[str, Any]:
        method = state.get("method") or _ask_method()
        hits = retriever.search(state["question"], top_k=ASK_TOP_K, method=method)
        return {
            "method": method,
            "hits": [h.to_dict() for h in hits],
            "regenerate_count": state.get("regenerate_count") or 0,
        }

    def evidence_judge(state: AskState) -> dict[str, Any]:
        decision = evidence_gate(state.get("hits") or [])
        if not decision.ok:
            return {
                "evidence_ok": False,
                "refused": True,
                "refusal_reason": decision.reason,
                "answer": "",
                "citations": [],
            }
        citations = hits_to_citations(state["hits"])
        return {
            "evidence_ok": True,
            "refused": False,
            "refusal_reason": None,
            "citations": citations,
        }

    def refuse(state: AskState) -> dict[str, Any]:
        reason = state.get("refusal_reason") or "insufficient_evidence"
        return {
            "refused": True,
            "refusal_reason": reason,
            "answer": "",
            "citations": state.get("citations") or [],
        }

    def generate(state: AskState) -> dict[str, Any]:
        citations = state.get("citations") or []
        answer = generate_answer(state["question"], citations, _contexts(state.get("hits") or []))
        return {"answer": answer}

    def self_check(state: AskState) -> dict[str, Any]:
        ok, reason = _check(
            state.get("answer") or "", state.get("citations") or [], state["question"], state.get("hits") or []
        )
        if ok:
            return {
                "self_check_ok": True,
                "self_check_reason": None,
                "refused": False,
                "refusal_reason": None,
            }
        # regenerate once, then refuse
        count = int(state.get("regenerate_count") or 0)
        if count < 1 and reason not in _NO_REGEN and not str(reason).startswith("low_grounding"):
            return {
                "self_check_ok": False,
                "self_check_reason": reason,
                "regenerate_count": count + 1,
                "answer": "",  # clear for regenerate
            }
        return {
            "self_check_ok": False,
            "self_check_reason": reason,
            "refused": True,
            "refusal_reason": f"self_check_failed:{reason}",
            "answer": "",
        }

    def route_after_judge(state: AskState) -> Literal["refuse", "generate"]:
        return "refuse" if state.get("refused") or not state.get("evidence_ok") else "generate"

    def route_after_check(state: AskState) -> Literal["refuse", "generate", "__end__"]:
        if state.get("refused"):
            return "refuse"
        if state.get("self_check_ok"):
            return "__end__"
        # failed check but regenerate_count bumped and answer cleared → generate again
        if (state.get("regenerate_count") or 0) == 1 and not (state.get("answer") or "").strip():
            return "generate"
        return "refuse"

    g = StateGraph(AskState)
    g.add_node("retrieve", retrieve)
    g.add_node("evidence_judge", evidence_judge)
    g.add_node("refuse", refuse)
    g.add_node("generate", generate)
    g.add_node("self_check", self_check)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "evidence_judge")
    g.add_conditional_edges("evidence_judge", route_after_judge, {"refuse": "refuse", "generate": "generate"})
    g.add_edge("generate", "self_check")
    g.add_conditional_edges(
        "self_check",
        route_after_check,
        {"refuse": "refuse", "generate": "generate", "__end__": END},
    )
    g.add_edge("refuse", END)
    return g.compile()


def run_ask(retriever: HybridRetriever, question: str, *, method: str | None = None) -> dict[str, Any]:
    graph = build_ask_graph(retriever)
    init: AskState = {"question": question, "regenerate_count": 0}
    if method:
        init["method"] = method
    final = graph.invoke(init)
    return {
        "answer": final.get("answer") or "",
        "refused": bool(final.get("refused")),
        "refusal_reason": final.get("refusal_reason"),
        "citations": final.get("citations") or [],
        "method": final.get("method") or _ask_method(),
    }


def stream_ask_events(
    retriever: HybridRetriever,
    question: str,
    *,
    method: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield SSE-friendly event dicts: status / token / final."""
    method = method or _ask_method()
    yield {"event": "status", "data": {"stage": "retrieve", "method": method}}
    hits = [h.to_dict() for h in retriever.search(question, top_k=ASK_TOP_K, method=method)]
    decision = evidence_gate(hits)
    if not decision.ok:
        final = {
            "answer": "",
            "refused": True,
            "refusal_reason": decision.reason,
            "citations": [],
            "method": method,
        }
        yield {"event": "final", "data": final}
        return

    citations = hits_to_citations(hits)
    yield {"event": "status", "data": {"stage": "generate", "n_citations": len(citations)}}

    answer_parts: list[str] = []
    for tok in stream_generate(question, citations, _contexts(hits)):
        answer_parts.append(tok)
        yield {"event": "token", "data": {"text": tok}}
    answer = normalize_citations("".join(answer_parts))

    ok, reason = _check(answer, citations, question, hits)
    if not ok and reason not in _NO_REGEN and not str(reason).startswith("low_grounding"):
        # one regenerate attempt (non-stream full regen), mirrors the graph path
        answer = generate_answer(question, citations, _contexts(hits))
        ok, reason = _check(answer, citations, question, hits)
    if not ok:
        final = {
            "answer": "",
            "refused": True,
            "refusal_reason": f"self_check_failed:{reason}",
            "citations": citations,
            "method": method,
        }
        yield {"event": "final", "data": final}
        return

    final = {
        "answer": answer,
        "refused": False,
        "refusal_reason": None,
        "citations": citations,
        "method": method,
    }
    yield {"event": "final", "data": final}
