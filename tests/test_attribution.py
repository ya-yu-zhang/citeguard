from citeguard.ask.attribution import is_out_of_scope_task, repair_citations, split_units

CTX = [
    "Use model_dump() to convert a model instance into a Python dictionary.",
    "Raise HTTPException with status_code and detail to return HTTP errors to the client.",
]
Q = "How do I serialize a Pydantic model?"


def test_split_attaches_trailing_markers():
    assert split_units("Foo bar baz. [1] Second sentence here.") == ["Foo bar baz [1].", "Second sentence here."]


def test_missing_marker_is_attributed_to_best_chunk():
    r = repair_citations("Call model_dump() to convert the instance into a Python dictionary.", Q, CTX)
    assert r.answer.endswith("[1].") and r.reattributed == 1 and r.wrong_source == 0


def test_literal_n_marker_is_repaired():
    r = repair_citations("Call model_dump() to get a Python dictionary [n].", Q, CTX)
    assert "[1]" in r.answer and "[n]" not in r.answer


def test_wrong_citation_reattributed_when_another_chunk_supports():
    r = repair_citations("Call model_dump() to get a Python dictionary [2].", Q, CTX)
    assert "[1]" in r.answer and "[2]" not in r.answer


def test_unsupported_cited_sentence_is_wrong_source():
    r = repair_citations("Use the orjson serializer with custom encoders and streaming [1].", Q, CTX)
    assert r.wrong_source == 1 and r.supported == 0


def test_uncited_unsupported_sentence_dropped_rest_kept():
    r = repair_citations(
        "Call model_dump() to get a Python dictionary [1]. Bananas grow quickly in tropical climates.", Q, CTX
    )
    assert r.dropped == 1 and r.supported == 1 and "Bananas" not in r.answer


def test_out_of_scope_task():
    assert is_out_of_scope_task("Write a romantic poem about two toasters falling in love.")
    assert not is_out_of_scope_task("Write a test for a FastAPI endpoint with TestClient")
    assert not is_out_of_scope_task("How do I stream tokens in LangGraph?")
