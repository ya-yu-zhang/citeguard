from citeguard.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_prefers_consensus():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = reciprocal_rank_fusion([a, b], k=60)
    ids = [i for i, _ in fused]
    # y and x appear in both → should outrank z/w
    assert ids[0] in {"x", "y"}
    assert "y" in ids[:2] and "x" in ids[:2]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_scores_decrease():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)
