from citeguard.ask import graph, judge
from citeguard.ask.judge import JudgeResult


def test_judge_mode_default_off(monkeypatch):
    monkeypatch.delenv("JUDGE_MODE", raising=False)
    assert judge.judge_mode() == "off"
    monkeypatch.setenv("JUDGE_MODE", "bogus")
    assert judge.judge_mode() == "off"


def test_retry_once_on_bad_json(monkeypatch):
    calls = []

    def fake(*a):
        calls.append(1)
        return JudgeResult([], None, usage={"cost": 1}, error="bad_json") if len(calls) == 1 else \
            JudgeResult([{"i": 1, "supported": True}], True, usage={"cost": 2})
    monkeypatch.setattr(judge, "_judge_once", fake)
    r = judge.judge_answer("q", "a [1].", ["t"])
    assert len(calls) == 2 and not r.flagged and r.usage["cost"] == 3


def test_veto_only_after_overlap_passes(monkeypatch):
    monkeypatch.setattr(graph, "llm_configured", lambda: True)
    monkeypatch.setattr(graph, "self_check_answer", lambda *a, **k: (True, None))

    class Rep:
        wrong_source = 0; supported = 1; answer = "x [1]."
    monkeypatch.setattr(graph, "repair_citations", lambda *a, **k: Rep())
    monkeypatch.setattr(graph, "judge_answer", lambda *a, **k: JudgeResult([{"supported": False}], True))
    monkeypatch.setenv("JUDGE_MODE", "off")
    assert graph._check("x [1].", [], "q", [{"text": "t"}])[1] is True
    monkeypatch.setenv("JUDGE_MODE", "veto")
    ans, ok, reason = graph._check("x [1].", [], "q", [{"text": "t"}])
    assert not ok and reason.startswith("judge_veto") and graph.refusal_reason(reason).startswith("ungrounded_answer")
