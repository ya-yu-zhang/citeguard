from fastapi.testclient import TestClient

from citeguard.api import app as app_module
from citeguard.api.ratelimit import RateLimiter


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def test_per_ip_minute_window():
    c = Clock(); rl = RateLimiter(per_min=2, daily_cap=0, clock=c)
    assert rl.check("a")[0] and rl.check("a")[0]
    ok, info = rl.check("a")
    assert not ok and info["scope"] == "per_ip_minute" and info["retry_after_s"] >= 1
    assert rl.check("b")[0]  # other IP unaffected
    c.t += 61
    assert rl.check("a")[0]


def test_global_daily_cap_and_reset():
    c = Clock(); rl = RateLimiter(per_min=0, daily_cap=3, clock=c)
    assert all(rl.check(ip)[0] for ip in "abc")
    ok, info = rl.check("d")
    assert not ok and info["scope"] == "global_daily"
    c.t += 86400
    assert rl.check("d")[0]


def test_429_json_on_ask_and_stream(monkeypatch):
    monkeypatch.setattr(app_module.app.state, "limiter", RateLimiter(per_min=1, daily_cap=0))
    monkeypatch.setitem(app_module._state, "retriever", None)
    with TestClient(app_module.app) as client:
        app_module._state["retriever"] = None
        r1 = client.post("/ask", json={"question": "hi"})
        assert r1.status_code != 429
        r2 = client.post("/ask/stream", json={"question": "hi"})
        assert r2.status_code == 429
        assert r2.json()["error"] == "rate_limited" and r2.headers["retry-after"]
        assert client.get("/health").status_code == 200  # non-ask routes not limited
