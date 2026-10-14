"""In-process rate limiting for /ask and /ask/stream.

  ASK_RATE_PER_MIN  per-client-IP requests per rolling 60 s (default 10; 0 = unlimited)
  ASK_DAILY_CAP     global requests per UTC day across all clients (default 500; 0 = unlimited)
  ASK_TRUST_PROXY   1 → use the first X-Forwarded-For hop as client IP (behind HF/other proxy)

Single-process only (state is in memory); multiple workers each get their own counters.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class RateLimiter:
    def __init__(self, per_min: int | None = None, daily_cap: int | None = None, clock=time.time):
        self.per_min = _int("ASK_RATE_PER_MIN", 10) if per_min is None else per_min
        self.daily_cap = _int("ASK_DAILY_CAP", 500) if daily_cap is None else daily_cap
        self.clock = clock
        self._ip: dict[str, deque] = {}
        self._day = ""
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, ip: str) -> tuple[bool, dict]:
        now = self.clock()
        day = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if day != self._day:
                self._day, self._day_count = day, 0
            if self.daily_cap and self._day_count >= self.daily_cap:
                return False, {"error": "rate_limited", "scope": "global_daily",
                               "limit": self.daily_cap, "retry_after_s": int(86400 - now % 86400)}
            q = self._ip.setdefault(ip, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if self.per_min and len(q) >= self.per_min:
                return False, {"error": "rate_limited", "scope": "per_ip_minute",
                               "limit": self.per_min, "retry_after_s": max(1, int(60 - (now - q[0])) + 1)}
            q.append(now)
            self._day_count += 1
            return True, {}


def client_ip(request: Request) -> str:
    if os.environ.get("ASK_TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def limited_response(info: dict) -> JSONResponse:
    return JSONResponse(status_code=429, content=info, headers={"Retry-After": str(info["retry_after_s"])})
