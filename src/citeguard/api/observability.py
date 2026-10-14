"""Lightweight request tracing: request_id contextvar + one JSON log line per /ask.

Zero extra deps (stdlib logging). Logger name: ``citeguard.request``.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

logger = logging.getLogger("citeguard.request")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # lets pytest caplog capture it


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def current_request_id() -> str | None:
    return request_id_var.get()


def log_ask(**fields: Any) -> None:
    """Emit one JSON line, e.g. {"event":"ask_complete","request_id":...}."""
    payload = {"event": "ask_complete", **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))
