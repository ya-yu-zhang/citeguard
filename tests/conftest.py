import pytest


@pytest.fixture(autouse=True)
def _stub_llm_by_default(monkeypatch, request):
    """Unit tests never hit a live LLM, even if .env sets LLM_MODEL."""
    if request.node.get_closest_marker("ollama") is None:
        monkeypatch.setenv("LLM_MODEL", "")
