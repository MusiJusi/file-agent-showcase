"""Keep developer launch profiles out of network-free behavior tests."""

import pytest


@pytest.fixture(autouse=True)
def isolate_runtime_profile(monkeypatch):
    # Individual tests can still opt into another mode with monkeypatch.setenv.
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("RAG_MODE", "standard")
    monkeypatch.setenv("RERANKER_MODEL", "")
    monkeypatch.setenv("MULTI_QUERY", "off")
