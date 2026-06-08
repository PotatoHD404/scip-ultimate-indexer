"""Tests for the optional model-backed reranker / HyDE providers and their
graceful integration into the query engine. No real models or network needed."""

from __future__ import annotations

import json

import pytest

from ultimate_indexer import hyde
from ultimate_indexer import model_providers as mp
from ultimate_indexer.models import RankedSymbol
from ultimate_indexer.query import QueryConfig, QueryEngine


class _FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


# --- pure parsing -----------------------------------------------------------


def test_resolvers_return_none_when_unconfigured():
    assert mp.resolve_rerank_provider(endpoint=None, model=None) is None
    assert mp.resolve_rerank_provider(endpoint="http://x", model=None) is None
    assert mp.resolve_rerank_provider(endpoint=None, model="m") is None
    assert mp.resolve_hyde_generator(endpoint=None, model=None) is None
    assert mp.resolve_rerank_provider(endpoint="http://x", model="m") is not None


def test_parse_rerank_response_reorders_by_index():
    result = {"results": [
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.1},
    ]}
    assert mp._parse_rerank_response(result, 3) == [0.1, 0.0, 0.9]


def test_parse_rerank_response_accepts_score_alias_and_data_key():
    result = {"data": [{"index": 0, "score": 0.5}, {"index": 1, "score": 0.25}]}
    assert mp._parse_rerank_response(result, 2) == [0.5, 0.25]


def test_parse_chat_response_openai_and_text_shapes():
    assert mp._parse_chat_response({"choices": [{"message": {"content": "def f(): ..."}}]}) == "def f(): ..."
    assert mp._parse_chat_response({"choices": [{"text": "x = 1"}]}) == "x = 1"
    with pytest.raises(ValueError):
        mp._parse_chat_response({"nope": 1})


# --- HTTP layer (fake urlopen) ---------------------------------------------


def test_api_reranker_posts_and_parses(monkeypatch):
    captured: dict = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {"results": [{"index": 0, "relevance_score": 0.7}, {"index": 1, "relevance_score": 0.2}]}
        )

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    reranker_provider = mp.APIReranker(
        endpoint="http://localhost:8080/v1/rerank", model="bge", api_key="secret"
    )
    scores = reranker_provider.score("greeting", ["doc a", "doc b"])
    assert scores == [0.7, 0.2]
    assert captured["body"]["query"] == "greeting"
    assert captured["body"]["documents"] == ["doc a", "doc b"]
    assert captured["body"]["model"] == "bge"
    # Bearer auth header is set (urllib title-cases header keys).
    assert captured["headers"].get("Authorization") == "Bearer secret"


def test_api_reranker_rejects_non_http_scheme():
    with pytest.raises(RuntimeError):
        mp.APIReranker(endpoint="file:///etc/passwd", model="x").score("q", ["d"])


def test_api_hyde_generator_posts_and_parses(monkeypatch):
    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        assert body["messages"][-1]["content"] == "how is auth handled"
        return _FakeHTTPResponse({"choices": [{"message": {"content": "def authenticate(): pass"}}]})

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    generator = mp.APIHydeGenerator(
        endpoint="http://localhost:8080/v1/chat/completions", model="m"
    )
    assert generator.generate("how is auth handled") == "def authenticate(): pass"


# --- query-engine integration (injected fakes) ------------------------------


class _DummyStorage:
    pass


class _DummyProvider:
    model_id = "dummy"


class _FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score(self, query: str, documents: list[str]) -> list[float]:
        return self._scores[: len(documents)]


class _RaisingReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        raise RuntimeError("boom")


def _engine(monkeypatch) -> QueryEngine:
    for var in (
        "ULTIMATE_INDEXER_RERANK_API_ENDPOINT", "ULTIMATE_INDEXER_RERANK_API_MODEL",
        "ULTIMATE_INDEXER_HYDE_API_ENDPOINT", "ULTIMATE_INDEXER_HYDE_API_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return QueryEngine(_DummyStorage(), _DummyProvider(), QueryConfig())


def _sym(symbol_id: str, name: str, stage1: float) -> RankedSymbol:
    return RankedSymbol(
        symbol_id=symbol_id, relative_path="pkg/x.py", display_name=name,
        kind="Function", score=stage1, signature="", docstring="", snippet="",
    )


def test_model_reranker_reorders_by_relevance(monkeypatch):
    engine = _engine(monkeypatch)
    engine._rerank_provider = _FakeReranker([0.1, 0.9])  # second doc far more relevant
    ranked = [_sym("a", "alpha", 0.9), _sym("b", "beta", 0.1)]
    out = engine._rerank("query", ranked)
    assert out[0].symbol_id == "b"  # model relevance lifts b above the stronger-stage1 a


def test_model_reranker_falls_back_to_features_on_error(monkeypatch):
    engine = _engine(monkeypatch)
    engine._rerank_provider = _RaisingReranker()
    ranked = [_sym("a", "build_greeting", 0.5), _sym("b", "unrelated", 0.45)]
    out = engine._rerank("build_greeting", ranked)
    assert out[0].symbol_id == "a"  # feature reranker still promotes the exact match


def test_hypothetical_text_prefers_generator_then_falls_back(monkeypatch):
    engine = _engine(monkeypatch)

    class _Gen:
        def generate(self, query: str) -> str:
            return "def generated_answer():\n    return True\n"

    engine._hyde_generator = _Gen()
    assert "generated_answer" in engine._hypothetical_text("how is x done")

    class _Raises:
        def generate(self, query: str) -> str:
            raise RuntimeError("down")

    engine._hyde_generator = _Raises()
    assert engine._hypothetical_text("how are users greeted") == hyde.hypothetical_code(
        "how are users greeted"
    )
