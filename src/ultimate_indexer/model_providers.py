"""Optional model-backed providers for the retrieval upgrades.

Both talk plain HTTP to OpenAI / llama.cpp-server-compatible endpoints, so the
same code works against a local ``llama-server``, TEI, infinity, Jina or Cohere.
They are entirely optional: the resolvers return ``None`` when unconfigured and
callers fall back to the deterministic feature reranker / template HyDE.

- ``APIReranker``    → cross-encoder rerank via ``/v1/rerank``
  (``{model, query, documents}`` → ``{results:[{index, relevance_score}]}``).
- ``APIHydeGenerator`` → hypothetical-document generation via
  ``/v1/chat/completions`` (``choices[0].message.content``).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


def _require_http_url(endpoint: str, label: str) -> None:
    scheme = urllib.parse.urlparse(endpoint).scheme.lower()
    if scheme not in ("http", "https"):
        raise RuntimeError(f"{label} must be an http(s) URL, got scheme {scheme!r}")


def _post_json(endpoint: str, payload: dict, *, api_key: str | None, timeout: float, label: str) -> object:
    _require_http_url(endpoint, label)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(f"{label} failed with status {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label} request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {label} response: {exc}") from exc


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class RerankProvider(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per input document, in input order."""
        ...


def _parse_rerank_response(result: object, expected: int) -> list[float]:
    scores = [0.0] * expected
    items: object = None
    if isinstance(result, dict):
        if "error" in result:
            raise RuntimeError(f"Rerank API error response: {result['error']}")
        items = result.get("results") or result.get("data")
    elif isinstance(result, list):
        items = result
    if not isinstance(items, list):
        raise ValueError(f"Unexpected rerank response format: {result!r}")
    for item in items:
        if not isinstance(item, dict):
            continue
        index = int(item.get("index", -1))
        raw = item.get("relevance_score", item.get("score"))
        if 0 <= index < expected and raw is not None:
            scores[index] = float(raw)
    return scores


@dataclass(slots=True)
class APIReranker:
    endpoint: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        result = _post_json(
            self.endpoint, payload, api_key=self.api_key,
            timeout=self.timeout_seconds, label="Rerank API",
        )
        return _parse_rerank_response(result, len(documents))


def resolve_rerank_provider(
    *,
    endpoint: str | None,
    model: str | None,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> RerankProvider | None:
    if not endpoint or not model:
        return None
    return APIReranker(
        endpoint=endpoint, model=model, api_key=api_key, timeout_seconds=timeout_seconds
    )


# ---------------------------------------------------------------------------
# HyDE generator
# ---------------------------------------------------------------------------


class HydeGenerator(Protocol):
    def generate(self, query: str) -> str:
        """Return a hypothetical code snippet that would answer *query*."""
        ...


_HYDE_SYSTEM_PROMPT = (
    "You write a short, plausible code snippet that would directly answer the "
    "user's question about a codebase. Output ONLY code with a brief docstring or "
    "comment — no prose, no markdown fences. Keep it under 15 lines."
)


def _parse_chat_response(result: object) -> str:
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected chat response type: {type(result)}")
    if "error" in result:
        raise RuntimeError(f"Chat API error response: {result['error']}")
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str):
            return text
    raise ValueError(f"Unexpected chat response format: {result!r}")


@dataclass(slots=True)
class APIHydeGenerator:
    endpoint: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_tokens: int = 256

    def generate(self, query: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        result = _post_json(
            self.endpoint, payload, api_key=self.api_key,
            timeout=self.timeout_seconds, label="HyDE API",
        )
        return _parse_chat_response(result).strip()


def resolve_hyde_generator(
    *,
    endpoint: str | None,
    model: str | None,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
    max_tokens: int = 256,
) -> HydeGenerator | None:
    if not endpoint or not model:
        return None
    return APIHydeGenerator(
        endpoint=endpoint, model=model, api_key=api_key,
        timeout_seconds=timeout_seconds, max_tokens=max_tokens,
    )
