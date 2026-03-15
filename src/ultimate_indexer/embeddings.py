from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

MAX_RETRIES = 3
BASE_DELAY_MS = 500
BATCH_SIZE = 32
PROVIDER_BATCH_DELAY_MS = {
    "hash-256": 0,
}


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    safe_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    return (matrix @ query_vector) / (safe_norms * query_norm)


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


def prepare_document_text(content: str, file_path: str) -> str:
    return f"search_document: {file_path}\n{content}"


def _with_retry(
    operation: Callable[[], np.ndarray | list[np.ndarray] | str],
    label: str,
    max_retries: int = MAX_RETRIES,
    base_delay_ms: int = BASE_DELAY_MS,
    sleep_fn: Callable[[float], None] = time.sleep,
):
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return operation()
        except Exception as exc:  # pragma: no cover - exercised via tests with fake providers
            last_error = exc
            if attempt >= max_retries:
                break
            message = str(exc)
            is_rate_limit = any(marker in message for marker in ("429", "RESOURCE_EXHAUSTED", "quota"))
            delay_ms = max(base_delay_ms * (2 ** (attempt - 1)), 15_000) if is_rate_limit else base_delay_ms * (2 ** (attempt - 1))
            sleep_fn(delay_ms / 1000.0)
    assert last_error is not None
    raise last_error


@dataclass(slots=True)
class HashEmbeddingProvider:
    dim: int = 256
    model_id: str = "hash-256"

    def _embed_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        data = bytearray()
        while len(data) < self.dim:
            digest = hashlib.sha256(digest + text.encode("utf-8")).digest()
            data.extend(digest)
        arr = np.frombuffer(bytes(data[: self.dim]), dtype=np.uint8).astype(np.float32)
        arr = (arr - 127.5) / 127.5
        return arr

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(f"query::{text}")


@dataclass(slots=True)
class LlamaCppEmbeddingProvider:
    model_path: Path
    model_id: str
    _llama: object | None = field(init=False, repr=False, default=None)

    def _client(self):
        if self._llama is None:
            from llama_cpp import Llama

            self._llama = Llama(
                model_path=str(self.model_path),
                embedding=True,
                pooling_type=1,
                verbose=False,
            )
        return self._llama

    def _embed_one(self, text: str) -> np.ndarray:
        response = self._client().create_embedding(text)
        vector = response["data"][0]["embedding"]
        return np.asarray(vector, dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(f"search_query: {text}")


def resolve_llama_cpp_provider(
    model_cache_dir: Path,
    repo_id: str,
    filename: str,
) -> LlamaCppEmbeddingProvider:
    from huggingface_hub import hf_hub_download

    model_path = Path(_with_retry(
        lambda: hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_cache_dir),
        ),
        label=f"Download model {repo_id}/{filename}",
    )
    )
    return LlamaCppEmbeddingProvider(
        model_path=model_path,
        model_id=f"{repo_id}:{filename}",
    )


def generate_embeddings(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    on_batch_complete: Callable[[int, int], None] | None = None,
    batch_size: int = BATCH_SIZE,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[np.ndarray]:
    if not texts:
        return []
    results: list[np.ndarray] = []
    batch_delay_ms = PROVIDER_BATCH_DELAY_MS.get(provider.model_id, 0)
    for start in range(0, len(texts), batch_size):
        if start > 0 and batch_delay_ms > 0:
            sleep_fn(batch_delay_ms / 1000.0)
        batch = texts[start : start + batch_size]
        embeddings = _with_retry(
            lambda: provider.embed(batch),
            label=f"Embedding batch {(start // batch_size) + 1}",
            sleep_fn=sleep_fn,
        )
        assert isinstance(embeddings, list)
        results.extend(embeddings)
        if on_batch_complete is not None:
            on_batch_complete(min(start + len(batch), len(texts)), len(texts))
    return results


def generate_query_embedding(
    provider: EmbeddingProvider,
    text: str,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    result = _with_retry(
        lambda: provider.embed_query(text),
        label="Query embedding",
        sleep_fn=sleep_fn,
    )
    assert isinstance(result, np.ndarray)
    return result
