from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from contextlib import nullcontext, redirect_stderr, redirect_stdout
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
DEFAULT_MODEL_FILENAMES = (
    "coderankembed-q8_0.gguf",
    "nomic-embed-code-Q4_K_M.gguf",
)


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


def _backend_log_guard(suppress_logs: bool):
    if not suppress_logs:
        return nullcontext()
    try:
        from llama_cpp._utils import suppress_stdout_stderr
    except Exception:
        return nullcontext()
    return suppress_stdout_stderr(disable=False)


def _stream_silence_guard(suppress_logs: bool):
    if not suppress_logs:
        return nullcontext(), nullcontext()
    sink = io.StringIO()
    return redirect_stdout(sink), redirect_stderr(sink)


class EmbeddingProvider(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


def prepare_document_text(content: str, file_path: str) -> str:
    return f"{file_path}\n{content}"


def _provider_prepare_document_text(provider: EmbeddingProvider, content: str, file_path: str) -> str:
    formatter = getattr(provider, "prepare_document_text", None)
    if callable(formatter):
        return str(formatter(content, file_path))
    return prepare_document_text(content, file_path)


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


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_local_model_path(
    project_root: Path,
    model_cache_dir: Path,
    *,
    model_path: str | None,
    filename: str,
) -> Path:
    tried: list[Path] = []

    def _append_candidate(candidate: Path) -> None:
        resolved = candidate.expanduser()
        if not resolved.is_absolute():
            resolved = (project_root / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved not in tried:
            tried.append(resolved)

    if model_path:
        _append_candidate(Path(model_path))

    repo_root = _package_root()
    for base in (
        project_root / "models",
        project_root / "graph-indexer" / "models",
        repo_root / "models",
        repo_root / "graph-indexer" / "models",
        model_cache_dir,
    ):
        _append_candidate(base / filename)

    for fallback_name in DEFAULT_MODEL_FILENAMES:
        if fallback_name == filename:
            continue
        for base in (
            project_root / "models",
            project_root / "graph-indexer" / "models",
            repo_root / "models",
            repo_root / "graph-indexer" / "models",
            model_cache_dir,
        ):
            _append_candidate(base / fallback_name)

    for candidate in tried:
        if candidate.exists() and candidate.is_file():
            return candidate

    attempted = "\n".join(f"- {candidate}" for candidate in tried)
    raise RuntimeError(
        "No local GGUF embedding model was found.\n"
        "Set `ULTIMATE_INDEXER_MODEL_PATH` explicitly or place the committed model under "
        "`models/` or `graph-indexer/models/`.\n"
        f"Tried:\n{attempted}"
    )


@dataclass(slots=True)
class HashEmbeddingProvider:
    dim: int = 256
    model_id: str = "hash-256"
    supports_batching: bool = True

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
    supports_batching: bool = False
    n_ctx: int = 2048
    n_batch: int = 2048
    n_ubatch: int = 2048
    n_gpu_layers: int = field(
        default_factory=lambda: -1 if sys.platform == "darwin" else 0
    )
    verbose: bool = False
    suppress_backend_logs: bool = True
    _llama: object | None = field(init=False, repr=False, default=None)

    def prepare_document_text(self, content: str, file_path: str) -> str:
        return content

    def prepare_query_text(self, text: str) -> str:
        return text

    def _client(self):
        if self._llama is None:
            from llama_cpp import Llama

            with _backend_log_guard(self.suppress_backend_logs):
                self._llama = Llama(
                    model_path=str(self.model_path),
                    embedding=True,
                    n_ctx=self.n_ctx,
                    n_batch=self.n_batch,
                    n_ubatch=self.n_ubatch,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=self.verbose,
                )
        return self._llama

    def _embed_one(self, text: str) -> np.ndarray:
        with _backend_log_guard(self.suppress_backend_logs):
            response = self._client().create_embedding(text)
        vector = response["data"][0]["embedding"]
        return np.asarray(vector, dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(self.prepare_query_text(text))


def resolve_llama_cpp_provider(
    project_root: Path,
    model_cache_dir: Path,
    *,
    model_path: str | None,
    filename: str,
) -> LlamaCppEmbeddingProvider:
    resolved_model_path = resolve_local_model_path(
        project_root,
        model_cache_dir,
        model_path=model_path,
        filename=filename,
    )
    n_ctx = int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_CTX", "2048"))
    n_batch = int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_BATCH", str(n_ctx)))
    n_ubatch = int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_UBATCH", str(n_ctx)))
    return LlamaCppEmbeddingProvider(
        model_path=resolved_model_path,
        model_id=f"local-gguf:{resolved_model_path.resolve()}",
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        n_gpu_layers=int(
            os.getenv(
                "ULTIMATE_INDEXER_LLAMA_N_GPU_LAYERS",
                "-1"
            )
        ),
        verbose=os.getenv("ULTIMATE_INDEXER_LLAMA_VERBOSE", "false").lower() == "true",
        suppress_backend_logs=os.getenv("ULTIMATE_INDEXER_LLAMA_SUPPRESS_LOGS", "true").lower() != "false",
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
    provider_batch_size = getattr(provider, "batch_size", batch_size)
    effective_batch_size = max(
        1,
        min(batch_size, int(provider_batch_size)) if getattr(provider, "supports_batching", True) else 1,
    )
    batch_delay_ms = PROVIDER_BATCH_DELAY_MS.get(provider.model_id, 0)
    for start in range(0, len(texts), effective_batch_size):
        if start > 0 and batch_delay_ms > 0:
            sleep_fn(batch_delay_ms / 1000.0)
        batch = texts[start : start + effective_batch_size]
        embeddings = _with_retry(
            lambda: provider.embed(batch),
            label=f"Embedding batch {(start // effective_batch_size) + 1}",
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
