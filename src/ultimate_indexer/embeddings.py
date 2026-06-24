from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

MAX_RETRIES = 3
BASE_DELAY_MS = 500
BATCH_SIZE = 32
# Default maximum tokens for API embedding requests
# Most APIs have limits (e.g., OpenAI: 8192, Cohere: 512-4096)
# We use a conservative default and allow override via environment
DEFAULT_MAX_TOKENS = int(os.getenv("ULTIMATE_INDEXER_API_MAX_TOKENS", "2048"))
# Approximate characters per token (varies by model/language)
# Using conservative estimate of 3 chars/token to account for code tokens being shorter
CHARS_PER_TOKEN = 3
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
            # Rate limits start at a 15 s floor and keep growing exponentially;
            # other errors use plain exponential backoff from the base delay.
            backoff_ms = base_delay_ms * (2 ** (attempt - 1))
            delay_ms = max(15_000 * (2 ** (attempt - 1)), backoff_ms) if is_rate_limit else backoff_ms
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


@dataclass(slots=True)
class APIEmbeddingProvider:
    """API-based embedding provider for remote embedding services."""
    
    api_endpoint: str
    api_model: str
    api_key: str | None = None
    model_id: str = "api"
    supports_batching: bool = True
    batch_size: int = 32
    timeout_seconds: float = 30.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_retries: int = MAX_RETRIES
    retry_base_delay_ms: int = BASE_DELAY_MS
    
    def prepare_document_text(self, content: str, file_path: str) -> str:
        return content
    
    def prepare_query_text(self, text: str) -> str:
        return text
    
    def _find_split_point(self, text: str) -> int:
        """Return the best character position to split *text* at, or 0 if none found.

        Tries split boundaries in preference order:
        1. Paragraph (double newline)
        2. Sentence end (.!?)
        3. Line boundary
        4. Word boundary

        Only accepts a split that leaves at least 30 % of the text on the left
        side so chunks stay reasonably large.
        """
        threshold = len(text) * 0.3
        for pattern in (r'\n\s*\n', r'(?<=[.!?])\s+', r'\n', r'\s'):
            for match in reversed(list(re.finditer(pattern, text))):
                if match.start() > threshold:
                    return match.end()
        return 0

    def _split_text(self, text: str, max_tokens: int) -> list[str]:
        """Split *text* into chunks that each fit within *max_tokens*.

        Uses boundary-aware splitting (paragraph → sentence → line → word) to
        avoid cutting in the middle of a logical unit.  Returns ``[text]`` when
        the input already fits.
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len(remaining) > max_chars:
            split_point = self._find_split_point(remaining[:max_chars])
            if split_point > 0:
                chunk, remaining = remaining[:split_point].strip(), remaining[split_point:].lstrip()
            else:
                chunk, remaining = remaining[:max_chars].strip(), remaining[max_chars:].lstrip()
            if chunk:
                chunks.append(chunk)
        if remaining.strip():
            chunks.append(remaining.strip())
        return chunks
    
    def _aggregate_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Aggregate multiple embeddings into a single vector.
        
        Uses mean pooling to combine embeddings from text chunks.
        This preserves semantic meaning while reducing dimensionality.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Single aggregated embedding vector
        """
        if not embeddings:
            return np.array([], dtype=np.float32)
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Stack embeddings and compute mean
        matrix = np.vstack(embeddings)
        return np.mean(matrix, axis=0).astype(np.float32)
    
    def _parse_embeddings_response(self, result: object, expected: int) -> list[list[float]]:
        embeddings: list[object]
        if isinstance(result, dict):
            if "error" in result:
                raise RuntimeError(f"Embedding API error response: {result['error']}")
            if "data" in result and isinstance(result["data"], list):
                data_items = sorted(
                    result["data"],
                    key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
                )
                embeddings = [item.get("embedding") if isinstance(item, dict) else None for item in data_items]
            elif "embeddings" in result and isinstance(result["embeddings"], list):
                embeddings = result["embeddings"]
            elif "embedding" in result:
                embeddings = [result["embedding"]]
            else:
                raise ValueError(f"Unexpected API response format: {result}")
        elif isinstance(result, list):
            embeddings = result
        else:
            raise ValueError(f"Unexpected API response type: {type(result)}")

        parsed: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("Embedding API returned an invalid embedding vector")
            parsed.append([float(value) for value in embedding])
        if expected > 1 and len(parsed) != expected:
            raise ValueError(
                f"Embedding API returned {len(parsed)} vectors for {expected} inputs"
            )
        return parsed

    def _call_api(self, texts: list[str], allow_context_split: bool = True) -> list[list[float]]:
        """Call the embedding API endpoint and return embeddings.

        Handles context length errors by automatically splitting oversized texts.
        ``allow_context_split=False`` disables that retry so the splitter cannot
        recurse into itself indefinitely when a chunk is still over the limit.
        """
        if not self.api_endpoint or not self.api_model:
            raise RuntimeError("Embedding API endpoint and model must both be configured")
        scheme = urllib.parse.urlparse(self.api_endpoint).scheme.lower()
        if scheme not in ("http", "https"):
            raise RuntimeError(
                f"Embedding API endpoint must be an http(s) URL, got scheme {scheme!r}"
            )
        payload = {
            "input": texts,
            "model": self.api_model,
        }
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        request = urllib.request.Request(
            self.api_endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
                return self._parse_embeddings_response(result, len(texts))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            
            # Check if this is a context length error - handle by splitting texts
            if (
                allow_context_split
                and e.code == 400
                and ("maximum context length" in error_body or "context length" in error_body)
            ):
                return self._handle_context_length_error(texts, error_body)

            raise RuntimeError(f"API request failed with status {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"API request failed: {e.reason}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse API response: {e}") from e
    
    def _handle_context_length_error(self, texts: list[str], error_body: str) -> list[list[float]]:
        """Re-embed *texts* after a 400 context-length error by splitting each one.

        Extracts the actual limit from the error body when available (uses 80 %
        of it to stay safe), otherwise falls back to 1 024 tokens.
        """
        match = re.search(r"maximum context length is (\d+) tokens", error_body)
        effective_max_tokens = int(int(match.group(1)) * 0.8) if match else 1024

        all_embeddings: list[list[float]] = []
        for text in texts:
            text_chunks = self._split_text(text, effective_max_tokens)
            # Re-embed sub-chunks with context-split DISABLED so a chunk that is
            # still over the real (tokenizer) limit fails loudly instead of
            # recursing into this handler forever.
            if len(text_chunks) == 1:
                all_embeddings.extend(
                    self._call_api(text_chunks, allow_context_split=False)
                )
            else:
                chunk_vecs = [
                    np.asarray(
                        self._call_api([chunk], allow_context_split=False)[0],
                        dtype=np.float32,
                    )
                    for chunk in text_chunks
                ]
                all_embeddings.append(self._aggregate_embeddings(chunk_vecs).tolist())
        return all_embeddings

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Embed a sequence of texts, batching those that need no splitting.

        Texts that fit within the API token limit are collected and sent as a
        single batched request.  Texts that exceed the limit are split, embedded
        chunk-by-chunk, and mean-pooled back to one vector.  This avoids N
        separate single-item API calls for the common case.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Classify each input: fits in one chunk vs. needs splitting.
        single_indices: list[int] = []
        single_texts: list[str] = []
        multi_chunks: dict[int, list[str]] = {}
        for i, text in enumerate(texts_list):
            chunks = self._split_text(text, self.max_tokens)
            if len(chunks) == 1:
                single_indices.append(i)
                single_texts.append(chunks[0])
            else:
                multi_chunks[i] = chunks

        results: list[np.ndarray | None] = [None] * len(texts_list)

        # Batch-embed all single-chunk texts in one API call sequence.
        if single_texts:
            batch_embeddings = self._embed_with_batching(single_texts)
            for idx, emb in zip(single_indices, batch_embeddings):
                results[idx] = emb

        # Split texts: embed each chunk set and aggregate.
        for i, chunks in multi_chunks.items():
            chunk_embs = self._embed_with_batching(chunks)
            results[i] = self._aggregate_embeddings(chunk_embs)

        return [e for e in results if e is not None]
    
    def _embed_with_batching(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts with API batching support.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            batch_embeddings = _with_retry(
                lambda: self._call_api(batch),
                label=f"API embedding batch {(start // self.batch_size) + 1}",
                max_retries=self.max_retries,
                base_delay_ms=self.retry_base_delay_ms,
            )
            assert isinstance(batch_embeddings, list)
            all_embeddings.extend(batch_embeddings)
        
        return [np.asarray(embedding, dtype=np.float32) for embedding in all_embeddings]
    
    def embed_query(self, text: str) -> np.ndarray:
        embeddings = self.embed([self.prepare_query_text(text)])
        return embeddings[0] if embeddings else np.array([], dtype=np.float32)


def resolve_api_provider(
    *,
    api_endpoint: str,
    api_model: str,
    api_key: str | None = None,
    model_id: str | None = None,
    batch_size: int = 32,
    timeout_seconds: float = 30.0,
    max_tokens: int | None = None,
    max_retries: int = MAX_RETRIES,
    retry_base_delay_ms: int = BASE_DELAY_MS,
) -> APIEmbeddingProvider:
    """Create an API embedding provider from configuration.
    
    Args:
        api_endpoint: The API endpoint URL
        api_model: The model name to use
        api_key: Optional API key for authentication
        model_id: Optional custom model identifier
        batch_size: Number of texts to send in each API batch
        timeout_seconds: Request timeout in seconds
        max_tokens: Maximum tokens per API request (default from env)
        
    Returns:
        Configured APIEmbeddingProvider instance
    """
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS
    return APIEmbeddingProvider(
        api_endpoint=api_endpoint,
        api_model=api_model,
        api_key=api_key,
        model_id=model_id or f"api:{api_model}",
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_base_delay_ms=retry_base_delay_ms,
    )


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
