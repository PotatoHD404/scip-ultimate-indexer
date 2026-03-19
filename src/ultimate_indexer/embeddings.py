from __future__ import annotations

import hashlib
import io
import logging
import os
import re
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
CODERANK_QUERY_PREFIX = "Represent this query for searching relevant code: "


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
    return f"search_document: {file_path}\n{content}"


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


def _remote_dependency_install_hint(model_name: str, exc: ImportError) -> str | None:
    message = str(exc)
    packages: list[str] = []
    quoted_module = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
    if quoted_module:
        packages = [quoted_module.group(1)]
    else:
        match = re.search(r"packages that were not found in your environment: ([^.]+)\.", message)
        if match:
            packages = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not packages:
        return None
    package_list = " ".join(packages)
    return (
        f"`{model_name}` requires additional packages: {', '.join(packages)}. "
        f"Install them with `poetry add {package_list}` or `pip install {package_list}`."
    )


def _sentence_transformer_batch_size(device: str) -> int:
    if device == "cuda":
        return 32
    if device == "mps":
        return 8
    return 16


def _sentence_transformer_max_seq_length(device: str) -> int:
    if device == "mps":
        return 512
    if device == "cuda":
        return 1024
    return 1024


def _apply_sentence_transformer_limits(model: object, max_seq_length: int) -> None:
    if max_seq_length <= 0:
        return
    current = getattr(model, "max_seq_length", None)
    if current is None:
        setattr(model, "max_seq_length", max_seq_length)
    else:
        setattr(model, "max_seq_length", min(int(current), max_seq_length))

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit is None:
        setattr(tokenizer, "model_max_length", max_seq_length)
    else:
        setattr(tokenizer, "model_max_length", min(int(tokenizer_limit), max_seq_length))


def _mps_memory_hint(model_name: str) -> str:
    return (
        f"`{model_name}` exhausted Apple GPU memory on `mps`. "
        "Retry with `ULTIMATE_INDEXER_ST_BATCH_SIZE=2` and/or "
        "`ULTIMATE_INDEXER_ST_MAX_SEQ_LENGTH=256`, or force CPU with "
        "`ULTIMATE_INDEXER_ST_DEVICE=cpu`."
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
    n_ctx: int = 0
    n_batch: int = 1024
    n_ubatch: int = 1024
    n_gpu_layers: int = field(
        default_factory=lambda: -1 if sys.platform == "darwin" else 0
    )
    verbose: bool = False
    suppress_backend_logs: bool = True
    _llama: object | None = field(init=False, repr=False, default=None)

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
        return self._embed_one(f"search_query: {text}")


@dataclass(slots=True)
class SentenceTransformerEmbeddingProvider:
    model_name: str
    cache_dir: Path
    model_id: str
    device: str
    batch_size: int = 64
    max_seq_length: int = 0
    use_fp16: bool = False
    normalize_embeddings: bool = True
    suppress_backend_logs: bool = True
    supports_batching: bool = True
    _model: object | None = field(init=False, repr=False, default=None)

    def prepare_document_text(self, content: str, file_path: str) -> str:
        return f"{file_path}\n{content}"

    def prepare_query_text(self, text: str) -> str:
        return f"{CODERANK_QUERY_PREFIX}{text}"

    def _client(self):
        if self._model is None:
            if self.device == "mps":
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            try:
                import torch
                from sentence_transformers import SentenceTransformer

                huggingface_logger = logging.getLogger("huggingface_hub")
                sentence_transformers_logger = logging.getLogger("sentence_transformers")
                transformers_logger = logging.getLogger("transformers")
                previous_levels = (
                    huggingface_logger.level,
                    sentence_transformers_logger.level,
                    transformers_logger.level,
                )
                try:
                    if self.suppress_backend_logs:
                        huggingface_logger.setLevel(logging.ERROR)
                        sentence_transformers_logger.setLevel(logging.ERROR)
                        transformers_logger.setLevel(logging.ERROR)
                    stdout_guard, stderr_guard = _stream_silence_guard(self.suppress_backend_logs)
                    with stdout_guard, stderr_guard:
                        if hasattr(torch, "set_float32_matmul_precision"):
                            torch.set_float32_matmul_precision("high")
                        self._model = SentenceTransformer(
                            self.model_name,
                            cache_folder=str(self.cache_dir),
                            trust_remote_code=True,
                            device=self.device,
                        )
                        _apply_sentence_transformer_limits(self._model, self.max_seq_length)
                        if self.use_fp16:
                            try:
                                self._model.half()
                            except Exception:
                                pass
                finally:
                    if self.suppress_backend_logs:
                        huggingface_logger.setLevel(previous_levels[0])
                        sentence_transformers_logger.setLevel(previous_levels[1])
                        transformers_logger.setLevel(previous_levels[2])
            except ImportError as exc:
                hint = _remote_dependency_install_hint(self.model_name, exc)
                if hint is None:
                    raise
                raise RuntimeError(hint) from exc
        return self._model

    def embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        import torch

        try:
            stdout_guard, stderr_guard = _stream_silence_guard(self.suppress_backend_logs)
            with torch.inference_mode(), stdout_guard, stderr_guard:
                encoded = self._client().encode(
                    list(texts),
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
        except RuntimeError as exc:
            message = str(exc)
            if self.device == "mps" and any(marker in message for marker in ("Invalid buffer size", "out of memory", "MPS")):
                raise RuntimeError(_mps_memory_hint(self.model_name)) from exc
            raise
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim == 1:
            return [matrix]
        return [row for row in matrix]

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([self.prepare_query_text(text)])[0]


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
        n_ctx=int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_CTX", "0")),
        n_batch=int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_BATCH", "64")),
        n_ubatch=int(os.getenv("ULTIMATE_INDEXER_LLAMA_N_UBATCH", "64")),
        n_gpu_layers=int(
            os.getenv(
                "ULTIMATE_INDEXER_LLAMA_N_GPU_LAYERS",
                "-1"
            )
        ),
        verbose=os.getenv("ULTIMATE_INDEXER_LLAMA_VERBOSE", "false").lower() == "true",
        suppress_backend_logs=os.getenv("ULTIMATE_INDEXER_LLAMA_SUPPRESS_LOGS", "true").lower() != "false",
    )


def _detect_sentence_transformer_device() -> str:
    device_override = os.getenv("ULTIMATE_INDEXER_ST_DEVICE")
    if device_override:
        return device_override
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_sentence_transformer_provider(
    model_cache_dir: Path,
    model_name: str,
) -> SentenceTransformerEmbeddingProvider:
    device = _detect_sentence_transformer_device()
    default_batch_size = _sentence_transformer_batch_size(device)
    default_max_seq_length = _sentence_transformer_max_seq_length(device)
    use_fp16_env = os.getenv("ULTIMATE_INDEXER_ST_USE_FP16")
    use_fp16 = (
        use_fp16_env.lower() == "true"
        if use_fp16_env is not None
        else device in {"cuda", "mps"}
    )
    return SentenceTransformerEmbeddingProvider(
        model_name=model_name,
        cache_dir=model_cache_dir,
        model_id=model_name,
        device=device,
        batch_size=int(os.getenv("ULTIMATE_INDEXER_ST_BATCH_SIZE", str(default_batch_size))),
        max_seq_length=int(os.getenv("ULTIMATE_INDEXER_ST_MAX_SEQ_LENGTH", str(default_max_seq_length))),
        use_fp16=use_fp16,
        normalize_embeddings=os.getenv("ULTIMATE_INDEXER_ST_NORMALIZE", "true").lower() != "false",
        suppress_backend_logs=os.getenv("ULTIMATE_INDEXER_ST_SUPPRESS_LOGS", "true").lower() != "false",
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
