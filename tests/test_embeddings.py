from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

import ultimate_indexer.embeddings as embeddings_module
from ultimate_indexer.embeddings import (
    LlamaCppEmbeddingProvider,
    generate_embeddings,
    resolve_llama_cpp_provider,
    resolve_local_model_path,
)


def test_llama_provider_loads_lazily(monkeypatch) -> None:
    calls: list[str] = []

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            calls.append("init")

        def create_embedding(self, text: str):
            calls.append(text)
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    fake_module = types.SimpleNamespace(Llama=FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

    provider = LlamaCppEmbeddingProvider(
        model_path=Path("/tmp/fake.gguf"),
        model_id="fake:model",
    )
    assert calls == []

    vector = provider.embed_query("hello world")
    assert calls[0] == "init"
    assert calls[1] == "hello world"
    assert vector.shape[0] == 3


def test_generate_embeddings_retries_once() -> None:
    class FlakyProvider:
        model_id = "hash-256"

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return [np.asarray([1.0, 2.0], dtype=np.float32) for _ in texts]

        def embed_query(self, text: str):
            return np.asarray([1.0, 2.0], dtype=np.float32)

    sleeps: list[float] = []
    provider = FlakyProvider()
    vectors = generate_embeddings(provider, ["a", "b"], sleep_fn=sleeps.append)
    assert provider.calls == 2
    assert sleeps == [0.5]
    assert len(vectors) == 2


def test_generate_embeddings_non_batching_provider_updates_per_item() -> None:
    class NonBatchingProvider:
        model_id = "non-batching"
        supports_batching = False

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [np.asarray([1.0, 2.0], dtype=np.float32) for _ in texts]

        def embed_query(self, text: str):
            return np.asarray([1.0, 2.0], dtype=np.float32)

    provider = NonBatchingProvider()
    progress: list[tuple[int, int]] = []
    vectors = generate_embeddings(
        provider,
        ["a", "b", "c"],
        on_batch_complete=lambda processed, total: progress.append((processed, total)),
    )

    assert len(vectors) == 3
    assert provider.calls == [["a"], ["b"], ["c"]]
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_generate_embeddings_respects_provider_batch_size() -> None:
    class BatchingProvider:
        model_id = "batching"
        supports_batching = True
        batch_size = 2

        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [np.asarray([1.0, 2.0], dtype=np.float32) for _ in texts]

        def embed_query(self, text: str):
            return np.asarray([1.0, 2.0], dtype=np.float32)

    provider = BatchingProvider()
    vectors = generate_embeddings(provider, ["a", "b", "c", "d", "e"])

    assert len(vectors) == 5
    assert provider.calls == [["a", "b"], ["c", "d"], ["e"]]


def test_resolve_local_model_path_prefers_explicit_file(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.gguf"
    explicit.write_bytes(b"gguf")

    resolved = resolve_local_model_path(
        tmp_path,
        tmp_path / ".cache",
        model_path=str(explicit),
        filename="coderankembed-q8_0.gguf",
    )

    assert resolved == explicit.resolve()


def test_resolve_local_model_path_uses_graph_indexer_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "graph-indexer" / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "coderankembed-q8_0.gguf"
    model_file.write_bytes(b"gguf")

    resolved = resolve_local_model_path(
        tmp_path,
        tmp_path / ".cache",
        model_path=None,
        filename="coderankembed-q8_0.gguf",
    )

    assert resolved == model_file.resolve()


def test_resolve_llama_cpp_provider_uses_existing_local_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_file = model_dir / "coderankembed-q8_0.gguf"
    model_file.write_bytes(b"gguf")

    provider = resolve_llama_cpp_provider(
        tmp_path,
        tmp_path / ".cache",
        model_path=None,
        filename=model_file.name,
    )

    assert provider.model_path == model_file.resolve()


def test_resolve_llama_cpp_provider_missing_model_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(embeddings_module, "DEFAULT_MODEL_FILENAMES", ("missing.gguf",))
    try:
        resolve_llama_cpp_provider(
            tmp_path,
            tmp_path / ".cache",
            model_path=None,
            filename="missing.gguf",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a RuntimeError for missing local model")

    assert "No local GGUF embedding model was found" in message
    assert "graph-indexer/models" in message
