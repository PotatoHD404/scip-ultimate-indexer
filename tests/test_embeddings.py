from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ultimate_indexer.embeddings import LlamaCppEmbeddingProvider, generate_embeddings


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
    assert calls[1] == "search_query: hello world"
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
