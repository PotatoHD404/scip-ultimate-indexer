from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from ultimate_indexer.embeddings import (
    CODERANK_QUERY_PREFIX,
    LlamaCppEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    generate_embeddings,
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


def test_sentence_transformer_provider_uses_query_prefix(monkeypatch) -> None:
    calls: list[object] = []

    class FakeTorch:
        @staticmethod
        def inference_mode():
            class _Ctx:
                def __enter__(self_inner):
                    calls.append("enter")
                    return None

                def __exit__(self_inner, exc_type, exc, tb):
                    calls.append("exit")
                    return False

            return _Ctx()

        @staticmethod
        def set_float32_matmul_precision(mode: str) -> None:
            calls.append(("matmul", mode))

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, cache_folder: str, trust_remote_code: bool, device: str) -> None:
            calls.append(("init", model_name, cache_folder, trust_remote_code, device))

        def half(self) -> None:
            calls.append("half")

        def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
            calls.append(("encode", list(texts), batch_size, normalize_embeddings))
            return np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = SentenceTransformerEmbeddingProvider(
        model_name="nomic-ai/CodeRankEmbed",
        cache_dir=Path("/tmp/models"),
        model_id="nomic-ai/CodeRankEmbed",
        device="mps",
        batch_size=96,
        use_fp16=True,
    )
    vector = provider.embed_query("find auth code")

    assert vector.shape == (3,)
    assert ("init", "nomic-ai/CodeRankEmbed", "/tmp/models", True, "mps") in calls
    assert "half" in calls
    assert ("encode", [f"{CODERANK_QUERY_PREFIX}find auth code"], 96, True) in calls


def test_sentence_transformer_provider_reports_missing_remote_dependency(monkeypatch) -> None:
    class FakeTorch:
        @staticmethod
        def inference_mode():
            class _Ctx:
                def __enter__(self_inner):
                    return None

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

        @staticmethod
        def set_float32_matmul_precision(mode: str) -> None:
            return None

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "This modeling file requires the following packages that were not found "
                "in your environment: einops. Run `pip install einops`"
            )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = SentenceTransformerEmbeddingProvider(
        model_name="nomic-ai/CodeRankEmbed",
        cache_dir=Path("/tmp/models"),
        model_id="nomic-ai/CodeRankEmbed",
        device="mps",
    )

    try:
        provider.embed_query("find auth code")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a RuntimeError for missing remote dependency")

    assert "nomic-ai/CodeRankEmbed" in message
    assert "einops" in message
    assert "poetry add einops" in message
