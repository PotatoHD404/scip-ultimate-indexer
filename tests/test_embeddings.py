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
    resolve_llama_cpp_provider,
    resolve_sentence_transformer_provider,
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
        def __init__(
            self,
            model_name: str,
            cache_folder: str,
            trust_remote_code: bool,
            device: str,
            local_files_only: bool = False,
        ) -> None:
            calls.append(("init", model_name, cache_folder, trust_remote_code, device, local_files_only))

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
        max_seq_length=512,
        use_fp16=True,
    )
    vector = provider.embed_query("find auth code")

    assert vector.shape == (3,)
    assert ("init", "nomic-ai/CodeRankEmbed", "/tmp/models", True, "mps", False) in calls
    assert "half" in calls
    assert ("encode", [f"{CODERANK_QUERY_PREFIX}find auth code"], 96, True) in calls


def test_sentence_transformer_provider_caps_max_seq_length(monkeypatch) -> None:
    captured: dict[str, object] = {}

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

    class FakeTokenizer:
        model_max_length = 8192

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs) -> None:
            self.max_seq_length = 8192
            self.tokenizer = FakeTokenizer()
            captured["model"] = self

        def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
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
        batch_size=8,
        max_seq_length=512,
    )
    provider.embed(["abc"])

    model = captured["model"]
    assert model.max_seq_length == 512
    assert model.tokenizer.model_max_length == 512


def test_sentence_transformer_provider_wraps_mps_memory_error(monkeypatch) -> None:
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
            self.max_seq_length = 8192
            self.tokenizer = types.SimpleNamespace(model_max_length=8192)

        def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
            raise RuntimeError("Invalid buffer size: 48.00 GiB")

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
        batch_size=8,
        max_seq_length=512,
    )

    try:
        provider.embed(["abc"])
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a RuntimeError for MPS memory exhaustion")

    assert "Apple GPU memory" in message
    assert "ULTIMATE_INDEXER_ST_BATCH_SIZE=2" in message
    assert "ULTIMATE_INDEXER_ST_MAX_SEQ_LENGTH=256" in message


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


def test_resolve_sentence_transformer_provider_uses_cached_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_ST_DEVICE", "cpu")

    fake_hf = types.SimpleNamespace(
        snapshot_download=lambda repo_id, cache_dir, local_files_only: str(tmp_path / "snapshots" / "coderank")
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    provider = resolve_sentence_transformer_provider(tmp_path, "nomic-ai/CodeRankEmbed")

    assert provider.model_id == "nomic-ai/CodeRankEmbed"
    assert provider.model_name == str(tmp_path / "snapshots" / "coderank")
    assert provider.local_files_only is True


def test_resolve_sentence_transformer_provider_offline_missing_cache_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_ST_DEVICE", "cpu")
    monkeypatch.setenv("ULTIMATE_INDEXER_HF_LOCAL_ONLY", "true")

    def missing_snapshot(repo_id, cache_dir, local_files_only):
        raise RuntimeError("not cached")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=missing_snapshot))

    try:
        resolve_sentence_transformer_provider(tmp_path, "nomic-ai/CodeRankEmbed")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a RuntimeError for offline cache miss")

    assert "local model cache" in message
    assert "network access is disabled" in message


def test_resolve_llama_cpp_provider_uses_existing_local_file(monkeypatch, tmp_path: Path) -> None:
    model_file = tmp_path / "nomic-embed-code-Q4_K_M.gguf"
    model_file.write_bytes(b"gguf")
    calls: list[tuple[str, str, str]] = []

    def fake_download(repo_id, filename, local_dir):
        calls.append((repo_id, filename, local_dir))
        return str(tmp_path / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))

    provider = resolve_llama_cpp_provider(tmp_path, "repo/model", model_file.name)

    assert provider.model_path == model_file
    assert calls == []


def test_resolve_llama_cpp_provider_offline_missing_cache_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_HF_LOCAL_ONLY", "true")

    try:
        resolve_llama_cpp_provider(tmp_path, "repo/model", "missing.gguf")
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected a RuntimeError for offline cache miss")

    assert "local model cache" in message
    assert "network access is disabled" in message
