"""Regression tests for prod-readiness fixes.

Covers paths the existing suite left unguarded: incremental delete of removed
files, the auto-backend degrade when the llama-cpp runtime is absent, CLI/env
backend precedence, and config-change-forces-rebuild.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import ultimate_indexer.indexer as indexer_module
from ultimate_indexer.embeddings import HashEmbeddingProvider
from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.python_scip import emit_python_scip


def _emit_scip(project_root: Path) -> Path:
    python_files = sorted(project_root.rglob("*.py"))
    out = project_root / ".ultimate_indexer" / "cache" / "test.scip"
    out.parent.mkdir(parents=True, exist_ok=True)
    return emit_python_scip(project_root, python_files, out)


def test_incremental_reindex_purges_removed_file(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        # First index (no force) populates file hashes for incremental mode.
        indexer.index(scip_path=_emit_scip(fixture_project))
        assert indexer.storage.get_file(indexer.project_id, "pkg/services.py") is not None
        groups = indexer.query("greeting service", limit=10)
        assert any(g.relative_path == "pkg/services.py" for g in groups)

        # Delete a source file and re-index WITHOUT force -> incremental delete.
        (fixture_project / "pkg" / "services.py").unlink()
        indexer.index(scip_path=_emit_scip(fixture_project))

        # The removed file's file/symbol/query rows must all be purged.
        assert indexer.storage.get_file(indexer.project_id, "pkg/services.py") is None
        rows = indexer.storage.get_symbol_rows(indexer.project_id)
        assert not any(str(r["relative_path"]) == "pkg/services.py" for r in rows.values())
        groups_after = indexer.query("greeting service", limit=10)
        assert not any(g.relative_path == "pkg/services.py" for g in groups_after)
    finally:
        indexer.close()


def test_auto_backend_degrades_to_hash_without_llama_cpp(fixture_project: Path, monkeypatch) -> None:
    # A model file is present but the llama_cpp runtime is not importable: the
    # auto backend must degrade to hash instead of crashing at embed time.
    monkeypatch.delenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", raising=False)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        indexer_module.importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None
        if name == "llama_cpp"
        else real_find_spec(name, *args, **kwargs),
    )
    model_dir = fixture_project / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "coderankembed-q8_0.gguf").write_bytes(b"gguf")

    indexer = UltimateIndexer(fixture_project, embedding_backend="auto")
    try:
        indexer.index(scip_path=_emit_scip(fixture_project))  # must not raise
        assert isinstance(indexer._provider_instance(), HashEmbeddingProvider)
    finally:
        indexer.close()


def test_explicit_llama_backend_without_runtime_errors_clearly(fixture_project: Path, monkeypatch) -> None:
    # Explicitly asking for the native backend with no runtime must fail loudly,
    # not silently degrade.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        indexer_module.importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None
        if name == "llama_cpp"
        else real_find_spec(name, *args, **kwargs),
    )
    indexer = UltimateIndexer(fixture_project, embedding_backend="local")
    try:
        try:
            indexer._provider_instance()
        except RuntimeError as exc:
            assert "llama-cpp-python" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for explicit native backend without runtime")
    finally:
        indexer.close()


def test_cli_default_backend_honors_env_var(fixture_project: Path, monkeypatch) -> None:
    # The CLI passes embedding_backend=None when --embedding-backend is omitted;
    # the documented env var must then win (it used to be shadowed by "auto").
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project, embedding_backend=None)
    try:
        assert indexer.settings.embedding_backend == "hash"
        assert isinstance(indexer._provider_instance(), HashEmbeddingProvider)
    finally:
        indexer.close()


def test_config_change_forces_full_rebuild(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_emit_scip(fixture_project))
        # A config-only change (chunk params) must rebuild fully even though no
        # file content changed -> no incremental reuse.
        indexer.settings.max_chunk_lines = indexer.settings.max_chunk_lines + 7
        summary = indexer.index(scip_path=_emit_scip(fixture_project))
        assert summary.reused_files == 0
    finally:
        indexer.close()


def test_unchanged_reindex_stays_incremental(fixture_project: Path, monkeypatch) -> None:
    # Guard the inverse of the config-change test: with nothing changed, the
    # config-signature check must NOT force a wasteful full rebuild.
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_emit_scip(fixture_project))
        summary = indexer.index(scip_path=_emit_scip(fixture_project))
        assert summary.reused_files >= 3
    finally:
        indexer.close()
