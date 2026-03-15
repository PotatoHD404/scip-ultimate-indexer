from __future__ import annotations

from pathlib import Path

from ultimate_indexer.indexer import UltimateIndexer


def test_query_survives_embedding_backend_mismatch(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    first = UltimateIndexer(fixture_project, embedding_backend="hash")
    try:
        first.index(force=True)
    finally:
        first.close()

    second = UltimateIndexer(fixture_project, embedding_backend="llama-cpp")
    try:
        # Reuse the existing hash-built index and ensure the query path does not crash
        groups = second.query("greeting service", limit=5)
        assert isinstance(groups, list)
    finally:
        second.close()
