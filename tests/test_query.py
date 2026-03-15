from __future__ import annotations

from pathlib import Path

from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.python_scip import emit_python_scip


def test_query_survives_embedding_backend_mismatch(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    python_files = sorted(fixture_project.rglob("*.py"))
    scip_path = fixture_project / ".ultimate_indexer" / "cache" / "fixture.scip"
    scip_path.parent.mkdir(parents=True, exist_ok=True)
    emit_python_scip(fixture_project, python_files, scip_path)
    first = UltimateIndexer(fixture_project, embedding_backend="hash")
    try:
        first.index(force=True, scip_path=scip_path)
    finally:
        first.close()

    second = UltimateIndexer(fixture_project, embedding_backend="llama-cpp")
    try:
        # Reuse the existing hash-built index and ensure the query path does not crash
        groups = second.query("greeting service", limit=5)
        assert isinstance(groups, list)
    finally:
        second.close()
