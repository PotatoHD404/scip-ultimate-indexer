from __future__ import annotations

from pathlib import Path

from ultimate_indexer.formatter import format_groups, format_top_symbols
from ultimate_indexer.indexer import UltimateIndexer


def test_index_query_and_cache(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        summary = indexer.index()
        assert summary.indexed_files >= 4
        assert summary.indexed_symbols >= 6
        assert summary.indexed_edges >= 4

        groups = indexer.query("greeting service user", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// pkg/services.py" in rendered
        assert "class GreetingService:" in rendered
        assert "def build_greeting" in rendered

        second = indexer.index()
        assert second.reused_files >= 3
    finally:
        indexer.close()


def test_top_symbols_and_visualization(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index()
        top_rows = indexer.top_symbols(limit=5)
        rendered = format_top_symbols(top_rows)
        assert "GreetingService" in rendered or "build_greeting" in rendered

        groups = indexer.query("greeting", limit=5)
        output_path = indexer.visualize(groups, title="Greeting graph")
        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "vis-network" in html
        assert "Greeting graph" in html
    finally:
        indexer.close()


def test_socraticode_artifact_query(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index()
        groups = indexer.query("tenants database schema", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// docs/schema.md" in rendered
        assert "database-schema" in rendered or "Database and architecture context" in rendered
    finally:
        indexer.close()
