from __future__ import annotations

from pathlib import Path

from ultimate_indexer.embeddings import HashEmbeddingProvider
from ultimate_indexer.formatter import format_groups
from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.models import ChunkRecord, FileRecord, QueryChunkHit, SymbolRecord
from ultimate_indexer.python_scip import emit_python_scip
from ultimate_indexer.query import QueryEngine


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


def test_query_collapses_field_hits_to_parent_struct_and_formats_cleanly(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    source = project / "models" / "query.go"
    source.parent.mkdir()
    source.write_text(
        "package models\n\ntype Table struct {\n    Name string `json:\"name\"`\n    Columns []Column `json:\"columns\"`\n}\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project, embedding_backend="hash")
    try:
        project_id = indexer.project_id
        relative_path = "models/query.go"
        file_symbol_id = f"file::{relative_path}"
        table_symbol_id = "scip-go gomod example `models`/Table#"
        field_symbol_id = "scip-go gomod example `models`/Table#Name."

        indexer.storage.replace_project_contents(
            project_id,
            files=[
                FileRecord(
                    project_id=project_id,
                    relative_path=relative_path,
                    abs_path=str(source),
                    language="go",
                    content_hash="file-hash",
                    content=source.read_text(encoding="utf-8"),
                )
            ],
            symbols=[
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=file_symbol_id,
                    scip_symbol=file_symbol_id,
                    display_name="query.go",
                    kind="File",
                    relative_path=relative_path,
                    start_line=1,
                    end_line=6,
                    signature=relative_path,
                    docstring="",
                    snippet=source.read_text(encoding="utf-8"),
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=table_symbol_id,
                    scip_symbol=table_symbol_id,
                    display_name="Table",
                    kind="TypeAlias",
                    relative_path=relative_path,
                    start_line=3,
                    end_line=6,
                    signature=table_symbol_id,
                    docstring="type Table struct\nTable represents a database table.",
                    snippet="type Table struct {\n    Name string\n    Columns []Column\n}",
                    enclosing_symbol_id=file_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=field_symbol_id,
                    scip_symbol=field_symbol_id,
                    display_name="Name",
                    kind="Field",
                    relative_path=relative_path,
                    start_line=4,
                    end_line=4,
                    signature=field_symbol_id,
                    docstring="struct field Name string",
                    snippet="    Name string `json:\"name\"`",
                    enclosing_symbol_id=table_symbol_id,
                ),
            ],
            edges=[],
            chunks=[
                ChunkRecord(
                    project_id=project_id,
                    chunk_id="field-chunk",
                    relative_path=relative_path,
                    symbol_id=field_symbol_id,
                    symbol_name="Name",
                    artifact_name=None,
                    chunk_kind="symbol",
                    start_line=4,
                    end_line=4,
                    content="name field table",
                    content_hash="field-chunk-hash",
                )
            ],
        )
        indexer.storage.upsert_project(project_id, project_id, "sig")

        engine = QueryEngine(indexer.storage, HashEmbeddingProvider())
        indexer.storage.search_bm25 = lambda project_id, query, limit: [  # type: ignore[method-assign]
            QueryChunkHit(
                chunk_id="field-chunk",
                relative_path=relative_path,
                symbol_id=field_symbol_id,
                symbol_name="Name",
                score=1.0,
                content="name field table",
                start_line=4,
                end_line=4,
            )
        ]
        engine._dense_hits = lambda project_id, query, limit: []  # type: ignore[method-assign]

        groups = engine.search(project_id, "table name", limit=5)
        assert groups
        assert groups[0].symbols[0].display_name == "Table"

        rendered = format_groups(indexer.storage, project_id, groups)
        assert "// models/query.go" in rendered
        assert "// Table represents a database table." in rendered
        assert "type Table struct" in rendered
        assert "Name string" in rendered
        assert '"""' not in rendered
        assert "scip-go gomod" not in rendered
        assert "interface omitted" not in rendered
    finally:
        indexer.close()
