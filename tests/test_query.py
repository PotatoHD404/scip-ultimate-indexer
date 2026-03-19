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
        "package models\n\ntype Table struct {\n    Name string `json:\"name\"`\n    Columns []Column `json:\"columns\"`\n}\n\nfunc (t *Table) Validate() error {\n    return nil\n}\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project, embedding_backend="hash")
    try:
        project_id = indexer.project_id
        relative_path = "models/query.go"
        file_symbol_id = f"file::{relative_path}"
        table_symbol_id = "scip-go gomod example `models`/Table#"
        field_symbol_id = "scip-go gomod example `models`/Table#Name."
        columns_symbol_id = "scip-go gomod example `models`/Table#Columns."
        method_symbol_id = "scip-go gomod example `models`/Table#Validate()."

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
                    snippet="type Table struct",
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
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=columns_symbol_id,
                    scip_symbol=columns_symbol_id,
                    display_name="Columns",
                    kind="Field",
                    relative_path=relative_path,
                    start_line=5,
                    end_line=5,
                    signature=columns_symbol_id,
                    docstring="struct field Columns []Column",
                    snippet="    Columns []Column `json:\"columns\"`",
                    enclosing_symbol_id=table_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=method_symbol_id,
                    scip_symbol=method_symbol_id,
                    display_name="Validate",
                    kind="Method",
                    relative_path=relative_path,
                    start_line=8,
                    end_line=10,
                    signature="func (t *Table) Validate() error",
                    docstring="func (t *Table) Validate() error",
                    snippet="func (t *Table) Validate() error {\n    return nil\n}",
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
        assert "type Table struct {" in rendered
        assert "    Name string" in rendered
        assert "    Columns []Column" in rendered
        assert "\n\nfunc (t *Table) Validate() error" in rendered
        assert "    return nil" in rendered
        assert '"""' not in rendered
        assert "scip-go gomod" not in rendered
        assert "interface omitted" not in rendered
    finally:
        indexer.close()


def test_query_renders_go_interface_members_inside_interface_block(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    source = project / "models" / "reader.go"
    source.parent.mkdir()
    source.write_text(
        "package models\n\ntype Reader interface {\n    Read(ctx context.Context, key string) error\n}\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project, embedding_backend="hash")
    try:
        project_id = indexer.project_id
        relative_path = "models/reader.go"
        file_symbol_id = f"file::{relative_path}"
        interface_symbol_id = "scip-go gomod example `models`/Reader#"
        method_symbol_id = "scip-go gomod example `models`/Reader#Read()."

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
                    display_name="reader.go",
                    kind="File",
                    relative_path=relative_path,
                    start_line=1,
                    end_line=5,
                    signature=relative_path,
                    docstring="",
                    snippet=source.read_text(encoding="utf-8"),
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=interface_symbol_id,
                    scip_symbol=interface_symbol_id,
                    display_name="Reader",
                    kind="Interface",
                    relative_path=relative_path,
                    start_line=3,
                    end_line=5,
                    signature=interface_symbol_id,
                    docstring="type Reader interface",
                    snippet="type Reader interface",
                    enclosing_symbol_id=file_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=method_symbol_id,
                    scip_symbol=method_symbol_id,
                    display_name="Read",
                    kind="Method",
                    relative_path=relative_path,
                    start_line=4,
                    end_line=4,
                    signature="Read(ctx context.Context, key string) error",
                    docstring="func (Reader).Read(ctx context.Context, key string) error",
                    snippet="Read(ctx context.Context, key string) error",
                    enclosing_symbol_id=interface_symbol_id,
                ),
            ],
            edges=[],
            chunks=[
                ChunkRecord(
                    project_id=project_id,
                    chunk_id="interface-chunk",
                    relative_path=relative_path,
                    symbol_id=method_symbol_id,
                    symbol_name="Read",
                    artifact_name=None,
                    chunk_kind="symbol",
                    start_line=4,
                    end_line=4,
                    content="reader interface read context key",
                    content_hash="interface-chunk-hash",
                )
            ],
        )
        indexer.storage.upsert_project(project_id, project_id, "sig")

        engine = QueryEngine(indexer.storage, HashEmbeddingProvider())
        indexer.storage.search_bm25 = lambda project_id, query, limit: [  # type: ignore[method-assign]
            QueryChunkHit(
                chunk_id="interface-chunk",
                relative_path=relative_path,
                symbol_id=method_symbol_id,
                symbol_name="Read",
                score=1.0,
                content="reader interface read context key",
                start_line=4,
                end_line=4,
            )
        ]
        engine._dense_hits = lambda project_id, query, limit: []  # type: ignore[method-assign]

        groups = engine.search(project_id, "reader read context", limit=5)
        assert groups
        assert groups[0].symbols[0].display_name == "Reader"

        rendered = format_groups(indexer.storage, project_id, groups)
        assert "type Reader interface {" in rendered
        assert "    Read(ctx context.Context, key string) error" in rendered
        assert "\n\nfunc " not in rendered
    finally:
        indexer.close()


def test_query_renders_go_type_with_raw_signature_without_variable_noise(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    source = project / "internal" / "api" / "rest" / "dto" / "audience" / "v2" / "update.go"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package v2\n\n// UpdateStaticParametersReq is a dto for update static parameters.\ntype UpdateStaticParametersReq struct {\n    Name string\n    Value string\n}\n\nfunc (UpdateStaticParametersReq) Validate() error {\n    return nil\n}\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project, embedding_backend="hash")
    try:
        project_id = indexer.project_id
        relative_path = "internal/api/rest/dto/audience/v2/update.go"
        file_symbol_id = f"file::{relative_path}"
        type_symbol_id = "scip-go gomod example `internal/api/rest/dto/audience/v2`/UpdateStaticParametersReq#"
        method_symbol_id = "scip-go gomod example `internal/api/rest/dto/audience/v2`/UpdateStaticParametersReq#Validate()."
        field_symbol_id = "scip-go gomod example `internal/api/rest/dto/audience/v2`/UpdateStaticParametersReq#Name."
        variable_symbol_id = "local::internal/api/rest/dto/audience/v2/update.go::local 3"

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
                    display_name="update.go",
                    kind="File",
                    relative_path=relative_path,
                    start_line=1,
                    end_line=10,
                    signature=relative_path,
                    docstring="",
                    snippet=source.read_text(encoding="utf-8"),
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=type_symbol_id,
                    scip_symbol=type_symbol_id,
                    display_name="UpdateStaticParametersReq",
                    kind="TypeAlias",
                    relative_path=relative_path,
                    start_line=4,
                    end_line=7,
                    signature=type_symbol_id,
                    docstring="UpdateStaticParametersReq is a dto for update static parameters.",
                    snippet="UpdateStaticParametersReq is a dto for update static parameters.",
                    enclosing_symbol_id=file_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=field_symbol_id,
                    scip_symbol=field_symbol_id,
                    display_name="Name",
                    kind="Field",
                    relative_path=relative_path,
                    start_line=5,
                    end_line=5,
                    signature=field_symbol_id,
                    docstring="struct field Name string",
                    snippet="    Name string",
                    enclosing_symbol_id=type_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=method_symbol_id,
                    scip_symbol=method_symbol_id,
                    display_name="Validate",
                    kind="Method",
                    relative_path=relative_path,
                    start_line=8,
                    end_line=10,
                    signature="func (UpdateStaticParametersReq) Validate() error",
                    docstring="func (UpdateStaticParametersReq).Validate() error",
                    snippet="func (UpdateStaticParametersReq) Validate() error {\n    return nil\n}",
                    enclosing_symbol_id=type_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=variable_symbol_id,
                    scip_symbol="local 3",
                    display_name="sp",
                    kind="Variable",
                    relative_path=relative_path,
                    start_line=8,
                    end_line=8,
                    signature="local 3",
                    docstring="var sp gitlab.tcsbank.ru/t-segments/bff/internal/api/rest/dto/audience/v2.UpdateStaticParametersReq",
                    snippet="var sp UpdateStaticParametersReq",
                    enclosing_symbol_id=type_symbol_id,
                ),
            ],
            edges=[],
            chunks=[
                ChunkRecord(
                    project_id=project_id,
                    chunk_id="type-chunk",
                    relative_path=relative_path,
                    symbol_id=type_symbol_id,
                    symbol_name="UpdateStaticParametersReq",
                    artifact_name=None,
                    chunk_kind="symbol",
                    start_line=4,
                    end_line=7,
                    content="update static parameters validate dto",
                    content_hash="type-chunk-hash",
                )
            ],
        )
        indexer.storage.upsert_project(project_id, project_id, "sig")

        engine = QueryEngine(indexer.storage, HashEmbeddingProvider())
        indexer.storage.search_bm25 = lambda project_id, query, limit: [  # type: ignore[method-assign]
            QueryChunkHit(
                chunk_id="type-chunk",
                relative_path=relative_path,
                symbol_id=type_symbol_id,
                symbol_name="UpdateStaticParametersReq",
                score=1.0,
                content="update static parameters validate dto",
                start_line=4,
                end_line=7,
            )
        ]
        engine._dense_hits = lambda project_id, query, limit: []  # type: ignore[method-assign]

        groups = engine.search(project_id, "update static parameters validate", limit=5)
        assert groups
        rendered = format_groups(indexer.storage, project_id, groups)
        assert rendered.count("// UpdateStaticParametersReq is a dto for update static parameters.") == 1
        assert "type UpdateStaticParametersReq struct {" in rendered
        assert "    Name string" in rendered
        assert "func (UpdateStaticParametersReq) Validate() error" in rendered
        assert "var sp " not in rendered
        assert "gitlab.tcsbank.ru/t-segments" not in rendered
    finally:
        indexer.close()


def test_query_collapses_local_variable_hit_to_function_and_shows_skipped_lines(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    source = project / "internal" / "postgres" / "online_segments_core.go"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package postgres\n\nfunc getOnlineSegment(id int64) (*segment, error) {\n    var err error\n    value := id + 1\n    _ = value\n    return nil, err\n}\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project, embedding_backend="hash")
    try:
        project_id = indexer.project_id
        relative_path = "internal/postgres/online_segments_core.go"
        file_symbol_id = f"file::{relative_path}"
        function_symbol_id = "scip-go gomod example `internal/postgres`/Repository#getOnlineSegment()."
        local_symbol_id = "local::internal/postgres/online_segments_core.go::local 1"

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
                    display_name="online_segments_core.go",
                    kind="File",
                    relative_path=relative_path,
                    start_line=1,
                    end_line=8,
                    signature=relative_path,
                    docstring="",
                    snippet=source.read_text(encoding="utf-8"),
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=function_symbol_id,
                    scip_symbol=function_symbol_id,
                    display_name="getOnlineSegment",
                    kind="Function",
                    relative_path=relative_path,
                    start_line=3,
                    end_line=7,
                    signature="func getOnlineSegment(id int64) (*segment, error)",
                    docstring="func getOnlineSegment(id int64) (*segment, error)",
                    snippet="func getOnlineSegment(id int64) (*segment, error) {\n    var err error\n    value := id + 1\n    _ = value\n    return nil, err\n}",
                    enclosing_symbol_id=file_symbol_id,
                ),
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=local_symbol_id,
                    scip_symbol="local 1",
                    display_name="err",
                    kind="Variable",
                    relative_path=relative_path,
                    start_line=4,
                    end_line=4,
                    signature="local 1",
                    docstring="var err error",
                    snippet="    var err error",
                    enclosing_symbol_id=function_symbol_id,
                ),
            ],
            edges=[],
            chunks=[
                ChunkRecord(
                    project_id=project_id,
                    chunk_id="local-chunk",
                    relative_path=relative_path,
                    symbol_id=local_symbol_id,
                    symbol_name="err",
                    artifact_name=None,
                    chunk_kind="symbol",
                    start_line=4,
                    end_line=4,
                    content="err local online segment",
                    content_hash="local-chunk-hash",
                )
            ],
        )
        indexer.storage.upsert_project(project_id, project_id, "sig")

        engine = QueryEngine(indexer.storage, HashEmbeddingProvider())
        indexer.storage.search_bm25 = lambda project_id, query, limit: [  # type: ignore[method-assign]
            QueryChunkHit(
                chunk_id="local-chunk",
                relative_path=relative_path,
                symbol_id=local_symbol_id,
                symbol_name="err",
                score=1.0,
                content="err local online segment",
                start_line=4,
                end_line=4,
            )
        ]
        engine._dense_hits = lambda project_id, query, limit: []  # type: ignore[method-assign]

        groups = engine.search(project_id, "online segment err", limit=5)
        assert groups
        assert groups[0].symbols[0].display_name == "getOnlineSegment"

        rendered = format_groups(indexer.storage, project_id, groups)
        assert "func getOnlineSegment(id int64) (*segment, error)" in rendered
        assert "    var err error" in rendered
        assert "    value := id + 1" in rendered
        assert "// skipped 2 rows" in rendered
        assert "\nvar err error\n" not in rendered
    finally:
        indexer.close()
