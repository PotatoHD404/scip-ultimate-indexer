from __future__ import annotations

from pathlib import Path

import ultimate_indexer.indexer as indexer_module
from ultimate_indexer.formatter import format_groups, format_top_symbols
from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.models import EdgeRecord, FileRecord, IndexProgress, SymbolRecord
from ultimate_indexer.python_scip import emit_python_scip
from ultimate_indexer.scip_parser import ParsedScip
from ultimate_indexer.scip_runner import ScipRunFailure, ScipRunReport, ScipRunResult


def _python_scip_path(project_root: Path) -> Path:
    python_files = sorted(project_root.rglob("*.py"))
    output_path = project_root / ".ultimate_indexer" / "cache" / "fixture.scip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return emit_python_scip(project_root, python_files, output_path)


def test_index_query_and_cache(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        summary = indexer.index(scip_path=_python_scip_path(fixture_project))
        assert summary.indexed_files >= 4
        assert summary.indexed_symbols >= 6
        assert summary.indexed_edges >= 4

        groups = indexer.query("greeting service user", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// pkg/services.py" in rendered
        assert "GreetingService" in rendered or "def build_greeting" in rendered
        assert "def build_greeting" in rendered

        second = indexer.index(scip_path=_python_scip_path(fixture_project))
        assert second.reused_files >= 3
    finally:
        indexer.close()


def test_top_symbols_and_visualization(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_python_scip_path(fixture_project))
        top_rows = indexer.top_symbols(limit=5)
        rendered = format_top_symbols(top_rows)
        assert "GreetingService" in rendered or "build_greeting" in rendered

        groups = indexer.query("greeting", limit=5)
        output_path = indexer.visualize(groups, title="Greeting graph")
        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "force-graph" in html
        assert "Greeting graph" in html
        assert "pkg/services.py::GreetingService" in html
    finally:
        indexer.close()


def test_socraticode_artifact_query(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_python_scip_path(fixture_project))
        groups = indexer.query("tenants database schema", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// docs/schema.md" in rendered
        assert "database-schema" in rendered or "Database and architecture context" in rendered
    finally:
        indexer.close()


def test_index_reports_progress(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    events: list[IndexProgress] = []
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_python_scip_path(fixture_project), progress_callback=events.append)
        assert any(event.stage == "discover" for event in events)
        assert any(event.stage == "embed" and event.total > 0 for event in events)
        assert any(event.stage == "embed" and event.detail == "Loading embedding model" for event in events)
        assert events[-1].stage == "pagerank"
    finally:
        indexer.close()


def test_reindex_only_reembeds_changed_chunks(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    scip_path = _python_scip_path(fixture_project)
    indexer = UltimateIndexer(fixture_project)
    generate_calls: list[int] = []
    original_generate_embeddings = indexer_module.generate_embeddings

    def capture_generate_embeddings(provider, texts, on_batch_complete=None, batch_size=32, sleep_fn=None):
        generate_calls.append(len(texts))
        if sleep_fn is None:
            from time import sleep as sleep_fn_value
        else:
            sleep_fn_value = sleep_fn
        return original_generate_embeddings(
            provider,
            texts,
            on_batch_complete=on_batch_complete,
            batch_size=batch_size,
            sleep_fn=sleep_fn_value,
        )

    monkeypatch.setattr("ultimate_indexer.indexer.generate_embeddings", capture_generate_embeddings)
    try:
        indexer.index(scip_path=scip_path, force=True)
        first_embed_count = sum(generate_calls)
        assert first_embed_count > 0

        before_rows = indexer.storage.connection.execute(
            "SELECT chunk_id, content_hash FROM chunks WHERE project_id = ?",
            (indexer.project_id,),
        ).fetchall()
        before_hashes = {str(row["chunk_id"]): str(row["content_hash"]) for row in before_rows}

        target = fixture_project / "pkg" / "services.py"
        original = target.read_text(encoding="utf-8")
        target.write_text(original.replace("Hello", "Greetings"), encoding="utf-8")

        generate_calls.clear()
        indexer.index(scip_path=_python_scip_path(fixture_project), force=True)
        second_embed_count = sum(generate_calls)

        after_rows = indexer.storage.connection.execute(
            "SELECT chunk_id, content_hash FROM chunks WHERE project_id = ?",
            (indexer.project_id,),
        ).fetchall()
        changed_chunk_count = sum(
            1
            for row in after_rows
            if before_hashes.get(str(row["chunk_id"])) != str(row["content_hash"])
        )

        assert changed_chunk_count > 0
        assert second_embed_count == changed_chunk_count
        assert second_embed_count < first_embed_count
    finally:
        indexer.close()


def test_index_continues_with_fallback_when_scip_root_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    target = project / "src" / "auth.ts"
    target.write_text("export function buildToken(userId: string) { return `token:${userId}` }\n", encoding="utf-8")

    monkeypatch.setattr(
        "ultimate_indexer.indexer.run_scip_indexers",
        lambda project_root, files, cache_dir: ScipRunReport(
            results=[],
            missing=[],
            failed=[
                ScipRunFailure(
                    language="typescript",
                    binary_name="scip-typescript",
                    install_hint="npm install -g @sourcegraph/scip-typescript",
                    working_directory=str(project / "src"),
                    command=("scip-typescript", "index"),
                    detail="missing tsconfig.json",
                )
            ],
        ),
    )

    indexer = UltimateIndexer(project)
    try:
        summary = indexer.index(force=True)
        assert summary.warnings
        assert "typescript at src" in summary.warnings[0]
        groups = indexer.query("buildToken token", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// src/auth.ts" in rendered
        assert "buildToken" in rendered
    finally:
        indexer.close()


def test_index_dedupes_overlapping_scip_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    source = project / "src"
    source.mkdir()
    target = source / "auth.ts"
    target.write_text("export const token = 'abc'\n", encoding="utf-8")

    scip_a = project / ".ultimate_indexer" / "cache" / "a.scip"
    scip_b = project / ".ultimate_indexer" / "cache" / "b.scip"
    scip_a.parent.mkdir(parents=True, exist_ok=True)
    scip_a.write_bytes(b"")
    scip_b.write_bytes(b"")

    file_record = FileRecord(
        project_id=str(project),
        relative_path="src/auth.ts",
        abs_path=str(target),
        language="ts",
        content_hash="hash",
        content=target.read_text(encoding="utf-8"),
    )
    file_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="file::src/auth.ts",
        scip_symbol="file::src/auth.ts",
        display_name="auth.ts",
        kind="File",
        relative_path="src/auth.ts",
        start_line=1,
        end_line=1,
        signature="src/auth.ts",
        docstring="",
        snippet="export const token = 'abc'",
    )
    value_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="scip-typescript npm demo 1.0 `auth.ts`/token.",
        scip_symbol="scip-typescript npm demo 1.0 `auth.ts`/token.",
        display_name="token",
        kind="Constant",
        relative_path="src/auth.ts",
        start_line=1,
        end_line=1,
        signature="token",
        docstring="",
        snippet="export const token = 'abc'",
        enclosing_symbol_id="file::src/auth.ts",
    )
    edge_record = EdgeRecord(
        project_id=str(project),
        source_symbol_id="file::src/auth.ts",
        target_symbol_id=value_symbol.symbol_id,
        edge_type="contains",
        weight=0.55,
    )

    monkeypatch.setattr(
        "ultimate_indexer.indexer.run_scip_indexers",
        lambda project_root, files, cache_dir: ScipRunReport(
            results=[
                ScipRunResult(language="typescript", index_path=scip_a, source_root=project),
                ScipRunResult(language="typescript", index_path=scip_b, source_root=project),
            ],
            missing=[],
            failed=[],
        ),
    )
    monkeypatch.setattr(
        "ultimate_indexer.indexer.parse_scip_index",
        lambda project_id, project_root, index_path, edge_weights, source_root=None: ParsedScip(
            files=[file_record],
            symbols=[file_symbol, value_symbol],
            edges=[edge_record],
        ),
    )

    indexer = UltimateIndexer(project)
    try:
        summary = indexer.index(force=True)
        assert summary.indexed_files == 1
        assert summary.indexed_symbols == 2
        assert summary.indexed_edges >= 1
        rows = indexer.storage.get_symbol_rows(indexer.project_id)
        assert len(rows) == 2
        assert "scip-typescript npm demo 1.0 `auth.ts`/token." in rows
    finally:
        indexer.close()


def test_generated_and_unknown_symbols_are_excluded_from_ranking_and_query(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "proto").mkdir()

    regular_file_path = project / "src" / "service.ts"
    generated_file_path = project / "proto" / "generated.ts"
    unknown_file_path = project / "src" / "unknown.ts"
    regular_file_path.write_text("export function handleAuth() { return 'ok' }\n", encoding="utf-8")
    generated_file_path.write_text("export interface GeneratedRecord { value: string }\n", encoding="utf-8")
    unknown_file_path.write_text("export const secretFlag = true\n", encoding="utf-8")

    scip_path = project / ".ultimate_indexer" / "cache" / "fixture.scip"
    scip_path.parent.mkdir(parents=True, exist_ok=True)
    scip_path.write_bytes(b"")

    regular_file = FileRecord(
        project_id=str(project),
        relative_path="src/service.ts",
        abs_path=str(regular_file_path),
        language="ts",
        content_hash="regular-file",
        content=regular_file_path.read_text(encoding="utf-8"),
    )
    generated_file = FileRecord(
        project_id=str(project),
        relative_path="proto/generated.ts",
        abs_path=str(generated_file_path),
        language="ts",
        content_hash="generated-file",
        content=generated_file_path.read_text(encoding="utf-8"),
    )
    unknown_file = FileRecord(
        project_id=str(project),
        relative_path="src/unknown.ts",
        abs_path=str(unknown_file_path),
        language="ts",
        content_hash="unknown-file",
        content=unknown_file_path.read_text(encoding="utf-8"),
    )

    regular_file_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="file::src/service.ts",
        scip_symbol="file::src/service.ts",
        display_name="service.ts",
        kind="File",
        relative_path="src/service.ts",
        start_line=1,
        end_line=1,
        signature="src/service.ts",
        docstring="",
        snippet="export function handleAuth() { return 'ok' }",
    )
    generated_file_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="file::proto/generated.ts",
        scip_symbol="file::proto/generated.ts",
        display_name="generated.ts",
        kind="File",
        relative_path="proto/generated.ts",
        start_line=1,
        end_line=1,
        signature="proto/generated.ts",
        docstring="",
        snippet="export interface GeneratedRecord { value: string }",
    )
    unknown_file_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="file::src/unknown.ts",
        scip_symbol="file::src/unknown.ts",
        display_name="unknown.ts",
        kind="File",
        relative_path="src/unknown.ts",
        start_line=1,
        end_line=1,
        signature="src/unknown.ts",
        docstring="",
        snippet="export const secretFlag = true",
    )

    regular_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="regular::handleAuth",
        scip_symbol="regular::handleAuth",
        display_name="handleAuth",
        kind="Function",
        relative_path="src/service.ts",
        start_line=1,
        end_line=1,
        signature="function handleAuth(): string",
        docstring="",
        snippet="export function handleAuth() { return 'ok' }",
        enclosing_symbol_id="file::src/service.ts",
    )
    generated_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="generated::GeneratedRecord",
        scip_symbol="generated::GeneratedRecord",
        display_name="GeneratedRecord",
        kind="Interface",
        relative_path="proto/generated.ts",
        start_line=1,
        end_line=1,
        signature="interface GeneratedRecord",
        docstring="",
        snippet="export interface GeneratedRecord { value: string }",
        enclosing_symbol_id="file::proto/generated.ts",
    )
    unknown_symbol = SymbolRecord(
        project_id=str(project),
        symbol_id="unknown::secretFlag",
        scip_symbol="unknown::secretFlag",
        display_name="secretFlag",
        kind="Unknown",
        relative_path="src/unknown.ts",
        start_line=1,
        end_line=1,
        signature="const secretFlag: boolean",
        docstring="",
        snippet="export const secretFlag = true",
        enclosing_symbol_id="file::src/unknown.ts",
    )

    edges = [
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="file::src/service.ts",
            target_symbol_id="regular::handleAuth",
            edge_type="contains",
            weight=0.55,
        ),
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="file::proto/generated.ts",
            target_symbol_id="generated::GeneratedRecord",
            edge_type="contains",
            weight=0.55,
        ),
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="file::src/unknown.ts",
            target_symbol_id="unknown::secretFlag",
            edge_type="contains",
            weight=0.55,
        ),
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="regular::handleAuth",
            target_symbol_id="generated::GeneratedRecord",
            edge_type="uses",
            weight=0.75,
        ),
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="generated::GeneratedRecord",
            target_symbol_id="generated::GeneratedRecord",
            edge_type="uses",
            weight=0.75,
        ),
        EdgeRecord(
            project_id=str(project),
            source_symbol_id="unknown::secretFlag",
            target_symbol_id="regular::handleAuth",
            edge_type="uses",
            weight=0.75,
        ),
    ]

    monkeypatch.setattr(
        "ultimate_indexer.indexer.run_scip_indexers",
        lambda project_root, files, cache_dir: ScipRunReport(
            results=[ScipRunResult(language="typescript", index_path=scip_path, source_root=project)],
            missing=[],
            failed=[],
        ),
    )
    monkeypatch.setattr(
        "ultimate_indexer.indexer.parse_scip_index",
        lambda project_id, project_root, index_path, edge_weights, source_root=None: ParsedScip(
            files=[regular_file, generated_file, unknown_file],
            symbols=[
                regular_file_symbol,
                generated_file_symbol,
                unknown_file_symbol,
                regular_symbol,
                generated_symbol,
                unknown_symbol,
            ],
            edges=edges,
        ),
    )

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        top_rows = indexer.top_symbols(limit=10)
        top_ids = {str(row["symbol_id"]) for row in top_rows}
        assert "regular::handleAuth" in top_ids
        assert "generated::GeneratedRecord" not in top_ids
        assert "unknown::secretFlag" not in top_ids

        regular_groups = indexer.query("handleAuth", limit=5)
        assert regular_groups
        assert regular_groups[0].relative_path == "src/service.ts"

        generated_groups = indexer.query("GeneratedRecord", limit=5)
        unknown_groups = indexer.query("secretFlag", limit=5)
        assert all(group.relative_path != "proto/generated.ts" for group in generated_groups)
        assert all(group.relative_path != "src/unknown.ts" for group in unknown_groups)
    finally:
        indexer.close()
