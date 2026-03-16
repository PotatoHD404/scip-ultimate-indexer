from __future__ import annotations

from pathlib import Path

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
        assert "class GreetingService:" in rendered
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
        assert "vis-network" in html
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
