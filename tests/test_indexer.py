from __future__ import annotations

from pathlib import Path

from ultimate_indexer.formatter import format_groups, format_top_symbols
from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.models import IndexProgress
from ultimate_indexer.python_scip import emit_python_scip
from ultimate_indexer.scip_runner import ScipRunFailure, ScipRunReport


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
