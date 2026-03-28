from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ultimate_indexer.cli import app
from ultimate_indexer.python_scip import emit_python_scip


runner = CliRunner()


def test_cli_end_to_end(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    cache_dir = fixture_project.parent / ".cli-cache"
    python_files = sorted(fixture_project.rglob("*.py"))
    scip_path = fixture_project / ".ultimate_indexer" / "cache" / "fixture.scip"
    scip_path.parent.mkdir(parents=True, exist_ok=True)
    emit_python_scip(fixture_project, python_files, scip_path)

    index_result = runner.invoke(
        app,
        [
            "index",
            str(fixture_project),
            "--embedding-backend",
            "hash",
            "--cache-dir",
            str(cache_dir),
            "--scip-path",
            str(scip_path),
        ],
    )
    assert index_result.exit_code == 0, index_result.stdout
    assert "Index Summary" in index_result.stdout

    query_result = runner.invoke(
        app,
        [
            "query",
            str(fixture_project),
            "greeting service",
            "--embedding-backend",
            "hash",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert query_result.exit_code == 0, query_result.stdout
    assert "// pkg/services.py" in query_result.stdout

    tree_result = runner.invoke(
        app,
        [
            "tree",
            str(fixture_project),
            "--embedding-backend",
            "hash",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert tree_result.exit_code == 0, tree_result.stdout
    assert "folder accumulation" in tree_result.stdout
    assert "value=" in tree_result.stdout

    top_result = runner.invoke(
        app,
        [
            "top-symbols",
            str(fixture_project),
            "--embedding-backend",
            "hash",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert top_result.exit_code == 0, top_result.stdout
    assert "GreetingService" in top_result.stdout or "build_greeting" in top_result.stdout
    assert any(path.name == "index.sqlite3" for path in cache_dir.rglob("index.sqlite3"))


def test_mcp_command_exposes_graph_indexer_compatible_flags() -> None:
    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "--cache-dir" in result.stdout
    assert "--embedding-model" in result.stdout
    assert "--embedding-n-ctx" in result.stdout
    assert "--transport" in result.stdout
