from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ultimate_indexer.cli import app


runner = CliRunner()


def test_cli_end_to_end(fixture_project: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")

    index_result = runner.invoke(
        app,
        ["index", str(fixture_project), "--embedding-backend", "hash"],
    )
    assert index_result.exit_code == 0, index_result.stdout
    assert "Index Summary" in index_result.stdout

    query_result = runner.invoke(
        app,
        ["query", str(fixture_project), "greeting service", "--embedding-backend", "hash"],
    )
    assert query_result.exit_code == 0, query_result.stdout
    assert "// pkg/services.py" in query_result.stdout

    top_result = runner.invoke(
        app,
        ["top-symbols", str(fixture_project), "--embedding-backend", "hash"],
    )
    assert top_result.exit_code == 0, top_result.stdout
    assert "GreetingService" in top_result.stdout or "build_greeting" in top_result.stdout
