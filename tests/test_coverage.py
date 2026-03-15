from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from ultimate_indexer.cli import app
from ultimate_indexer.formatter import format_groups
from ultimate_indexer.indexer import UltimateIndexer


runner = CliRunner()


def test_unsupported_language_fallback_is_searchable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "auth.rb").write_text(
        "def build_token(user_id)\n  \"token:#{user_id}\"\nend\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project)
    try:
        summary = indexer.index(force=True)
        assert summary.indexed_files >= 1
        groups = indexer.query("build_token token", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// src/auth.rb" in rendered
        assert "build_token" in rendered
    finally:
        indexer.close()


def test_supported_language_without_scip_tool_exits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "auth.ts").write_text(
        "export function buildToken(userId: string) { return `token:${userId}` }\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["index", str(project), "--embedding-backend", "hash", "--no-progress"],
    )

    assert result.exit_code == 1, result.stdout
    assert "fallback indexing was" in result.stdout
    assert "skipped" in result.stdout
    assert "scip-typescript" in result.stdout
    assert "npm install -g @sourcegraph/scip-typescript" in result.stdout


def test_visualization_uses_path_qualified_labels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "frontend").mkdir()
    (project / "backend").mkdir()
    (project / "frontend" / ".gitignore").write_text("dist/\n", encoding="utf-8")
    (project / "backend" / ".gitignore").write_text("build/\n", encoding="utf-8")

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        groups = indexer.query("dist build", limit=5)
        output_path = indexer.visualize(groups, title="Ignore graph")
        html = output_path.read_text(encoding="utf-8")
        assert "frontend/.gitignore" in html
        assert "backend/.gitignore" in html
    finally:
        indexer.close()


def test_ignore_files_are_respected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "keep.rb").write_text("def keep_me\n  'visible_token'\nend\n", encoding="utf-8")
    (project / "src" / "ignored.rb").write_text("def drop_me\n  'hidden_token'\nend\n", encoding="utf-8")
    (project / ".gitignore").write_text("src/ignored.rb\n", encoding="utf-8")
    (project / ".socraticodeignore").write_text("node_modules/\n", encoding="utf-8")
    (project / ".cgcignore").write_text("*.secret.rb\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "dep.js").write_text("export const dep = 'ignored_dep'\n", encoding="utf-8")
    (project / "src" / "local.secret.rb").write_text("def secret\n  'secret_token'\nend\n", encoding="utf-8")

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        assert indexer.storage.get_file(indexer.project_id, "src/ignored.rb") is None
        assert indexer.storage.get_file(indexer.project_id, "src/local.secret.rb") is None
        assert indexer.storage.get_file(indexer.project_id, "node_modules/dep.js") is None
        visible = format_groups(indexer.storage, indexer.project_id, indexer.query("visible_token", limit=5))
        hidden = format_groups(indexer.storage, indexer.project_id, indexer.query("hidden_token", limit=5))
        secret = format_groups(indexer.storage, indexer.project_id, indexer.query("secret_token", limit=5))
        dep = format_groups(indexer.storage, indexer.project_id, indexer.query("ignored_dep", limit=5))
        assert "// src/keep.rb" in visible
        assert "// src/ignored.rb" not in hidden
        assert "// src/local.secret.rb" not in secret
        assert "node_modules" not in dep
    finally:
        indexer.close()


def test_extra_extensions_are_indexed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("EXTRA_EXTENSIONS", ".tpl")
    project = tmp_path / "project"
    project.mkdir()
    (project / "views").mkdir()
    (project / "views" / "layout.tpl").write_text(
        "<div>{{ greeting_message }}</div>\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        rendered = format_groups(indexer.storage, indexer.project_id, indexer.query("greeting_message", limit=5))
        assert "// views/layout.tpl" in rendered
    finally:
        indexer.close()


def test_state_directory_is_not_reindexed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.rb").write_text("TOKEN = 'safe'\n", encoding="utf-8")

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        visuals_dir = project / ".ultimate_indexer" / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (visuals_dir / "query_graph.html").write_text("<html>generated</html>\n", encoding="utf-8")
        indexer.index(force=True)
        assert indexer.storage.get_file(indexer.project_id, ".ultimate_indexer/visuals/query_graph.html") is None
    finally:
        indexer.close()


def test_extra_ignored_directories_are_skipped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "keep.rb").write_text("VISIBLE_TOKEN = 'kept'\n", encoding="utf-8")
    (project / ".build").mkdir()
    (project / ".build" / "generated.rb").write_text("VISIBLE_TOKEN = 'generated'\n", encoding="utf-8")
    (project / ".pycache").mkdir()
    (project / ".pycache" / "cached.rb").write_text("VISIBLE_TOKEN = 'cached'\n", encoding="utf-8")

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        assert indexer.storage.get_file(indexer.project_id, ".build/generated.rb") is None
        assert indexer.storage.get_file(indexer.project_id, ".pycache/cached.rb") is None
        rendered = format_groups(indexer.storage, indexer.project_id, indexer.query("kept", limit=5))
        assert "// src/keep.rb" in rendered
    finally:
        indexer.close()
