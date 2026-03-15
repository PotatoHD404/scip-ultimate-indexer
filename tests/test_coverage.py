from __future__ import annotations

from pathlib import Path

from ultimate_indexer.formatter import format_groups
from ultimate_indexer.indexer import UltimateIndexer


def test_typescript_fallback_is_searchable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "auth.ts").write_text(
        "export function buildToken(userId: string) { return `token:${userId}` }\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project)
    try:
        summary = indexer.index(force=True)
        assert summary.indexed_files >= 1
        groups = indexer.query("buildToken token", limit=5)
        rendered = format_groups(indexer.storage, indexer.project_id, groups)
        assert "// src/auth.ts" in rendered
        assert "buildToken" in rendered
    finally:
        indexer.close()


def test_typescript_fallback_builds_import_graph(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "auth.ts").write_text(
        "export function buildToken(userId: string) { return `token:${userId}` }\n",
        encoding="utf-8",
    )
    (project / "src" / "main.ts").write_text(
        "import { buildToken } from './auth'\nexport const token = buildToken('42')\n",
        encoding="utf-8",
    )

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        imports_edges = [
            edge for edge in indexer.storage.get_edges(indexer.project_id)
            if edge["edge_type"] == "imports"
        ]
        assert any(
            str(edge["source_symbol_id"]) == "module::src/main.ts"
            and str(edge["target_symbol_id"]) == "module::src/auth.ts"
            for edge in imports_edges
        )
        groups = indexer.query("token builder", limit=5)
        output_path = indexer.visualize(groups, title="TS graph")
        html = output_path.read_text(encoding="utf-8")
        assert "main" in html
        assert "auth" in html
    finally:
        indexer.close()


def test_ignore_files_are_respected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "keep.py").write_text("def keep_me():\n    return 'visible_token'\n", encoding="utf-8")
    (project / "src" / "ignored.py").write_text("def drop_me():\n    return 'hidden_token'\n", encoding="utf-8")
    (project / ".gitignore").write_text("src/ignored.py\n", encoding="utf-8")
    (project / ".socraticodeignore").write_text("node_modules/\n", encoding="utf-8")
    (project / ".cgcignore").write_text("*.secret.py\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "dep.js").write_text("export const dep = 'ignored_dep'\n", encoding="utf-8")
    (project / "src" / "local.secret.py").write_text("def secret():\n    return 'secret_token'\n", encoding="utf-8")

    indexer = UltimateIndexer(project)
    try:
        indexer.index(force=True)
        assert indexer.storage.get_file(indexer.project_id, "src/ignored.py") is None
        assert indexer.storage.get_file(indexer.project_id, "src/local.secret.py") is None
        assert indexer.storage.get_file(indexer.project_id, "node_modules/dep.js") is None
        visible = format_groups(indexer.storage, indexer.project_id, indexer.query("visible_token", limit=5))
        hidden = format_groups(indexer.storage, indexer.project_id, indexer.query("hidden_token", limit=5))
        secret = format_groups(indexer.storage, indexer.project_id, indexer.query("secret_token", limit=5))
        dep = format_groups(indexer.storage, indexer.project_id, indexer.query("ignored_dep", limit=5))
        assert "// src/keep.py" in visible
        assert "// src/ignored.py" not in hidden
        assert "// src/local.secret.py" not in secret
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
    (project / "src" / "main.ts").write_text(
        "export const token = 'safe'\n",
        encoding="utf-8",
    )

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
