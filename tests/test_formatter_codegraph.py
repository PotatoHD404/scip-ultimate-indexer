from __future__ import annotations

from pathlib import Path

from ultimate_indexer.formatter import (
    _qualified_display_name,
    _render_function_block,
    format_search_symbols_codegraph,
)
from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.python_scip import emit_python_scip


def _python_scip_path(project_root: Path) -> Path:
    python_files = sorted(project_root.rglob("*.py"))
    output_path = project_root / ".ultimate_indexer" / "cache" / "fixture.scip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return emit_python_scip(project_root, python_files, output_path)


def test_qualified_display_name_keeps_module_context() -> None:
    symbol_rows = {
        "file::models/request.go": {
            "display_name": "request.go",
            "enclosing_symbol_id": None,
            "kind": "File",
            "scip_symbol": "file::models/request.go",
        },
        "module::models": {
            "display_name": "models",
            "enclosing_symbol_id": "file::models/request.go",
            "kind": "Module",
            "scip_symbol": "module::models",
        },
        "type::Request": {
            "display_name": "Request",
            "enclosing_symbol_id": "module::models",
            "kind": "TypeAlias",
            "scip_symbol": "type::Request",
        },
        "method::Validate": {
            "display_name": "Validate",
            "enclosing_symbol_id": "type::Request",
            "kind": "Method",
            "scip_symbol": "method::Validate",
        },
    }
    assert _qualified_display_name("method::Validate", "Validate", symbol_rows) == "models.Request.Validate"


def test_qualified_display_name_falls_back_to_scip_symbol_when_parent_missing() -> None:
    symbol_id = "scip-go gomod example.com/app v1 `models`/Table#Validate()."
    symbol_rows = {
        symbol_id: {
            "display_name": "Validate",
            "enclosing_symbol_id": None,
            "kind": "Method",
            "scip_symbol": symbol_id,
        }
    }
    qualified = _qualified_display_name(symbol_id, "Validate", symbol_rows)
    assert qualified != "Validate"
    assert qualified.endswith("Table.Validate")


def test_render_function_block_uses_snippet_header_without_duplicate_signature() -> None:
    rows = _render_function_block(
        signature="func (*AccumulationProcess).Audience() *Audience",
        snippet="func (a *AccumulationProcess) Audience() *Audience {\n    return a.audience\n}",
        comment_prefix="//",
    )
    assert rows[0] == "func (a *AccumulationProcess) Audience() *Audience {"
    assert all("(*AccumulationProcess).Audience" not in row for row in rows[1:])


def test_format_search_symbols_codegraph_keeps_doc_and_code_files_visible(
    fixture_project: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    indexer = UltimateIndexer(fixture_project)
    try:
        indexer.index(scip_path=_python_scip_path(fixture_project))
        groups = indexer.query("users tenant email", limit=2)
        rendered = format_search_symbols_codegraph(
            indexer.storage,
            indexer.project_id,
            "users tenant email",
            groups,
            max_results=2,
        )
        assert "docs/schema.md" in rendered
        assert ".py" in rendered
    finally:
        indexer.close()
