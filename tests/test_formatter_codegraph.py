from __future__ import annotations

from ultimate_indexer.formatter import _qualified_display_name


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
