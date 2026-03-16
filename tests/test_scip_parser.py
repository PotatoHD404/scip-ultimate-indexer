from __future__ import annotations

from pathlib import Path

from ultimate_indexer import scip_pb2
from ultimate_indexer.scip_parser import parse_scip_index


def _write_index(path: Path) -> Path:
    index = scip_pb2.Index()

    for relative_path, source in {
        "a.ts": "const a = 1;\nconsole.log(a)\n",
        "b.ts": "const b = 2;\nconsole.log(b)\n",
    }.items():
        document = index.documents.add()
        document.language = "typescript"
        document.relative_path = relative_path
        document.text = source

        info = document.symbols.add()
        info.symbol = "local 1"
        info.display_name = "value"
        info.kind = scip_pb2.SymbolInformation.Variable
        info.enclosing_symbol = ""

        definition = document.occurrences.add()
        definition.symbol = "local 1"
        definition.range.extend([0, 6, 0, 7])
        definition.enclosing_range.extend([0, 0, 1, 14])
        definition.symbol_roles = scip_pb2.Definition
        definition.syntax_kind = scip_pb2.IdentifierLocal

        reference = document.occurrences.add()
        reference.symbol = "local 1"
        reference.range.extend([1, 12, 1, 13])
        reference.enclosing_range.extend([0, 0, 1, 14])
        reference.symbol_roles = scip_pb2.ReadAccess
        reference.syntax_kind = scip_pb2.IdentifierLocal

    path.write_bytes(index.SerializeToString())
    return path


def test_parse_scip_index_namespaces_local_symbols(tmp_path: Path) -> None:
    index_path = _write_index(tmp_path / "fixture.scip")

    parsed = parse_scip_index(
        project_id=str(tmp_path),
        project_root=tmp_path,
        index_path=index_path,
    )

    local_symbols = {symbol.symbol_id: symbol for symbol in parsed.symbols if symbol.scip_symbol == "local 1"}
    assert "local::a.ts::local 1" in local_symbols
    assert "local::b.ts::local 1" in local_symbols

    local_edges = {
        (edge.source_symbol_id, edge.target_symbol_id, edge.edge_type)
        for edge in parsed.edges
        if edge.target_symbol_id.startswith("local::")
    }
    assert ("local::a.ts::local 1", "local::a.ts::local 1", "uses") in local_edges
    assert ("local::b.ts::local 1", "local::b.ts::local 1", "uses") in local_edges
