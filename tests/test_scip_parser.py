from __future__ import annotations

from pathlib import Path

from ultimate_indexer import scip_pb2
from ultimate_indexer.scip_parser import _infer_kind, _kind_name, parse_scip_index


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


def test_kind_name_maps_unspecified_to_unknown() -> None:
    assert _kind_name(0) == "Unknown"


def test_parse_scip_index_cleans_display_name_from_docstring(tmp_path: Path) -> None:
    index = scip_pb2.Index()
    document = index.documents.add()
    document.language = "typescript"
    document.relative_path = "proto/object.ts"
    document.text = "export interface ObjectDetails {}\n"

    info = document.symbols.add()
    info.symbol = "scip-typescript npm demo 1.0 proto/`object.ts`/ObjectDetails#"
    info.display_name = ""
    info.kind = scip_pb2.SymbolInformation.UnspecifiedKind
    info.documentation.append("```ts\ninterface ObjectDetails\n```")

    definition = document.occurrences.add()
    definition.symbol = info.symbol
    definition.range.extend([0, 17, 0, 30])
    definition.enclosing_range.extend([0, 0, 0, 31])
    definition.symbol_roles = scip_pb2.Definition
    definition.syntax_kind = scip_pb2.IdentifierType

    index_path = tmp_path / "fixture.scip"
    index_path.write_bytes(index.SerializeToString())

    parsed = parse_scip_index(
        project_id=str(tmp_path),
        project_root=tmp_path,
        index_path=index_path,
    )

    symbol = next(item for item in parsed.symbols if item.scip_symbol == info.symbol)
    assert symbol.display_name == "ObjectDetails"
    assert symbol.kind == "Interface"


def test_infer_kind_uses_docstring_and_symbol_shape() -> None:
    assert _infer_kind(
        "scip-go gomod demo `pkg`/main().",
        "```go\nfunc main()\n```",
        "",
        "func main() {}",
    ) == "Function"
    assert _infer_kind(
        "scip-go gomod demo pkg/Response#Valid.",
        "```go\nstruct field Valid bool\n```",
        "",
        "Valid bool",
    ) == "Field"
    assert _infer_kind(
        "scip-typescript npm demo lib/`router.ts`/ROUTING.",
        "```ts\nconst ROUTING: Record<string, string>\n```",
        "",
        "export const ROUTING = {}",
    ) == "Constant"
