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


def test_parse_scip_index_attaches_local_variable_to_enclosing_function(tmp_path: Path) -> None:
    index = scip_pb2.Index()
    document = index.documents.add()
    document.language = "go"
    document.relative_path = "main.go"
    document.text = (
        "package main\n\n"
        "func doThing() error {\n"
        "    var err error\n"
        "    return err\n"
        "}\n"
    )

    function_symbol = "scip-go gomod demo `main`/doThing()."
    local_symbol = "local 1"

    function_info = document.symbols.add()
    function_info.symbol = function_symbol
    function_info.display_name = "doThing"
    function_info.kind = scip_pb2.SymbolInformation.UnspecifiedKind
    function_info.documentation.append("```go\nfunc doThing() error\n```")

    local_info = document.symbols.add()
    local_info.symbol = local_symbol
    local_info.display_name = "err"
    local_info.kind = scip_pb2.SymbolInformation.UnspecifiedKind
    local_info.documentation.append("```go\nvar err error\n```")

    function_def = document.occurrences.add()
    function_def.symbol = function_symbol
    function_def.range.extend([2, 5, 2, 12])
    function_def.enclosing_range.extend([2, 0, 5, 1])
    function_def.symbol_roles = scip_pb2.Definition
    function_def.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    local_def = document.occurrences.add()
    local_def.symbol = local_symbol
    local_def.range.extend([3, 8, 3, 11])
    local_def.enclosing_range.extend([2, 0, 5, 1])
    local_def.symbol_roles = scip_pb2.Definition
    local_def.syntax_kind = scip_pb2.IdentifierLocal

    local_ref = document.occurrences.add()
    local_ref.symbol = local_symbol
    local_ref.range.extend([4, 11, 4, 14])
    local_ref.enclosing_range.extend([2, 0, 5, 1])
    local_ref.symbol_roles = scip_pb2.ReadAccess
    local_ref.syntax_kind = scip_pb2.IdentifierLocal

    index_path = tmp_path / "fixture.scip"
    index_path.write_bytes(index.SerializeToString())

    parsed = parse_scip_index(
        project_id=str(tmp_path),
        project_root=tmp_path,
        index_path=index_path,
    )

    local_record = next(symbol for symbol in parsed.symbols if symbol.scip_symbol == local_symbol)
    function_record = next(symbol for symbol in parsed.symbols if symbol.scip_symbol == function_symbol)
    assert local_record.enclosing_symbol_id == function_record.symbol_id


def test_parse_scip_index_uses_descriptor_parent_for_method_when_enclosing_missing(tmp_path: Path) -> None:
    index = scip_pb2.Index()
    document = index.documents.add()
    document.language = "go"
    document.relative_path = "models/request.go"
    document.text = "package models\n\ntype Request struct{}\n"

    type_symbol = "scip-go gomod demo `models`/Request#"
    method_symbol = "scip-go gomod demo `models`/Request#Validate()."

    type_info = document.symbols.add()
    type_info.symbol = type_symbol
    type_info.display_name = "Request"
    type_info.kind = scip_pb2.SymbolInformation.Type

    method_info = document.symbols.add()
    method_info.symbol = method_symbol
    method_info.display_name = "Validate"
    method_info.kind = scip_pb2.SymbolInformation.Method
    method_info.enclosing_symbol = ""

    index_path = tmp_path / "fixture.scip"
    index_path.write_bytes(index.SerializeToString())

    parsed = parse_scip_index(
        project_id=str(tmp_path),
        project_root=tmp_path,
        index_path=index_path,
    )

    type_record = next(symbol for symbol in parsed.symbols if symbol.scip_symbol == type_symbol)
    method_record = next(symbol for symbol in parsed.symbols if symbol.scip_symbol == method_symbol)
    assert method_record.enclosing_symbol_id == type_record.symbol_id
    assert any(
        edge.source_symbol_id == type_record.symbol_id
        and edge.target_symbol_id == method_record.symbol_id
        and edge.edge_type == "contains"
        for edge in parsed.edges
    )


def test_parse_scip_index_reference_source_prefers_function_over_local_on_tied_ranges(tmp_path: Path) -> None:
    index = scip_pb2.Index()
    document = index.documents.add()
    document.language = "go"
    document.relative_path = "main.go"
    document.text = (
        "package main\n\n"
        "func doThing() {\n"
        "    var err error\n"
        "    helper()\n"
        "}\n\n"
        "func helper() {}\n"
    )

    function_symbol = "scip-go gomod demo `main`/doThing()."
    local_symbol = "local 1"
    helper_symbol = "scip-go gomod demo `main`/helper()."

    function_info = document.symbols.add()
    function_info.symbol = function_symbol
    function_info.display_name = "doThing"
    function_info.kind = scip_pb2.SymbolInformation.Function

    local_info = document.symbols.add()
    local_info.symbol = local_symbol
    local_info.display_name = "err"
    local_info.kind = scip_pb2.SymbolInformation.Variable

    helper_info = document.symbols.add()
    helper_info.symbol = helper_symbol
    helper_info.display_name = "helper"
    helper_info.kind = scip_pb2.SymbolInformation.Function

    local_def = document.occurrences.add()
    local_def.symbol = local_symbol
    local_def.range.extend([3, 8, 3, 11])
    local_def.enclosing_range.extend([2, 0, 5, 1])
    local_def.symbol_roles = scip_pb2.Definition
    local_def.syntax_kind = scip_pb2.IdentifierLocal

    function_def = document.occurrences.add()
    function_def.symbol = function_symbol
    function_def.range.extend([2, 5, 2, 12])
    function_def.enclosing_range.extend([2, 0, 5, 1])
    function_def.symbol_roles = scip_pb2.Definition
    function_def.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    helper_def = document.occurrences.add()
    helper_def.symbol = helper_symbol
    helper_def.range.extend([7, 5, 7, 11])
    helper_def.enclosing_range.extend([7, 0, 7, 16])
    helper_def.symbol_roles = scip_pb2.Definition
    helper_def.syntax_kind = scip_pb2.IdentifierFunctionDefinition

    helper_ref = document.occurrences.add()
    helper_ref.symbol = helper_symbol
    helper_ref.range.extend([4, 4, 4, 10])
    helper_ref.enclosing_range.extend([2, 0, 5, 1])
    helper_ref.symbol_roles = scip_pb2.ReadAccess
    helper_ref.syntax_kind = scip_pb2.IdentifierFunction

    index_path = tmp_path / "fixture.scip"
    index_path.write_bytes(index.SerializeToString())

    parsed = parse_scip_index(
        project_id=str(tmp_path),
        project_root=tmp_path,
        index_path=index_path,
    )

    normalized_local = "local::main.go::local 1"
    assert any(
        edge.source_symbol_id == function_symbol
        and edge.target_symbol_id == helper_symbol
        and edge.edge_type == "calls"
        for edge in parsed.edges
    )
    assert not any(
        edge.source_symbol_id == normalized_local
        and edge.target_symbol_id == helper_symbol
        and edge.edge_type == "calls"
        for edge in parsed.edges
    )
