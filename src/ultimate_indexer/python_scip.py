from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from . import scip_pb2


@dataclass(slots=True)
class DefinitionDescriptor:
    relative_path: str
    module_name: str
    name: str
    qualname: str
    symbol_id: str
    kind: str
    lineno: int
    end_lineno: int
    col_offset: int
    end_col_offset: int
    signature: str
    docstring: str
    parent_symbol: str | None


def _module_name(relative_path: str) -> str:
    path = relative_path[:-3] if relative_path.endswith(".py") else relative_path
    parts = [part for part in path.split("/") if part != "__init__"]
    return ".".join(parts) or "module"


def _build_symbol(module_name: str, qualname: str, kind: str) -> str:
    return f"py:{module_name}:{qualname}:{kind.lower()}"


def _render_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    pieces: list[str] = []
    if node.decorator_list:
        pieces.extend(f"@{ast.unparse(item)}" for item in node.decorator_list)
    args = ast.unparse(node.args)
    returns = ""
    if node.returns is not None:
        returns = f" -> {ast.unparse(node.returns)}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    pieces.append(f"{prefix} {node.name}({args}){returns}:")
    return "\n".join(pieces)


def _render_class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    suffix = f"({bases})" if bases else ""
    return f"class {node.name}{suffix}:"


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.module_name = _module_name(relative_path)
        self.stack: list[DefinitionDescriptor] = []
        self.definitions: list[DefinitionDescriptor] = []

    def _push(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
        signature: str,
    ) -> None:
        qualname = ".".join([item.name for item in self.stack] + [node.name])
        descriptor = DefinitionDescriptor(
            relative_path=self.relative_path,
            module_name=self.module_name,
            name=node.name,
            qualname=qualname,
            symbol_id=_build_symbol(self.module_name, qualname, kind),
            kind=kind,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            col_offset=node.col_offset,
            end_col_offset=node.end_col_offset or node.col_offset + len(node.name),
            signature=signature,
            docstring=ast.get_docstring(node, clean=False) or "",
            parent_symbol=self.stack[-1].symbol_id if self.stack else None,
        )
        self.definitions.append(descriptor)
        self.stack.append(descriptor)
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push(node, "Class", _render_class_signature(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "Method" if self.stack and self.stack[-1].kind == "Class" else "Function"
        self._push(node, kind, _render_function_signature(node))

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        kind = "Method" if self.stack and self.stack[-1].kind == "Class" else "Function"
        self._push(node, kind, _render_function_signature(node))


def _name_range(node: ast.AST, name: str) -> list[int]:
    start_line = getattr(node, "lineno", 1) - 1
    start_col = getattr(node, "col_offset", 0)
    return [start_line, start_col, start_line, start_col + len(name)]


def _enclosing_range(descriptor: DefinitionDescriptor) -> list[int]:
    return [
        descriptor.lineno - 1,
        descriptor.col_offset,
        descriptor.end_lineno - 1,
        descriptor.end_col_offset,
    ]


class DocumentEmitter(ast.NodeVisitor):
    def __init__(
        self,
        document: scip_pb2.Document,
        relative_path: str,
        source: str,
        definitions: list[DefinitionDescriptor],
        global_symbols: dict[str, DefinitionDescriptor],
        module_index: dict[str, dict[str, str]],
    ) -> None:
        self.document = document
        self.relative_path = relative_path
        self.source = source
        self.global_symbols = global_symbols
        self.module_index = module_index
        self.module_name = _module_name(relative_path)
        self.definition_by_lineno = {(item.lineno, item.name): item for item in definitions}
        self.child_definitions: dict[str | None, dict[str, DefinitionDescriptor]] = defaultdict(dict)
        for item in definitions:
            self.child_definitions[item.parent_symbol][item.name] = item
        self.scope_stack: list[DefinitionDescriptor] = []
        self.import_stack: list[dict[str, str]] = [dict()]
        self._add_file_symbol()

    def _add_file_symbol(self) -> None:
        info = self.document.symbols.add()
        info.symbol = f"file::{self.relative_path}"
        info.display_name = Path(self.relative_path).name
        info.kind = scip_pb2.SymbolInformation.File
        info.documentation.append(self.relative_path)
        occ = self.document.occurrences.add()
        occ.symbol = info.symbol
        occ.symbol_roles = scip_pb2.Definition
        occ.syntax_kind = scip_pb2.IdentifierNamespace
        occ.range.extend([0, 0, 0, max(1, len(Path(self.relative_path).name))])
        occ.enclosing_range.extend([0, 0, max(0, len(self.source.splitlines()) - 1), 0])

    def _resolve_name(self, name: str) -> str | None:
        for scope in reversed(self.scope_stack):
            children = self.child_definitions.get(scope.symbol_id, {})
            if name in children:
                return children[name].symbol_id
        top_level = self.child_definitions.get(None, {})
        if name in top_level:
            return top_level[name].symbol_id
        for imports in reversed(self.import_stack):
            if name in imports:
                return imports[name]
        module_symbols = self.module_index.get(self.module_name, {})
        if name in module_symbols:
            return module_symbols[name]
        return None

    def _emit_reference(self, node: ast.AST, symbol: str, syntax_kind: int) -> None:
        name = getattr(node, "id", getattr(node, "attr", symbol.split(":")[-1]))
        occ = self.document.occurrences.add()
        occ.symbol = symbol
        occ.syntax_kind = syntax_kind
        occ.range.extend(_name_range(node, str(name)))
        if self.scope_stack:
            occ.enclosing_range.extend(_enclosing_range(self.scope_stack[-1]))

    def _emit_annotation(self, node: ast.AST | None) -> None:
        if node is None:
            return
        if isinstance(node, ast.Name):
            symbol = self._resolve_name(node.id)
            if symbol:
                self._emit_reference(node, symbol, scip_pb2.IdentifierType)
        elif isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name):
                base_symbol = self.import_stack[-1].get(base.id)
                if base_symbol:
                    target = self.module_index.get(base_symbol, {}).get(node.attr)
                    if target:
                        self._emit_reference(node, target, scip_pb2.IdentifierType)
        else:
            for child in ast.iter_child_nodes(node):
                self._emit_annotation(child)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound_name = alias.asname or alias.name.split(".")[-1]
            self.import_stack[-1][bound_name] = alias.name
            occ = self.document.occurrences.add()
            occ.symbol = alias.name
            occ.syntax_kind = scip_pb2.IdentifierModule
            occ.range.extend(_name_range(node, bound_name))
            if self.scope_stack:
                occ.enclosing_range.extend(_enclosing_range(self.scope_stack[-1]))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        for alias in node.names:
            bound_name = alias.asname or alias.name
            resolved = self.module_index.get(module_name, {}).get(alias.name)
            if resolved:
                self.import_stack[-1][bound_name] = resolved
                self._emit_reference(node, resolved, scip_pb2.IdentifierModule)
            else:
                self.import_stack[-1][bound_name] = f"{module_name}.{alias.name}".strip(".")

    def _open_definition(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> DefinitionDescriptor:
        descriptor = self.definition_by_lineno[(node.lineno, node.name)]
        info = self.document.symbols.add()
        info.symbol = descriptor.symbol_id
        info.display_name = descriptor.name
        info.documentation.append(descriptor.docstring)
        info.kind = getattr(scip_pb2.SymbolInformation, descriptor.kind, scip_pb2.SymbolInformation.UnspecifiedKind)
        info.enclosing_symbol = descriptor.parent_symbol or f"file::{self.relative_path}"
        info.signature_documentation.language = "python"
        info.signature_documentation.relative_path = descriptor.relative_path
        info.signature_documentation.text = descriptor.signature
        occ = self.document.occurrences.add()
        occ.symbol = descriptor.symbol_id
        occ.symbol_roles = scip_pb2.Definition
        occ.syntax_kind = (
            scip_pb2.IdentifierType
            if descriptor.kind == "Class"
            else scip_pb2.IdentifierFunctionDefinition
        )
        occ.range.extend(_name_range(node, descriptor.name))
        occ.enclosing_range.extend(_enclosing_range(descriptor))
        self.scope_stack.append(descriptor)
        self.import_stack.append(dict(self.import_stack[-1]))
        return descriptor

    def _close_definition(self) -> None:
        self.import_stack.pop()
        self.scope_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._open_definition(node)
        for base in node.bases:
            self._emit_annotation(base)
        for decorator in node.decorator_list:
            self._emit_annotation(decorator)
        for child in node.body:
            self.visit(child)
        self._close_definition()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._open_definition(node)
        for arg in (
            list(node.args.posonlyargs)
            + list(node.args.args)
            + list(node.args.kwonlyargs)
        ):
            self._emit_annotation(arg.annotation)
        if node.args.vararg:
            self._emit_annotation(node.args.vararg.annotation)
        if node.args.kwarg:
            self._emit_annotation(node.args.kwarg.annotation)
        self._emit_annotation(node.returns)
        for decorator in node.decorator_list:
            self._emit_annotation(decorator)
        for child in node.body:
            self.visit(child)
        self._close_definition()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            symbol = self._resolve_name(node.func.id)
            if symbol:
                self._emit_reference(node.func, symbol, scip_pb2.IdentifierFunction)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            symbol = self._resolve_name(node.id)
            if symbol:
                self._emit_reference(node, symbol, scip_pb2.Identifier)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            module_binding = self.import_stack[-1].get(node.value.id)
            if module_binding and module_binding in self.module_index:
                target = self.module_index[module_binding].get(node.attr)
                if target:
                    self._emit_reference(node, target, scip_pb2.IdentifierFunction)
        self.generic_visit(node)


def emit_python_scip(project_root: Path, files: list[Path], output_path: Path) -> Path:
    project_root = project_root.resolve()
    definitions_by_file: dict[str, list[DefinitionDescriptor]] = {}
    module_index: dict[str, dict[str, str]] = defaultdict(dict)
    global_symbols: dict[str, DefinitionDescriptor] = {}

    # Parse each file once, skipping anything we cannot read or parse (e.g.
    # notebooks, files with syntax errors, paths outside the project root).
    # Those fall through to generic non-SCIP coverage instead of aborting the
    # whole built-in emission.
    parsed_sources: list[tuple[str, str, ast.AST]] = []
    for path in files:
        try:
            relative_path = str(path.relative_to(project_root))
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_path)
        except (OSError, ValueError, SyntaxError, UnicodeDecodeError):
            continue
        parsed_sources.append((relative_path, source, tree))
        collector = DefinitionCollector(relative_path)
        collector.visit(tree)
        definitions_by_file[relative_path] = collector.definitions
        for descriptor in collector.definitions:
            global_symbols[descriptor.symbol_id] = descriptor
            module_index[descriptor.module_name][descriptor.name] = descriptor.symbol_id

    index = scip_pb2.Index()
    index.metadata.version = scip_pb2.UnspecifiedProtocolVersion
    index.metadata.project_root = str(project_root)
    index.metadata.text_document_encoding = scip_pb2.UTF8
    index.metadata.tool_info.name = "ultimate-indexer"
    index.metadata.tool_info.version = "0.1.0"

    for relative_path, source, tree in parsed_sources:
        document = index.documents.add()
        document.language = "python"
        document.relative_path = relative_path
        document.text = source
        document.position_encoding = scip_pb2.UTF8CodeUnitOffsetFromLineStart
        emitter = DocumentEmitter(
            document=document,
            relative_path=relative_path,
            source=source,
            definitions=definitions_by_file[relative_path],
            global_symbols=global_symbols,
            module_index=module_index,
        )
        emitter.visit(tree)

    output_path.write_bytes(index.SerializeToString())
    return output_path
