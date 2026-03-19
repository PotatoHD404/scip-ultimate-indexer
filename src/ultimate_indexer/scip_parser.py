from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import DEFAULT_EDGE_WEIGHTS
from .embeddings import hash_text
from .models import EdgeRecord, FileRecord, SymbolRecord
from .ranking_rules import clean_symbol_display_name
from . import scip_pb2


TYPE_KINDS = {
    scip_pb2.IdentifierType,
    scip_pb2.IdentifierBuiltinType,
}
CALL_KINDS = {
    scip_pb2.IdentifierFunction,
    scip_pb2.IdentifierFunctionDefinition,
}
IMPORT_KINDS = {
    scip_pb2.IdentifierModule,
    scip_pb2.IdentifierNamespace,
}


@dataclass(slots=True)
class ParsedScip:
    files: list[FileRecord]
    symbols: list[SymbolRecord]
    edges: list[EdgeRecord]


def _normalize_symbol_id(relative_path: str, symbol: str) -> str:
    if symbol.startswith("local "):
        return f"local::{relative_path}::{symbol}"
    return symbol


def _kind_name(value: int) -> str:
    try:
        name = scip_pb2.SymbolInformation.Kind.Name(value)
    except ValueError:
        return "Unknown"
    if name == "UnspecifiedKind":
        return "Unknown"
    return name


def _first_symbol_doc_line(docstring: str, signature: str) -> str:
    for source in (docstring, signature):
        for raw_line in source.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            return line
    return ""


def _infer_kind(symbol: str, docstring: str, signature: str, snippet: str) -> str:
    line = _first_symbol_doc_line(docstring, signature)
    lowered = line.lower()
    if lowered.startswith("func ") or lowered.startswith("function "):
        return "Function"
    if lowered.startswith("method "):
        return "Method"
    if lowered.startswith("interface "):
        return "Interface"
    if lowered.startswith("class "):
        return "Class"
    if lowered.startswith("enum "):
        return "Enum"
    if lowered.startswith("type "):
        return "TypeAlias"
    if lowered.startswith("struct field ") or lowered.startswith("(property) "):
        return "Field"
    if lowered.startswith("(parameter) "):
        return "Parameter"
    if lowered.startswith("const "):
        return "Constant"
    if lowered.startswith("var ") or lowered.startswith("let "):
        return "Variable"
    if lowered.startswith("module "):
        return "Module"

    snippet_line = next((item.strip() for item in snippet.splitlines() if item.strip()), "")
    if re.match(r"^(?:export\s+)?(?:async\s+)?function\b", snippet_line):
        return "Function"
    if re.match(r"^(?:export\s+)?(?:const|let|var)\b", snippet_line):
        return "Variable"
    if re.match(r"^(?:export\s+)?interface\b", snippet_line):
        return "Interface"
    if re.match(r"^(?:export\s+)?type\b", snippet_line):
        return "TypeAlias"

    tail = symbol.rsplit("/", 1)[-1]
    if tail.endswith("()."):
        return "Function"
    if "#" in tail and tail.endswith("."):
        return "Field"
    if tail.endswith("#"):
        return "TypeAlias"
    if tail.endswith("/"):
        return "Module"
    return "Unknown"


def _range_to_lines(rng: list[int]) -> tuple[int, int]:
    if not rng:
        return 1, 1
    if len(rng) == 3:
        return rng[0] + 1, rng[0] + 1
    return rng[0] + 1, rng[2] + 1


def _slice_snippet(text: str, start_line: int, end_line: int) -> str:
    lines = text.splitlines()
    start_index = max(0, start_line - 1)
    end_index = min(len(lines), end_line)
    return "\n".join(lines[start_index:end_index])


def _classify_edge(syntax_kind: int) -> str:
    if syntax_kind in TYPE_KINDS:
        return "type"
    if syntax_kind in CALL_KINDS:
        return "calls"
    if syntax_kind in IMPORT_KINDS:
        return "imports"
    return "uses"


def _definition_priority(symbol_id: str, kind: str) -> int:
    if symbol_id.startswith("local::"):
        return 0
    normalized = kind.replace("_", "").replace("-", "").lower()
    if normalized in {"function", "method", "class", "struct", "interface", "trait", "enum", "typealias", "type"}:
        return 5
    if normalized in {"module", "namespace", "package", "constant", "const"}:
        return 3
    if normalized in {"field", "property", "parameter", "variable"}:
        return 1
    if normalized in {"file", "section", "unknown"}:
        return 0
    return 2


def _parent_symbol_from_scip_symbol(symbol: str) -> str | None:
    if symbol.startswith("local "):
        return None
    parts = symbol.split()
    if len(parts) < 4:
        return None
    prefix = " ".join(parts[:-1])
    descriptor = parts[-1].strip()
    if not descriptor or descriptor.endswith("/"):
        return None

    if "#" in descriptor:
        hash_index = descriptor.rfind("#")
        member_part = descriptor[hash_index + 1 :]
        if member_part in {"", "."}:
            slash_index = descriptor.rfind("/")
            if slash_index <= 0:
                return None
            return f"{prefix} {descriptor[: slash_index + 1]}"
        return f"{prefix} {descriptor[: hash_index + 1]}"

    slash_index = descriptor.rfind("/")
    if slash_index <= 0:
        return None
    return f"{prefix} {descriptor[: slash_index + 1]}"


def _best_enclosing_definition(
    definitions: list[tuple[tuple[int, int, int, int], str, int]],
    position: tuple[int, ...],
    *,
    exclude_symbol_id: str | None = None,
) -> str | None:
    best_choice: tuple[tuple[int, int], int, str] | None = None
    for full_range, symbol_id, priority in definitions:
        if exclude_symbol_id is not None and symbol_id == exclude_symbol_id:
            continue
        start_line, start_col, end_line, end_col = (full_range + (0, 0, 0, 0))[:4]
        pos_line = position[0]
        pos_col = position[1] if len(position) > 1 else 0
        if (start_line < pos_line or (start_line == pos_line and start_col <= pos_col)) and (
            end_line > pos_line or (end_line == pos_line and end_col >= pos_col)
        ):
            span = (end_line - start_line, end_col - start_col)
            if best_choice is None or span < best_choice[0] or (span == best_choice[0] and priority > best_choice[1]):
                best_choice = (span, priority, symbol_id)
    return None if best_choice is None else best_choice[2]


def parse_scip_index(
    project_id: str,
    project_root: Path,
    index_path: Path,
    edge_weights: dict[str, float] | None = None,
    source_root: Path | None = None,
) -> ParsedScip:
    edge_weights = edge_weights or DEFAULT_EDGE_WEIGHTS
    source_root = source_root or project_root
    index = scip_pb2.Index()
    index.ParseFromString(index_path.read_bytes())

    files: list[FileRecord] = []
    symbols: list[SymbolRecord] = []
    edges: list[EdgeRecord] = []
    symbols_by_key: dict[str, SymbolRecord] = {}
    def_ranges: dict[str, list[tuple[tuple[int, int, int, int], str, int]]] = {}

    for document in index.documents:
        document_relative_path = document.relative_path
        abs_path = str((source_root / document_relative_path).resolve())
        try:
            relative_path = Path(abs_path).resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            relative_path = document_relative_path
        content = document.text or Path(abs_path).read_text(encoding="utf-8")
        files.append(
            FileRecord(
                project_id=project_id,
                relative_path=relative_path,
                abs_path=abs_path,
                language=document.language or Path(relative_path).suffix.lstrip("."),
                content_hash=hash_text(content),
                content=content,
            )
        )
        file_symbol = SymbolRecord(
            project_id=project_id,
            symbol_id=f"file::{relative_path}",
            scip_symbol=f"file::{relative_path}",
            display_name=Path(relative_path).name,
            kind="File",
            relative_path=relative_path,
            start_line=1,
            end_line=max(1, len(content.splitlines())),
            signature=relative_path,
            docstring="",
            snippet="\n".join(content.splitlines()[: min(40, len(content.splitlines()))]),
        )
        symbols.append(file_symbol)
        symbols_by_key[file_symbol.symbol_id] = file_symbol

        declared_symbol_ids: set[str] = set()
        declared_symbol_kinds: dict[str, str] = {}
        for info in document.symbols:
            if info.symbol.startswith("file::"):
                continue
            normalized_symbol_id = _normalize_symbol_id(relative_path, info.symbol)
            declared_symbol_ids.add(normalized_symbol_id)
            declared_symbol_kinds[normalized_symbol_id] = _kind_name(info.kind)

        defs: list[tuple[tuple[int, int, int, int], str, int]] = []
        occurrences = list(document.occurrences)
        for occurrence in occurrences:
            role = getattr(occurrence, "symbol_roles", getattr(occurrence, "role", 0))
            if role & scip_pb2.Definition:
                full_range = tuple(occurrence.enclosing_range or occurrence.range or [0, 0, 0, 0])
                normalized_definition_id = _normalize_symbol_id(relative_path, occurrence.symbol)
                priority = _definition_priority(
                    normalized_definition_id,
                    declared_symbol_kinds.get(normalized_definition_id, "Unknown"),
                )
                defs.append((full_range, normalized_definition_id, priority))
        def_ranges[relative_path] = defs

        for info in document.symbols:
            if info.symbol.startswith("file::"):
                continue
            normalized_symbol_id = _normalize_symbol_id(relative_path, info.symbol)
            if normalized_symbol_id in symbols_by_key:
                continue
            occurrence = next(
                (
                    occ
                    for occ in occurrences
                    if _normalize_symbol_id(relative_path, occ.symbol) == normalized_symbol_id
                    and (getattr(occ, "symbol_roles", 0) & scip_pb2.Definition)
                ),
                None,
            )
            start_line, end_line = _range_to_lines(list(occurrence.enclosing_range or occurrence.range if occurrence else []))
            signature = info.signature_documentation.text or info.display_name or info.symbol
            docstring = "\n".join(part for part in info.documentation if part)
            snippet = _slice_snippet(content, start_line, end_line) or signature
            kind = _kind_name(info.kind)
            if kind == "Unknown":
                kind = _infer_kind(info.symbol, docstring, signature, snippet)
            definition_position = tuple(occurrence.enclosing_range or occurrence.range or [0, 0, 0, 0]) if occurrence else (0, 0, 0, 0)
            derived_enclosing_symbol_id = _best_enclosing_definition(
                defs,
                definition_position,
                exclude_symbol_id=normalized_symbol_id,
            )
            enclosing_from_info = ""
            if info.enclosing_symbol:
                candidate = _normalize_symbol_id(relative_path, info.enclosing_symbol)
                if candidate == f"file::{relative_path}" or candidate in declared_symbol_ids:
                    enclosing_from_info = candidate
            parsed_parent_symbol_id = ""
            parsed_parent = _parent_symbol_from_scip_symbol(info.symbol)
            if parsed_parent:
                candidate = _normalize_symbol_id(relative_path, parsed_parent)
                if candidate in declared_symbol_ids:
                    parsed_parent_symbol_id = candidate
            record = SymbolRecord(
                project_id=project_id,
                symbol_id=normalized_symbol_id,
                scip_symbol=info.symbol,
                display_name=clean_symbol_display_name(
                    symbol=info.symbol,
                    display_name=info.display_name,
                    docstring=docstring,
                    relative_path=relative_path,
                ),
                kind=kind,
                relative_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                docstring=docstring,
                snippet=snippet,
                enclosing_symbol_id=(
                    derived_enclosing_symbol_id
                    if derived_enclosing_symbol_id is not None
                    else enclosing_from_info
                    if enclosing_from_info
                    else parsed_parent_symbol_id
                    if parsed_parent_symbol_id
                    else f"file::{relative_path}"
                ),
            )
            symbols.append(record)
            symbols_by_key[record.symbol_id] = record
            edges.append(
                EdgeRecord(
                    project_id=project_id,
                    source_symbol_id=record.enclosing_symbol_id or f"file::{relative_path}",
                    target_symbol_id=record.symbol_id,
                    edge_type="contains",
                    weight=edge_weights["contains"],
                )
            )

    seen_edges: set[tuple[str, str, str]] = {(edge.source_symbol_id, edge.target_symbol_id, edge.edge_type) for edge in edges}

    for document in index.documents:
        abs_path = str((source_root / document.relative_path).resolve())
        try:
            normalized_relative_path = Path(abs_path).resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            normalized_relative_path = document.relative_path
        definitions = def_ranges[normalized_relative_path]
        occurrences = list(document.occurrences)
        for occurrence in occurrences:
            role = getattr(occurrence, "symbol_roles", getattr(occurrence, "role", 0))
            if role & scip_pb2.Definition:
                continue
            target_symbol_id = _normalize_symbol_id(normalized_relative_path, occurrence.symbol)
            if target_symbol_id not in symbols_by_key:
                continue
            source_symbol_id = f"file::{normalized_relative_path}"
            position = tuple(occurrence.enclosing_range or occurrence.range or [0, 0, 0, 0])
            enclosing_symbol_id = _best_enclosing_definition(definitions, position)
            if enclosing_symbol_id is not None and enclosing_symbol_id in symbols_by_key:
                source_symbol_id = enclosing_symbol_id
            edge_type = _classify_edge(occurrence.syntax_kind)
            edge_key = (source_symbol_id, target_symbol_id, edge_type)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append(
                EdgeRecord(
                    project_id=project_id,
                    source_symbol_id=source_symbol_id,
                    target_symbol_id=target_symbol_id,
                    edge_type=edge_type,
                    weight=edge_weights.get(edge_type, 0.5),
                )
            )

    return ParsedScip(files=files, symbols=symbols, edges=edges)
