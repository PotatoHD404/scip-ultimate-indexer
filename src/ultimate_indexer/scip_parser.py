from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_EDGE_WEIGHTS
from .embeddings import hash_text
from .models import EdgeRecord, FileRecord, SymbolRecord
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
    return name.replace("Unspecified", "").replace("Method", "Method") or "Unknown"


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
    def_ranges: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {}

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

        defs: list[tuple[tuple[int, int, int, int], str]] = []
        occurrences = list(document.occurrences)
        for occurrence in occurrences:
            role = getattr(occurrence, "symbol_roles", getattr(occurrence, "role", 0))
            if role & scip_pb2.Definition:
                full_range = tuple(occurrence.enclosing_range or occurrence.range or [0, 0, 0, 0])
                defs.append((full_range, _normalize_symbol_id(relative_path, occurrence.symbol)))
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
            record = SymbolRecord(
                project_id=project_id,
                symbol_id=normalized_symbol_id,
                scip_symbol=info.symbol,
                display_name=info.display_name or info.symbol.split(":")[-1],
                kind=_kind_name(info.kind),
                relative_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                docstring=docstring,
                snippet=snippet,
                enclosing_symbol_id=(
                    _normalize_symbol_id(relative_path, info.enclosing_symbol)
                    if info.enclosing_symbol
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
            best_span = None
            for full_range, symbol_id in definitions:
                start_line, start_col, end_line, end_col = (full_range + (0, 0, 0, 0))[:4]
                pos_line = position[0]
                pos_col = position[1] if len(position) > 1 else 0
                if (start_line < pos_line or (start_line == pos_line and start_col <= pos_col)) and (
                    end_line > pos_line or (end_line == pos_line and end_col >= pos_col)
                ):
                    span = (end_line - start_line, end_col - start_col)
                    if best_span is None or span < best_span[0]:
                        best_span = (span, symbol_id)
            if best_span is not None:
                source_symbol_id = best_span[1]
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
