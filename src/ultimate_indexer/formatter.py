from __future__ import annotations

from pathlib import Path

from .models import FileGroup
from .storage import Storage


CLASS_LIKE_KINDS = {"Class", "Struct", "Interface", "Trait", "Enum"}


def _comment_prefix(relative_path: str) -> str:
    if relative_path.endswith(".py"):
        return "#"
    if relative_path.endswith((".sql", ".hs")):
        return "--"
    return "//"


def _meaningful_doc_lines(docstring: str) -> list[str]:
    if not docstring.strip():
        return []
    lines: list[str] = []
    in_fence = False
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        lowered = line.lower()
        if lowered.startswith(("function ", "func ", "method ", "type ", "class ", "interface ", "enum ")):
            continue
        if lowered.startswith(("struct field ", "(property) ", "(parameter) ", "var ", "let ", "const ")):
            continue
        lines.append(line)
    return lines


def _render_docstring(docstring: str, comment_prefix: str) -> list[str]:
    lines = _meaningful_doc_lines(docstring)
    if not lines:
        return []
    return [f"{comment_prefix} {line}" for line in lines]


def _first_doc_or_signature_line(docstring: str, signature: str) -> str:
    for source in (docstring, signature):
        for raw_line in source.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            return line
    return ""


def _pretty_signature(kind: str, display_name: str, signature: str, docstring: str, snippet: str) -> str:
    first_line = _first_doc_or_signature_line(docstring, signature)
    lowered = first_line.lower()
    if lowered.startswith("struct field "):
        return first_line[len("struct field ") :].strip()
    if lowered.startswith("(property) "):
        return first_line[len("(property) ") :].strip()
    if lowered.startswith("(parameter) "):
        return first_line[len("(parameter) ") :].strip()
    if first_line and not first_line.startswith(("scip-", "local ")):
        return first_line
    snippet_line = next((line.strip() for line in snippet.splitlines() if line.strip()), "")
    if snippet_line:
        return snippet_line
    return display_name or signature


def _is_composite_symbol(kind: str, signature: str, docstring: str, snippet: str) -> bool:
    if kind in CLASS_LIKE_KINDS:
        return True
    first_line = _first_doc_or_signature_line(docstring, signature).lower()
    if first_line.startswith(("type ", "class ", "interface ", "enum ")):
        return True
    snippet_line = next((line.strip().lower() for line in snippet.splitlines() if line.strip()), "")
    return snippet_line.startswith(("type ", "class ", "interface ", "enum "))


def _render_composite_snippet(signature: str, snippet: str, comment_prefix: str) -> list[str]:
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    if not snippet_lines:
        return [signature.strip()]
    first_line = signature.strip() or snippet_lines[0].strip()
    remaining_lines = snippet_lines[1:] if snippet_lines and snippet_lines[0].strip() == first_line else snippet_lines[1:]
    rows = [first_line]
    rows.extend(remaining_lines)
    if len(rows) == 1:
        rows.append(f"    {comment_prefix} definition omitted")
    return rows


def _render_function_block(signature: str, snippet: str, comment_prefix: str) -> list[str]:
    lines = [signature.strip()]
    snippet_line_count = max(0, len([line for line in snippet.splitlines() if line.strip()]))
    skipped = max(0, snippet_line_count - 1)
    if skipped > 0:
        lines.append(f"{comment_prefix} skipped {skipped} rows")
    return lines


def _render_class_interface(
    storage: Storage,
    project_id: str,
    symbol_id: str,
    signature: str,
    comment_prefix: str,
    snippet: str,
) -> list[str]:
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    child_signatures: list[str] = []
    children = storage.get_symbol_children(project_id, symbol_id)
    for child in children:
        child_signature = _pretty_signature(
            kind=str(child["kind"]),
            display_name=str(child["display_name"]),
            signature=str(child["signature"]),
            docstring=str(child["docstring"]),
            snippet=str(child["snippet"]),
        ).strip()
        if not child_signature:
            continue
        child_signatures.append(child_signature)
    if len(snippet_lines) > 1:
        rows = _render_composite_snippet(signature, snippet, comment_prefix)
        rendered_text = "\n".join(rows)
        for child_signature in child_signatures:
            if child_signature in rendered_text:
                continue
            rows.append(f"    {child_signature}")
        return rows

    rows = [signature.strip()]
    for child_signature in child_signatures:
        rows.append(f"    {child_signature}")
    return rows


def format_groups(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    max_symbols_per_file: int = 3,
) -> str:
    lines: list[str] = []
    symbol_rows = storage.get_symbol_rows(project_id)
    for group in groups:
        comment_prefix = _comment_prefix(group.relative_path)
        lines.append(f"// {group.relative_path}")
        selected = [
            symbol
            for symbol in group.symbols
            if symbol.kind != "File"
        ][:max_symbols_per_file]
        if not selected and group.symbols:
            selected = group.symbols[:1]
        for symbol in selected:
            row = symbol_rows.get(symbol.symbol_id)
            kind = str(row["kind"]) if row is not None else symbol.kind
            display_name = str(row["display_name"]) if row is not None else symbol.display_name
            signature = str(row["signature"]) if row is not None else symbol.signature
            docstring = str(row["docstring"]) if row is not None else symbol.docstring
            snippet = str(row["snippet"]) if row is not None else symbol.snippet
            pretty_signature = _pretty_signature(kind, display_name, signature, docstring, snippet)
            lines.extend(_render_docstring(docstring, comment_prefix))
            if _is_composite_symbol(kind, signature, docstring, snippet):
                lines.extend(
                    _render_class_interface(
                        storage=storage,
                        project_id=project_id,
                        symbol_id=symbol.symbol_id,
                        signature=pretty_signature,
                        comment_prefix=comment_prefix,
                        snippet=snippet,
                    )
                )
            else:
                lines.extend(
                    _render_function_block(
                        signature=pretty_signature,
                        snippet=snippet,
                        comment_prefix=comment_prefix,
                    )
                )
            lines.append("")
        if lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


def truncate_text(text: str, max_chars: int, *, note_prefix: str = "//") -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = f"\n\n{note_prefix} output truncated for MCP"
    available = max_chars - len(suffix)
    if available <= 0:
        return text[:max_chars]
    return f"{text[:available].rstrip()}{suffix}"


def format_groups_compact(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    *,
    max_files: int = 5,
    max_symbols_per_file: int = 2,
    max_chars: int = 4_000,
) -> str:
    text = format_groups(
        storage=storage,
        project_id=project_id,
        groups=groups[:max_files],
        max_symbols_per_file=max_symbols_per_file,
    )
    return truncate_text(text, max_chars)


def format_top_symbols(rows: list, *, include_scores: bool = True) -> str:
    lines = []
    for index, row in enumerate(rows, start=1):
        line = f"{index}. {row['display_name']} [{row['kind']}] {row['relative_path']}"
        if include_scores:
            line += f" score={float(row['global_rank']):.5f}"
        lines.append(line)
    return "\n".join(lines)
