from __future__ import annotations

from pathlib import Path

from .models import FileGroup
from .storage import Storage


CLASS_LIKE_KINDS = {"Class", "Struct", "Interface", "Trait", "Enum"}
TYPE_LIKE_KINDS = CLASS_LIKE_KINDS | {"TypeAlias"}


def _comment_prefix(relative_path: str) -> str:
    if relative_path.endswith(".py"):
        return "#"
    if relative_path.endswith((".sql", ".hs")):
        return "--"
    return "//"


def _is_go_path(relative_path: str) -> bool:
    return relative_path.endswith(".go")


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


def _first_snippet_line(snippet: str) -> str:
    return next((line.strip() for line in snippet.splitlines() if line.strip()), "")


def _is_raw_symbol_text(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("scip-", "local ")):
        return True
    return "`" in stripped and "/" in stripped


def _is_go_type_header(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("type ") or stripped.startswith("interface ")


def _pretty_signature(
    kind: str,
    display_name: str,
    signature: str,
    docstring: str,
    snippet: str,
) -> str:
    first_line = _first_doc_or_signature_line(docstring, signature)
    signature_line = _first_doc_or_signature_line(signature, "")
    snippet_line = _first_snippet_line(snippet)
    lowered = first_line.lower()
    if lowered.startswith("struct field "):
        return first_line[len("struct field ") :].strip()
    if lowered.startswith("(property) "):
        return first_line[len("(property) ") :].strip()
    if lowered.startswith("(parameter) "):
        return first_line[len("(parameter) ") :].strip()
    if kind in {"Function", "Method"} and signature_line and not signature_line.startswith(("scip-", "local ")):
        return signature_line
    if kind in TYPE_LIKE_KINDS:
        if snippet_line and _is_go_type_header(snippet_line):
            return snippet_line
        if signature_line and _is_go_type_header(signature_line):
            return signature_line
        if first_line and _is_go_type_header(first_line):
            return first_line
        if display_name:
            return display_name
    if first_line and not first_line.startswith(("scip-", "local ")):
        return first_line
    if snippet_line:
        return snippet_line
    return display_name or signature


def _is_composite_symbol(kind: str, signature: str, docstring: str, snippet: str) -> bool:
    if kind in TYPE_LIKE_KINDS:
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


def _render_function_block(
    signature: str,
    snippet: str,
    comment_prefix: str,
    *,
    max_preview_lines: int = 4,
) -> list[str]:
    signature_line = signature.strip()
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    if not snippet_lines:
        return [signature_line]

    header_line = signature_line or snippet_lines[0].strip()
    rows = [header_line]

    preview_budget = max(0, max_preview_lines - 1)
    remaining_lines: list[str] = []
    if len(snippet_lines) > 1:
        remaining_lines = snippet_lines[1:]
    elif snippet_lines[0].strip() != header_line:
        remaining_lines = [snippet_lines[0]]

    rows.extend(remaining_lines[:preview_budget])
    skipped = max(0, len(snippet_lines) - len(rows))
    if skipped > 0:
        rows.append(f"{comment_prefix} skipped {skipped} rows")
    return rows


def _split_composite_children(
    child_renderings: list[tuple[str, str, str]],
) -> tuple[list[str], list[tuple[str, str]]]:
    member_signatures: list[str] = []
    method_renderings: list[tuple[str, str]] = []
    for child_kind, child_signature, child_snippet in child_renderings:
        if child_kind in {"Variable", "Parameter", "Unknown", "Module"}:
            continue
        if child_kind in {"Method", "Function"}:
            method_renderings.append((child_signature, child_snippet))
            continue
        member_signatures.append(child_signature)
    return member_signatures, method_renderings


def _inject_before_closing_brace(rows: list[str], additions: list[str]) -> list[str]:
    if not additions:
        return rows
    closing_index = len(rows)
    if rows and rows[-1].strip() == "}":
        closing_index -= 1
    return rows[:closing_index] + additions + rows[closing_index:]


def _render_go_composite_block(
    kind: str,
    display_name: str,
    signature: str,
    snippet: str,
    child_renderings: list[tuple[str, str, str]],
    comment_prefix: str,
) -> list[str]:
    signature_line = signature.strip()
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    header_line = signature_line or (snippet_lines[0].strip() if snippet_lines else "")
    member_signatures, method_renderings = _split_composite_children(child_renderings)
    if not _is_go_type_header(header_line):
        if kind == "Interface":
            header_line = f"type {display_name} interface"
        elif member_signatures:
            header_line = f"type {display_name} struct"
        else:
            header_line = f"type {display_name}"
    lowered_header = header_line.lower()
    is_struct = " struct" in lowered_header
    is_interface = " interface" in lowered_header
    if is_interface:
        member_signatures.extend(
            method_signature for method_signature, _ in method_renderings
        )
        method_renderings = []

    if len(snippet_lines) > 1:
        rows = list(snippet_lines)
        rendered_text = "\n".join(rows)
        missing_members = [
            signature
            for signature in member_signatures
            if signature not in rendered_text
        ]
        if missing_members and (is_struct or is_interface):
            rows = _inject_before_closing_brace(
                rows,
                [f"    {member_signature}" for member_signature in missing_members],
            )
        rendered_text = "\n".join(rows)
        for method_signature, method_snippet in method_renderings:
            if is_interface or method_signature in rendered_text:
                continue
            rows.append("")
            rows.extend(
                _render_function_block(
                    method_signature,
                    method_snippet,
                    comment_prefix,
                )
            )
        return rows

    if is_struct or is_interface:
        opening = header_line if header_line.endswith("{") else f"{header_line} {{"
        rows = [opening]
        if member_signatures:
            rows.extend(f"    {member_signature}" for member_signature in member_signatures)
        rows.append("}")
        if not is_interface:
            for method_signature, method_snippet in method_renderings:
                rows.append("")
                rows.extend(
                    _render_function_block(
                        method_signature,
                        method_snippet,
                        comment_prefix,
                    )
                )
        return rows

    rows = [signature_line]
    for child_signature in member_signatures:
        rows.append(f"    {child_signature}")
    for method_signature, method_snippet in method_renderings:
        rows.append("")
        rows.extend(
            _render_function_block(
                method_signature,
                method_snippet,
                comment_prefix,
            )
        )
    return rows


def _render_class_interface(
    storage: Storage,
    project_id: str,
    symbol_id: str,
    kind: str,
    display_name: str,
    signature: str,
    comment_prefix: str,
    snippet: str,
    relative_path: str,
) -> list[str]:
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    child_renderings: list[tuple[str, str, str]] = []
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
        child_renderings.append((str(child["kind"]), child_signature, str(child["snippet"])))
    if _is_go_path(relative_path):
        return _render_go_composite_block(
            kind=kind,
            display_name=display_name,
            signature=signature,
            snippet=snippet,
            child_renderings=child_renderings,
            comment_prefix=comment_prefix,
        )
    if len(snippet_lines) > 1:
        rows = _render_composite_snippet(signature, snippet, comment_prefix)
        rendered_text = "\n".join(rows)
        for child_kind, child_signature, child_snippet in child_renderings:
            if child_signature in rendered_text:
                continue
            if _is_go_path(relative_path) and child_kind in {"Method", "Function"}:
                rows.append("")
                rows.extend(_render_function_block(child_signature, child_snippet, comment_prefix))
                continue
            rows.append(f"    {child_signature}")
        return rows

    rows = [signature.strip()]
    for child_kind, child_signature, child_snippet in child_renderings:
        if _is_go_path(relative_path) and child_kind in {"Method", "Function"}:
            rows.append("")
            rows.extend(_render_function_block(child_signature, child_snippet, comment_prefix))
            continue
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
            doc_lines = [
                line
                for line in _render_docstring(docstring, comment_prefix)
                if line.removeprefix(f"{comment_prefix} ").strip() != pretty_signature.strip()
            ]
            lines.extend(doc_lines)
            if _is_composite_symbol(kind, signature, docstring, snippet):
                lines.extend(
                    _render_class_interface(
                        storage=storage,
                        project_id=project_id,
                        symbol_id=symbol.symbol_id,
                        kind=kind,
                        display_name=display_name,
                        signature=pretty_signature,
                        comment_prefix=comment_prefix,
                        snippet=snippet,
                        relative_path=group.relative_path,
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
