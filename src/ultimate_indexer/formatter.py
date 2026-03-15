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


def _render_docstring(docstring: str) -> list[str]:
    if not docstring.strip():
        return []
    return ['"""', docstring.strip(), '"""']


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
) -> list[str]:
    rows = [signature.strip()]
    children = storage.get_symbol_children(project_id, symbol_id)
    if children:
        for child in children:
            child_signature = str(child["signature"]).strip()
            if not child_signature:
                continue
            rows.append(f"    {child_signature}")
    else:
        rows.append(f"    {comment_prefix} interface omitted")
    return rows


def format_groups(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    max_symbols_per_file: int = 3,
) -> str:
    lines: list[str] = []
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
            lines.extend(_render_docstring(symbol.docstring))
            if symbol.kind in CLASS_LIKE_KINDS:
                lines.extend(
                    _render_class_interface(
                        storage=storage,
                        project_id=project_id,
                        symbol_id=symbol.symbol_id,
                        signature=symbol.signature,
                        comment_prefix=comment_prefix,
                    )
                )
            else:
                lines.extend(
                    _render_function_block(
                        signature=symbol.signature or symbol.display_name,
                        snippet=symbol.snippet,
                        comment_prefix=comment_prefix,
                    )
                )
            lines.append("")
        if lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


def format_top_symbols(rows: list) -> str:
    lines = []
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index}. {row['display_name']} [{row['kind']}] {row['relative_path']} score={float(row['global_rank']):.5f}"
        )
    return "\n".join(lines)
