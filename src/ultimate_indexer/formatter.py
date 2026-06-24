from __future__ import annotations

from pathlib import Path
import re

from .models import FileGroup, RankedSymbol, TreeScoreNode
from .storage import Storage

# ---------------------------------------------------------------------------
# Token counting — tiktoken (Qwen2-compatible cl100k_base) with char fallback
# ---------------------------------------------------------------------------
# Qwen2 uses a BPE tokenizer very close to GPT-4's cl100k_base.
#
# The encoding is built LAZILY on first use: tiktoken downloads the BPE table
# over the network when its cache is cold, so constructing it at import time
# can block process startup (the MCP server must answer the stdio handshake
# immediately). A failed load — offline, blocked CDN, full disk — permanently
# degrades to the character-ratio estimate instead of raising.
_CHARS_PER_TOKEN = 3.5  # empirical ratio for mixed code/prose
_enc = None
_enc_failed = False


def _char_estimate(text: str) -> int:
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


def count_tokens(text: str) -> int:
    """Return the token count for *text* using tiktoken cl100k_base.

    cl100k_base is the GPT-4 / Qwen2-compatible BPE encoding. Counts are
    within ±5 % of the true Qwen2 count for typical code content. Falls back
    to a character-ratio estimate when tiktoken (or its BPE download) is
    unavailable.
    """
    global _enc, _enc_failed
    if _enc is None and not _enc_failed:
        try:
            import tiktoken as _tiktoken

            _enc = _tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc_failed = True
    if _enc is None:
        return _char_estimate(text)
    try:
        return len(_enc.encode(text, disallowed_special=()))
    except Exception:
        return _char_estimate(text)


CLASS_LIKE_KINDS = {"Class", "Struct", "Interface", "Trait", "Enum"}
TYPE_LIKE_KINDS = CLASS_LIKE_KINDS | {"TypeAlias"}
DOC_KINDS = {"Document", "Section"}


def _comment_prefix(relative_path: str) -> str:
    if relative_path.endswith(".py"):
        return "#"
    if relative_path.endswith((".sql", ".hs")):
        return "--"
    return "//"


def _is_go_path(relative_path: str) -> bool:
    return relative_path.endswith(".go")


def _is_external_path(relative_path: str) -> bool:
    return relative_path.startswith("_external/")


def _path_label(relative_path: str, start_line: int = 0) -> str:
    """Return a human-readable location string for a symbol header comment.

    External library stubs render as ``[library] <package>`` since they have
    no real source file.  Project symbols render as ``<path>:<line>``.
    """
    if _is_external_path(relative_path):
        pkg = relative_path[len("_external/"):]
        return f"[library] {pkg}"
    if start_line > 0:
        return f"{relative_path}:{start_line}"
    return relative_path


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
    prefer_snippet_header: bool = True,
) -> list[str]:
    signature_line = signature.strip()
    snippet_lines = [line.rstrip() for line in snippet.splitlines() if line.strip()]
    if not snippet_lines:
        return [signature_line]

    snippet_header = snippet_lines[0].strip()
    if prefer_snippet_header and snippet_header and not _is_raw_symbol_text(snippet_header):
        header_line = snippet_header
        remaining_lines = snippet_lines[1:]
    else:
        header_line = signature_line or snippet_header
        if snippet_lines and snippet_lines[0].strip() == header_line:
            remaining_lines = snippet_lines[1:]
        else:
            remaining_lines = snippet_lines
    rows = [header_line]

    preview_budget = max(0, max_preview_lines - 1)
    rows.extend(remaining_lines[:preview_budget])
    skipped = max(0, len(remaining_lines) - preview_budget)
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
                        prefer_snippet_header=not kind.startswith("Artifact"),
                    )
                )
            lines.append("")
        if lines and lines[-1] != "":
            lines.append("")
    return "\n".join(lines).strip()


def truncate_to_tokens(text: str, max_tokens: int, *, note_prefix: str = "//") -> str:
    """Trim *text* to fit within *max_tokens* as counted by ``count_tokens``.

    Lines are removed from the end until the remaining text fits, so the
    output is always a clean line-by-line prefix of the input.  A trailing
    comment marks the cut so consumers know the output is incomplete.
    Pass ``max_tokens <= 0`` to disable trimming.
    """
    if max_tokens <= 0 or count_tokens(text) <= max_tokens:
        return text
    note = f"\n{note_prefix} ... truncated"
    note_tokens = count_tokens(note)
    budget = max_tokens - note_tokens
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = count_tokens(line + "\n")
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept) + note


def format_groups_compact(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    *,
    max_files: int = 5,
    max_symbols_per_file: int = 2,
    max_tokens: int = 1_200,
) -> str:
    text = format_groups(
        storage=storage,
        project_id=project_id,
        groups=groups[:max_files],
        max_symbols_per_file=max_symbols_per_file,
    )
    return truncate_to_tokens(text, max_tokens)


def format_top_symbols(rows: list, *, include_scores: bool = True) -> str:
    lines = []
    for index, row in enumerate(rows, start=1):
        line = f"{index}. {row['display_name']} [{row['kind']}] {row['relative_path']}"
        if include_scores:
            line += f" score={float(row['global_rank']):.5f}"
        lines.append(line)
    return "\n".join(lines)


def _qualified_from_scip_symbol(symbol: str) -> str:
    if not symbol:
        return ""
    if symbol.startswith(
        (
            "local ",
            "local::",
            "file::",
            "module::",
            "section::",
            "artifact::",
            "artifact-item::",
        )
    ):
        return ""
    parts = symbol.split()
    if len(parts) < 4:
        return ""
    package = parts[2].strip()
    descriptor = parts[-1].strip()
    if not descriptor:
        return ""

    clean = descriptor.rstrip(".")
    clean = clean.replace("().", "").replace("()", "")
    descriptor_parts = [
        part.strip("`")
        for part in re.split(r"[#:/]", clean)
        if part and part.strip("`")
    ]
    if not descriptor_parts:
        return ""
    descriptor_path = ".".join(descriptor_parts)

    package_clean = re.sub(
        r"^(github\.com|gitlab\.com|golang\.org/x|bitbucket\.org)/[^/]+/",
        "",
        package,
    )
    package_parts = [part for part in package_clean.replace("\\", "/").split("/") if part]
    if len(package_parts) >= 2:
        package_short = "/".join(package_parts[-2:])
    elif package_parts:
        package_short = package_parts[-1]
    else:
        package_short = ""

    if package_short and package_short not in descriptor_path:
        return f"{package_short}.{descriptor_path}"
    return descriptor_path


def _qualified_display_name(symbol_id: str, fallback_name: str, symbol_rows: dict[str, object]) -> str:
    row = symbol_rows.get(symbol_id)
    if row is None:
        return fallback_name
    names = [str(row["display_name"])]
    current = row
    seen: set[str] = {symbol_id}
    while True:
        enclosing_symbol_id = str(current["enclosing_symbol_id"] or "")
        if not enclosing_symbol_id or enclosing_symbol_id in seen:
            break
        seen.add(enclosing_symbol_id)
        parent = symbol_rows.get(enclosing_symbol_id)
        if parent is None:
            break
        parent_kind = str(parent["kind"])
        if parent_kind in {"File", "Section", "Unknown"}:
            break
        names.append(str(parent["display_name"]))
        current = parent
    qualified_parts = list(reversed([name for name in names if name]))
    deduped_parts: list[str] = []
    for part in qualified_parts:
        if deduped_parts and deduped_parts[-1] == part:
            continue
        deduped_parts.append(part)
    qualified_from_hierarchy = ".".join(deduped_parts)

    try:
        scip_symbol = str(row["scip_symbol"] or symbol_id)
    except Exception:
        scip_symbol = symbol_id
    fallback_from_symbol = _qualified_from_scip_symbol(scip_symbol)
    if fallback_from_symbol:
        hierarchy_depth = qualified_from_hierarchy.count(".") + qualified_from_hierarchy.count("/")
        fallback_depth = fallback_from_symbol.count(".") + fallback_from_symbol.count("/")
        if fallback_depth > hierarchy_depth:
            return fallback_from_symbol
    if qualified_from_hierarchy:
        return qualified_from_hierarchy
    if fallback_from_symbol:
        return fallback_from_symbol
    return fallback_name


def qualified_display_name(symbol_id: str, fallback_name: str, symbol_rows: dict[str, object]) -> str:
    return _qualified_display_name(symbol_id, fallback_name, symbol_rows)


def _symbol_declaration_block(
    storage: Storage,
    project_id: str,
    symbol: RankedSymbol,
    symbol_rows: dict[str, object],
    *,
    include_members: bool,
) -> list[str]:
    row = symbol_rows.get(symbol.symbol_id)
    kind = str(row["kind"]) if row is not None else symbol.kind
    display_name = str(row["display_name"]) if row is not None else symbol.display_name
    signature = str(row["signature"]) if row is not None else symbol.signature
    docstring = str(row["docstring"]) if row is not None else symbol.docstring
    snippet = str(row["snippet"]) if row is not None else symbol.snippet
    relative_path = str(row["relative_path"]) if row is not None else symbol.relative_path
    pretty_signature = _pretty_signature(kind, display_name, signature, docstring, snippet)
    comment_prefix = _comment_prefix(relative_path)

    # External library stubs have no source; render just the signature name.
    if _is_external_path(relative_path):
        return [display_name]

    lines: list[str] = []
    doc_lines = [
        line
        for line in _render_docstring(docstring, comment_prefix)
        if line.removeprefix(f"{comment_prefix} ").strip() != pretty_signature.strip()
    ]
    lines.extend(doc_lines)
    if include_members and _is_composite_symbol(kind, signature, docstring, snippet):
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
                relative_path=relative_path,
            )
        )
    else:
        lines.extend(
            _render_function_block(
                signature=pretty_signature,
                snippet=snippet,
                comment_prefix=comment_prefix,
                prefer_snippet_header=not kind.startswith("Artifact"),
            )
        )
    return lines


def format_search_symbols_codegraph(
    storage: Storage,
    project_id: str,
    query: str,
    groups: list[FileGroup],
    *,
    max_results: int = 10,
    max_tokens: int = 0,
) -> str:
    symbol_rows = storage.get_symbol_rows(project_id)
    symbol_map: dict[str, RankedSymbol] = {}
    for group in groups:
        for symbol in group.symbols:
            existing = symbol_map.get(symbol.symbol_id)
            if existing is None or symbol.score > existing.score:
                symbol_map[symbol.symbol_id] = symbol

    ranked = sorted(
        symbol_map.values(),
        key=lambda item: (-item.score, item.relative_path, item.display_name),
    )
    selected: list[RankedSymbol] = []
    selected_ids: set[str] = set()
    seen_paths: set[str] = set()

    # First pass keeps the strongest symbol from each file so mixed doc/code
    # results do not collapse back into multiple hits from the same document.
    for symbol in ranked:
        if symbol.relative_path in seen_paths:
            continue
        selected.append(symbol)
        selected_ids.add(symbol.symbol_id)
        seen_paths.add(symbol.relative_path)
        if len(selected) >= max_results:
            break

    if len(selected) < max_results:
        for symbol in ranked:
            if symbol.symbol_id in selected_ids:
                continue
            selected.append(symbol)
            selected_ids.add(symbol.symbol_id)
            if len(selected) >= max_results:
                break

    ranked = selected
    lines = [f"// Search: {query}", ""]
    for symbol in ranked:
        row = symbol_rows.get(symbol.symbol_id)
        kind = str(row["kind"]) if row is not None else symbol.kind
        relative_path = str(row["relative_path"]) if row is not None else symbol.relative_path
        start_line = int(row["start_line"]) if row is not None else 0
        qualified = _qualified_display_name(symbol.symbol_id, symbol.display_name, symbol_rows)
        lines.append(f"// {qualified}  ({kind})  {_path_label(relative_path, start_line)}")
        include_members = kind in {"Class", "Struct", "Interface", "Trait", "Enum", "TypeAlias"}
        lines.extend(
            _symbol_declaration_block(
                storage,
                project_id,
                symbol,
                symbol_rows,
                include_members=include_members,
            )
        )
        lines.append("")
    return truncate_to_tokens("\n".join(lines).strip(), max_tokens)


def format_important_symbols_codegraph(
    storage: Storage,
    project_id: str,
    rows: list,
    *,
    max_tokens: int = 0,
) -> str:
    symbol_rows = storage.get_symbol_rows(project_id)
    lines = [f"// Top {len(rows)} symbols", ""]
    for index, row in enumerate(rows, start=1):
        symbol_id = str(row["symbol_id"])
        symbol = RankedSymbol(
            symbol_id=symbol_id,
            relative_path=str(row["relative_path"]),
            display_name=str(row["display_name"]),
            kind=str(row["kind"]),
            score=float(row["global_rank"]),
            signature=str(row["signature"]),
            docstring=str(row["docstring"]),
            snippet=str(row["snippet"]),
        )
        start_line = int(row["start_line"]) if "start_line" in row.keys() else 0
        qualified = _qualified_display_name(symbol_id, symbol.display_name, symbol_rows)
        lines.append(f"// #{index}  {qualified}  ({symbol.kind})  {_path_label(symbol.relative_path, start_line)}")
        include_members = symbol.kind in {"Class", "Struct", "Interface", "Trait", "Enum", "TypeAlias"}
        lines.extend(
            _symbol_declaration_block(
                storage,
                project_id,
                symbol,
                symbol_rows,
                include_members=include_members,
            )
        )
        lines.append("")
    return truncate_to_tokens("\n".join(lines).strip(), max_tokens)


def _tree_label(node: TreeScoreNode, *, include_value_details: bool = False) -> str:
    if node.node_type == "dir":
        details: list[str] = []
        if include_value_details:
            details.append(f"acc={node.raw_score:.5f}")
        detail_text = f" ({', '.join(details)})" if details else ""
        return f"{node.name}/{detail_text}"
    details: list[str] = []
    if include_value_details:
        details.append(f"value={node.raw_score:.5f}")
    if node.useful_symbol_count > 0:
        details.append(f"{node.useful_symbol_count} syms")
    if node.source_kind != "code":
        details.append(node.source_kind)
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{node.name}{detail_text}"


def format_documentation_section(
    storage: Storage,
    project_id: str,
    relative_path: str,
    max_tokens: int = 1_200,
) -> str:
    """Format a documentation file for output, trimmed to *max_tokens*.

    Markdown files are returned as-is (trimmed by token budget).  OpenAPI
    specs are rendered as a structured comment hierarchy of Document/Section
    symbols before trimming.
    """
    file_row = storage.get_file(project_id, relative_path)
    if file_row is None:
        return f"// Documentation not found: {relative_path}"

    content = str(file_row["content"])

    if relative_path.endswith((".md", ".markdown")):
        return truncate_to_tokens(content, max_tokens)

    if relative_path.endswith((".yaml", ".yml", ".json")):
        lines = [f"# OpenAPI Specification: {relative_path}", ""]
        symbol_rows = storage.get_symbol_rows(project_id)
        file_symbols = [
            row for row in symbol_rows.values()
            if str(row["relative_path"]) == relative_path
            and str(row["kind"]) in DOC_KINDS
        ]
        docs = [s for s in file_symbols if str(s["kind"]) == "Document"]
        sections = [s for s in file_symbols if str(s["kind"]) == "Section"]
        for doc in docs:
            docstring = str(doc["docstring"])
            if docstring:
                lines.append(f"## {str(doc['display_name'])}")
                lines.append(docstring)
                lines.append("")
        for section in sorted(sections, key=lambda s: float(s["start_line"])):
            lines.append(f"### {str(section['display_name'])}")
            snippet = str(section["snippet"])[:500]
            if snippet:
                lines.append(snippet)
            lines.append("")
        return truncate_to_tokens("\n".join(lines), max_tokens)

    return truncate_to_tokens(content, max_tokens)


def _render_tree_lines(
    node: TreeScoreNode,
    prefix: str = "",
    *,
    is_last: bool = True,
    include_value_details: bool = False,
) -> list[str]:
    connector = "`-- " if is_last else "|-- "
    score_text = f"{node.score:6.2f}"
    lines = [f"{prefix}{connector}[{score_text}] {_tree_label(node, include_value_details=include_value_details)}"]
    child_prefix = f"{prefix}{'    ' if is_last else '|   '}"
    sorted_children = sorted(
        node.children,
        # Directories are sorted by accumulated descendant score, while files are
        # sorted by their own value score. Both use the normalized tree score.
        key=lambda child: (child.node_type != "dir", -child.score, child.name.lower()),
    )
    for index, child in enumerate(sorted_children):
        lines.extend(
            _render_tree_lines(
                child,
                child_prefix,
                is_last=index == len(sorted_children) - 1,
                include_value_details=include_value_details,
            )
        )
    return lines


def format_scored_tree(
    root: TreeScoreNode,
    *,
    max_tokens: int | None = 3_000,
    top_k: int | None = None,
    header_title: str = "Project tree scored by usefulness.",
    header_description: list[str] | None = None,
    include_value_details: bool = False,
) -> str:
    header_lines = [header_title]
    if top_k is not None and top_k > 0:
        header_lines.append(f"Showing top {top_k} files by PageRank-based score.")
    else:
        header_lines.append("Showing all files.")
    header_lines.extend(
        header_description
        or [
            "File score blends symbol rank, strongest symbol, and a small structure fallback.",
            "Directory score rolls up all descendants.",
        ]
    )
    header = "\n".join(header_lines) + "\n"
    body_lines = _render_tree_lines(
        root,
        prefix="",
        is_last=True,
        include_value_details=include_value_details,
    )
    text = header + "\n".join(body_lines)
    if max_tokens is None:
        return text
    return truncate_to_tokens(text, max_tokens)


# ---------------------------------------------------------------------------
# Context-window formatter — greedy token packing
# ---------------------------------------------------------------------------

# Symbol kinds shown in the symbol section (functions, named types, globals).
_CONTEXT_SYMBOL_KINDS = {
    "Function", "Method",
    "Interface", "Struct", "Class", "Trait", "Enum", "TypeAlias",
    "Constant", "Const", "Property",
}


def _context_symbol_block(
    row: object,
    symbol_rows: dict[str, object],
) -> str:
    """Render one symbol as a compact signature line with location comment.

    Format::

        // QualifiedName  (Kind)  path:line
        signature_or_display_name
    """
    symbol_id = str(row["symbol_id"])
    kind = str(row["kind"])
    relative_path = str(row["relative_path"])
    start_line = int(row["start_line"])
    qualified = _qualified_display_name(symbol_id, str(row["display_name"]), symbol_rows)
    # External library stubs: no source snippet, just the name.
    if _is_external_path(relative_path):
        return f"// {qualified}  ({kind})  {_path_label(relative_path)}\n{str(row['display_name'])}"
    sig = _pretty_signature(
        kind=kind,
        display_name=str(row["display_name"]),
        signature=str(row["signature"]),
        docstring=str(row["docstring"]),
        snippet=str(row["snippet"]),
    ).strip()
    # Include the first docstring line only when it adds information beyond the sig.
    doc_lines = _meaningful_doc_lines(str(row["docstring"]))
    doc_note = ""
    if doc_lines and doc_lines[0].strip() != sig:
        # Condense to a single brief comment (≤ 120 chars).
        brief = doc_lines[0][:120]
        comment_prefix = _comment_prefix(relative_path)
        doc_note = f"\n{comment_prefix} {brief}"
    return f"// {qualified}  ({kind})  {_path_label(relative_path, start_line)}{doc_note}\n{sig}"


def _context_doc_block(row: object) -> str:
    """Render one documentation symbol as a compact summary block."""
    relative_path = str(row["relative_path"])
    docstring = str(row["docstring"])
    display_name = str(row["display_name"])
    first_para = next(
        (line for line in docstring.splitlines() if line.strip()),
        display_name,
    )[:200]
    return f"// doc  {relative_path}\n// {first_para}"


def format_context_window(
    storage: "Storage",
    project_id: str,
    *,
    symbol_tokens: int = 8192,
    doc_tokens: int = 2048,
) -> str:
    """Return a compact context window packed to a Qwen-token budget.

    Two sections are assembled greedily:

    * **Symbols** (up to *symbol_tokens* Qwen tokens): top-ranked functions,
      methods, interfaces, structs, classes, traits, enums, type aliases,
      constants, and properties — one signature per line with a location
      comment header.  Symbols are ordered by ``global_rank`` descending so
      the most architecturally important symbols appear first.

    * **Docs** (up to *doc_tokens* Qwen tokens): top-ranked ``Document``
      symbols with a one-line summary.

    The function never simply truncates a symbol mid-block; it stops before
    adding a block that would exceed the remaining budget.  The total output
    therefore fits in approximately ``symbol_tokens + doc_tokens`` Qwen tokens.
    """
    symbol_rows = storage.get_symbol_rows(project_id)

    # --- Symbol section ---
    rankable = [
        row
        for row in symbol_rows.values()
        if str(row["kind"]) in _CONTEXT_SYMBOL_KINDS
        and float(row["global_rank"]) > 0
    ]
    rankable.sort(key=lambda r: (-float(r["global_rank"]), str(r["display_name"])))

    sym_lines: list[str] = ["// Symbols (by graph rank)"]
    sym_budget = symbol_tokens - count_tokens(sym_lines[0])
    for row in rankable:
        block = _context_symbol_block(row, symbol_rows)
        cost = count_tokens(block) + 1  # +1 for the blank separator
        if cost > sym_budget:
            break
        sym_lines.append(block)
        sym_lines.append("")
        sym_budget -= cost

    # --- Doc section ---
    doc_rows = [
        row
        for row in symbol_rows.values()
        if str(row["kind"]) == "Document"
        and float(row["global_rank"]) > 0
    ]
    doc_rows.sort(key=lambda r: (-float(r["global_rank"]), str(r["relative_path"])))

    doc_lines: list[str] = ["// Docs (by graph rank)"]
    doc_budget = doc_tokens - count_tokens(doc_lines[0])
    for row in doc_rows:
        block = _context_doc_block(row)
        cost = count_tokens(block) + 1
        if cost > doc_budget:
            break
        doc_lines.append(block)
        doc_lines.append("")
        doc_budget -= cost

    parts: list[str] = []
    if len(sym_lines) > 1:
        parts.append("\n".join(sym_lines).rstrip())
    if len(doc_lines) > 1:
        parts.append("\n".join(doc_lines).rstrip())
    return "\n\n".join(parts)
