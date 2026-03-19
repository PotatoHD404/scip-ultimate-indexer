from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

try:
    import tomllib
except Exception:  # pragma: no cover - Python 3.12+ ships tomllib
    tomllib = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass(slots=True)
class StructuredArtifactView:
    kind: str
    label: str
    summary: str
    rendered: str
    start_line: int
    end_line: int


def _first_text_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


def _summarize_value(value: Any) -> str:
    if isinstance(value, dict):
        preview = ", ".join(list(value.keys())[:6])
        suffix = ", ..." if len(value) > 6 else ""
        return f"object with {len(value)} keys: {preview}{suffix}".strip(": ")
    if isinstance(value, list):
        if not value:
            return "empty list"
        if all(not isinstance(item, (dict, list)) for item in value[:6]):
            preview = ", ".join(str(item) for item in value[:4])
            suffix = f", ... ({len(value)} total)" if len(value) > 4 else ""
            return f"list: {preview}{suffix}"
        return f"list of {len(value)} items"
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact[:117] + "..." if len(compact) > 120 else compact
    return str(value)


def _render_value(value: Any, indent: int = 0, max_depth: int = 4) -> str:
    pad = "  " * indent
    if indent > max_depth:
        return f"{pad}..."
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines: list[str] = []
        for key, child in value.items():
            rendered_child = _render_value(child, indent + 1, max_depth)
            if "\n" not in rendered_child.strip():
                lines.append(f"{pad}  {key}: {rendered_child.strip()}")
            else:
                lines.append(f"{pad}  {key}:")
                lines.append(rendered_child)
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value[:8]) and len(value) <= 8:
            return "[" + ", ".join(str(item) for item in value) + "]"
        lines = []
        for item in value[:12]:
            rendered_item = _render_value(item, indent + 1, max_depth)
            if "\n" not in rendered_item.strip():
                lines.append(f"{pad}  - {rendered_item.strip()}")
            else:
                lines.append(f"{pad}  -")
                lines.append(rendered_item)
        if len(value) > 12:
            lines.append(f"{pad}  ... ({len(value) - 12} more items)")
        return "\n".join(lines)
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        if any(char in value for char in ("\n", ":", "{", "[")):
            return json.dumps(value)
        return value
    return str(value)


def _flatten_config_entries(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    limit: int = 32,
) -> list[tuple[str, Any]]:
    if limit <= 0 or depth > 4:
        return []
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            qualified = f"{prefix}.{key}" if prefix else str(key)
            items.append((qualified, child))
            items.extend(
                _flatten_config_entries(child, prefix=qualified, depth=depth + 1, limit=limit - len(items))
            )
            if len(items) >= limit:
                return items[:limit]
        return items[:limit]
    if isinstance(value, list):
        items: list[tuple[str, Any]] = []
        for index, child in enumerate(value[:8]):
            qualified = f"{prefix}[{index}]"
            items.append((qualified, child))
            items.extend(
                _flatten_config_entries(child, prefix=qualified, depth=depth + 1, limit=limit - len(items))
            )
            if len(items) >= limit:
                return items[:limit]
        return items[:limit]
    return []


def _parse_structured_config(relative_path: str, content: str) -> Any | None:
    suffix = PurePosixPath(relative_path).suffix.lower()
    try:
        if suffix == ".json":
            return json.loads(content)
        if suffix == ".toml" and tomllib is not None:
            return tomllib.loads(content)
        if suffix in {".yaml", ".yml"} and yaml is not None:
            return yaml.safe_load(content)
    except Exception:
        return None
    return None


def _config_views(relative_path: str, content: str) -> list[StructuredArtifactView]:
    parsed = _parse_structured_config(relative_path, content)
    if parsed is None:
        return []

    views: list[StructuredArtifactView] = []
    if isinstance(parsed, dict):
        for key, value in list(parsed.items())[:20]:
            rendered_value = _render_value(value)
            rendered = f"{key}:\n{rendered_value}" if "\n" in rendered_value else f"{key}: {rendered_value}"
            views.append(
                StructuredArtifactView(
                    kind="ArtifactConfig",
                    label=str(key),
                    summary=_summarize_value(value),
                    rendered=rendered,
                    start_line=1,
                    end_line=max(1, len(content.splitlines())),
                )
            )
        for qualified_key, value in _flatten_config_entries(parsed):
            if isinstance(value, (dict, list)):
                continue
            parent_path = qualified_key.rsplit(".", 1)[0] if "." in qualified_key else "(root)"
            views.append(
                StructuredArtifactView(
                    kind="ArtifactConfig",
                    label=qualified_key,
                    summary=f"value from {parent_path}",
                    rendered=(
                        f"Qualified key: {qualified_key}\n"
                        f"Parent path: {parent_path}\n"
                        f"Value: {_render_value(value)}"
                    ),
                    start_line=1,
                    end_line=max(1, len(content.splitlines())),
                )
            )
            if len(views) >= 48:
                break
    else:
        views.append(
            StructuredArtifactView(
                kind="ArtifactConfig",
                label=PurePosixPath(relative_path).name,
                summary=_summarize_value(parsed),
                rendered=_render_value(parsed),
                start_line=1,
                end_line=max(1, len(content.splitlines())),
            )
        )
    return views[:48]


def _ini_views(relative_path: str, content: str) -> list[StructuredArtifactView]:
    section_name = "(global)"
    section_start = 1
    section_lines: list[str] = []
    views: list[StructuredArtifactView] = []

    def _flush(current_line: int) -> None:
        nonlocal section_lines, section_start, section_name
        if not section_lines:
            return
        body = "\n".join(section_lines).strip()
        if body:
            views.append(
                StructuredArtifactView(
                    kind="ArtifactConfig",
                    label=section_name,
                    summary=f"{len(section_lines)} config lines",
                    rendered=body,
                    start_line=section_start,
                    end_line=max(section_start, current_line - 1),
                )
            )
        section_lines = []

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        match = re.match(r"^\[(.+)\]$", stripped)
        if match:
            _flush(line_no)
            section_name = match.group(1).strip()
            section_start = line_no
            continue
        section_lines.append(raw_line.rstrip())
        kv_match = re.match(r"^([A-Za-z0-9_.-]+)\s*=\s*(.+)$", stripped)
        if kv_match:
            key = kv_match.group(1)
            value = kv_match.group(2)
            qualified = f"{section_name}.{key}" if section_name != "(global)" else key
            views.append(
                StructuredArtifactView(
                    kind="ArtifactConfig",
                    label=qualified,
                    summary=f"value in {section_name}",
                    rendered=(
                        f"Qualified key: {qualified}\n"
                        f"Section: {section_name}\n"
                        f"Value: {value}"
                    ),
                    start_line=line_no,
                    end_line=line_no,
                )
            )
    _flush(len(content.splitlines()) + 1)
    return views[:48]


def _markdown_views(content: str) -> list[StructuredArtifactView]:
    lines = content.splitlines()
    headings: list[tuple[int, int, str]] = []
    for index, raw_line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw_line)
        if match:
            headings.append((index, len(match.group(1)), match.group(2).strip()))
    if not headings:
        return []

    views: list[StructuredArtifactView] = []
    for idx, (line_no, level, heading) in enumerate(headings):
        end_line = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
        parent_heading = ""
        for parent_line, parent_level, parent_text in reversed(headings[:idx]):
            if parent_level < level:
                parent_heading = parent_text
                break
        subsections = [
            child_heading
            for _, child_level, child_heading in headings[idx + 1 :]
            if child_level == level + 1
        ][:6]
        body = "\n".join(lines[line_no:end_line]).strip()
        rendered_lines = [f"{'#' * level} {heading}"]
        if parent_heading:
            rendered_lines.append(f"<!-- parent: {parent_heading} -->")
        if subsections:
            rendered_lines.append(f"<!-- subsections: {', '.join(subsections)} -->")
        if body:
            rendered_lines.extend(["", body])
        rendered = "\n".join(rendered_lines)
        views.append(
            StructuredArtifactView(
                kind="ArtifactSection",
                label=heading,
                summary=_first_text_line(body) or f"Markdown section level {level}",
                rendered=rendered,
                start_line=line_no,
                end_line=max(line_no, end_line),
            )
        )
    return views[:48]


def extract_structured_artifact_views(relative_path: str, content: str) -> list[StructuredArtifactView]:
    suffix = PurePosixPath(relative_path).suffix.lower()
    if suffix == ".md":
        return _markdown_views(content)
    if suffix in {".json", ".toml", ".yaml", ".yml"}:
        return _config_views(relative_path, content)
    if suffix in {".ini", ".cfg", ".conf", ".env"}:
        return _ini_views(relative_path, content)
    return []
