from __future__ import annotations

import re
from pathlib import PurePosixPath


NON_RANKABLE_KINDS = {
    "Artifact",
    "Field",
    "File",
    "Kind",
    "Module",
    "Parameter",
    "Section",
    "Unknown",
    "Variable",
}
NON_QUERYABLE_KINDS = {
    "Field",
    "File",
    "Kind",
    "Module",
    "Parameter",
    "Unknown",
    "Variable",
}
GENERATED_SUFFIXES = (
    ".pb.go",
    "_pb2.py",
    "_pb2.pyi",
)
PROTO_GENERATED_SUFFIXES = {
    ".go",
    ".js",
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
}
_DOCSTRING_NAME_PATTERNS = (
    re.compile(r"^\((?:property|parameter)\)\s+([A-Za-z_][\w$]*)"),
    re.compile(r"^(?:var|let|const|type|enum|interface|class|struct)\s+([A-Za-z_][\w$]*)"),
    re.compile(r"^(?:function|method)\s+([A-Za-z_][\w$]*)"),
)


def is_generated_path(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    parts = {part.lower() for part in path.parts}
    filename = path.name.lower()
    suffix = path.suffix.lower()

    if ".next" in parts:
        return True
    if filename.endswith(GENERATED_SUFFIXES):
        return True
    if "proto" in parts and suffix in PROTO_GENERATED_SUFFIXES:
        return True
    return False


def is_rankable_symbol(relative_path: str, kind: str) -> bool:
    if kind in NON_RANKABLE_KINDS:
        return False
    return not is_generated_path(relative_path)


def is_queryable_symbol(relative_path: str, kind: str) -> bool:
    if kind in NON_QUERYABLE_KINDS:
        return False
    return not is_generated_path(relative_path)


def _first_docstring_line(docstring: str) -> str:
    for raw_line in docstring.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        return line
    return ""


def clean_symbol_display_name(symbol: str, display_name: str, docstring: str, relative_path: str) -> str:
    cleaned_display_name = display_name.strip()
    if cleaned_display_name and cleaned_display_name != symbol:
        return cleaned_display_name

    first_docstring_line = _first_docstring_line(docstring)
    for pattern in _DOCSTRING_NAME_PATTERNS:
        match = pattern.match(first_docstring_line)
        if match:
            return match.group(1)

    candidate = symbol.rsplit("/", 1)[-1].strip() or PurePosixPath(relative_path).stem
    candidate = re.sub(r"[#.:]$", "", candidate)
    candidate = re.sub(r"'\+\d+$", "", candidate)
    if "#" in candidate and not re.search(r"[A-Za-z_][\w$]*$", candidate.split("#", 1)[-1]):
        candidate = candidate.split("#", 1)[0]
    return candidate or PurePosixPath(relative_path).stem
