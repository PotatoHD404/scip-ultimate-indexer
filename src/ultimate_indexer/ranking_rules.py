from __future__ import annotations

import re
from pathlib import PurePosixPath


NON_RANKABLE_KINDS = {
    "Artifact",
    "ArtifactConfig",
    "ArtifactSection",
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


def _symbol_leaf_name(symbol: str, relative_path: str) -> str:
    descriptor = symbol
    parts = symbol.split()
    if len(parts) >= 4:
        descriptor = parts[-1]
    descriptor = descriptor.strip()
    if not descriptor:
        return PurePosixPath(relative_path).stem

    descriptor = re.sub(r"'\+\d+$", "", descriptor.rstrip("."))
    if "#" in descriptor:
        _, member = descriptor.rsplit("#", 1)
        member = member.strip().rstrip(":")
        if member.endswith("()"):
            member = member[:-2]
        member_match = re.search(r"[A-Za-z_][\w$]*$", member)
        if member_match:
            return member_match.group(0)
        owner = descriptor.rsplit("#", 1)[0].split("/")[-1].strip("`")
        if owner:
            return owner

    leaf = descriptor.split("/")[-1].strip("`").rstrip(":")
    if leaf.endswith("()"):
        leaf = leaf[:-2]
    leaf_match = re.search(r"[A-Za-z_][\w$]*$", leaf)
    if leaf_match:
        return leaf_match.group(0)
    return leaf or PurePosixPath(relative_path).stem


def clean_symbol_display_name(symbol: str, display_name: str, docstring: str, relative_path: str) -> str:
    cleaned_display_name = display_name.strip()
    if cleaned_display_name and cleaned_display_name != symbol:
        return cleaned_display_name

    first_docstring_line = _first_docstring_line(docstring)
    for pattern in _DOCSTRING_NAME_PATTERNS:
        match = pattern.match(first_docstring_line)
        if match:
            return match.group(1)

    candidate = _symbol_leaf_name(symbol, relative_path)
    return candidate or PurePosixPath(relative_path).stem
