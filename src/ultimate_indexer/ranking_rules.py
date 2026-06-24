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
# Documentation kinds that should be rankable and queryable
DOC_RANKABLE_KINDS = {
    "Document",
}
DOC_QUERYABLE_KINDS = {
    "Document",
    "Section",  # Allow section queries for documentation
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


def is_external_symbol(relative_path: str) -> bool:
    """Return True for symbols that are library stubs (not from project source)."""
    return relative_path.startswith("_external/")



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


_TEST_DIR_NAMES = {"tests", "test", "__tests__", "spec", "specs", "testdata"}


def is_test_path(relative_path: str) -> bool:
    """Return True for paths that belong to test code.

    Test classes hold many member methods, which donate rank to their container
    in the compositional graph — without damping, ``top-symbols`` on a real
    repository is dominated by test suites instead of product code. Tests stay
    fully searchable; only their *importance* weight is reduced.
    """
    normalized = relative_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if any(part.lower() in _TEST_DIR_NAMES for part in path.parts[:-1]):
        return True
    stem = path.stem.lower()
    return (
        stem.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".test")
        or stem.endswith(".spec")
        or stem == "conftest"
    )


def is_rankable_symbol(relative_path: str, kind: str) -> bool:
    # External library stubs always participate in PageRank so that project
    # symbols which implement/call widely-used library types rank higher.
    if is_external_symbol(relative_path):
        return True
    if kind in NON_RANKABLE_KINDS:
        # Allow documentation kinds even if they're in the non-rankable set
        if kind in DOC_RANKABLE_KINDS:
            return not is_generated_path(relative_path)
        return False
    # Documentation sections are rankable
    if kind in DOC_RANKABLE_KINDS:
        return not is_generated_path(relative_path)
    return not is_generated_path(relative_path)


def is_queryable_symbol(relative_path: str, kind: str) -> bool:
    # External library stubs are queryable so they can surface in search when
    # they happen to rank highly (e.g. http.Handler implemented many times).
    if is_external_symbol(relative_path):
        return True
    if kind in NON_QUERYABLE_KINDS:
        # Allow documentation kinds even if they're in the non-queryable set
        if kind in DOC_QUERYABLE_KINDS:
            return not is_generated_path(relative_path)
        return False
    # Documentation sections are queryable
    if kind in DOC_QUERYABLE_KINDS:
        return not is_generated_path(relative_path)
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
