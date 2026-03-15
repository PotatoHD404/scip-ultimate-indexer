from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .constants import MAX_AVG_LINE_LENGTH, MAX_CHUNK_CHARS, SUPPORTED_EXTENSIONS, get_language_from_extension
from .models import ChunkRecord, EdgeRecord, FileRecord, SymbolRecord


_SAFE_SPLIT_CHARS = {"\n", " ", "\t", ";", ",", "}"}
_JS_LIKE_LANGUAGES = {"javascript", "typescript", "vue", "svelte"}
_C_LIKE_LANGUAGES = {"c", "cpp"}
_IMPORT_RE = re.compile(r"^(?:import|export)\s+(?:.+?\s+from\s+)?['\"]([^'\"]+)['\"]", re.MULTILINE)
_REQUIRE_RE = re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_DYNAMIC_IMPORT_RE = re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
_PY_FROM_RE = re.compile(r"^\s*from\s+([.\w]+)\s+import\b", re.MULTILINE)
_PY_IMPORT_RE = re.compile(r"^\s*import\s+([.\w,\s]+)", re.MULTILINE)
_C_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
_RUBY_REQUIRE_RE = re.compile(r"""^\s*require(?:_relative)?\s*(?:\(\s*)?['"]([^'"]+)['"]""", re.MULTILINE)
_PHP_INCLUDE_RE = re.compile(r"""(?:require|include)(?:_once)?\s*(?:\(\s*)?['"]([^'"]+)['"]""")
_DART_IMPORT_RE = re.compile(r"""^(?:import|export|part)\s+['"]([^'"]+)['"]""", re.MULTILINE)
_LUA_REQUIRE_RE = re.compile(r"""require\s*[(]?\s*['"]([^'"]+)['"]\s*[)]?""")
_LUA_LOAD_RE = re.compile(r"""(?:dofile|loadfile)\s*\(\s*['"]([^'"]+)['"]\s*\)""")
_SHELL_SOURCE_RE = re.compile(r"""^\s*(?:source|\.)\s+([^\s#;]+)""", re.MULTILINE)
_SECTION_COMMENT_PREFIXES = ("#", "//", "--", ";", "/*", "*")


@dataclass(slots=True)
class ChunkSpan:
    discriminator: str
    start_line: int
    end_line: int
    content: str


@dataclass(slots=True)
class FallbackBundle:
    files: list[FileRecord]
    symbols: list[SymbolRecord]
    edges: list[EdgeRecord]
    chunks: list[ChunkRecord]


def _cap_chunk_content(content: str) -> str:
    if len(content) <= MAX_CHUNK_CHARS:
        return content
    return content[:MAX_CHUNK_CHARS]


def _chunk_by_characters(content: str) -> list[ChunkSpan]:
    spans: list[ChunkSpan] = []
    offset = 0
    current_line = 1
    while offset < len(content):
        end = min(offset + MAX_CHUNK_CHARS, len(content))
        if end < len(content):
            for index in range(end, offset, -1):
                if content[index - 1] in _SAFE_SPLIT_CHARS:
                    end = index
                    break
        chunk_content = content[offset:end]
        newline_count = chunk_content.count("\n")
        end_line = current_line + newline_count
        spans.append(
            ChunkSpan(
                discriminator=f"offset:{offset}",
                start_line=current_line,
                end_line=max(current_line, end_line),
                content=chunk_content,
            )
        )
        current_line = end_line + 1 if chunk_content.endswith("\n") else max(current_line, end_line)
        offset = end
    return spans


def chunk_file_content(
    content: str,
    max_chunk_lines: int,
    chunk_overlap: int,
) -> list[ChunkSpan]:
    if not content:
        return [ChunkSpan(discriminator="line:1", start_line=1, end_line=1, content="")]

    lines = content.splitlines()
    if not lines:
        return [ChunkSpan(discriminator="line:1", start_line=1, end_line=1, content=content)]

    avg_line_length = len(content) / max(len(lines), 1)
    if avg_line_length > MAX_AVG_LINE_LENGTH:
        return _chunk_by_characters(content)

    if len(lines) <= max_chunk_lines:
        return [
            ChunkSpan(
                discriminator="line:1",
                start_line=1,
                end_line=len(lines),
                content=_cap_chunk_content(content),
            )
        ]

    spans: list[ChunkSpan] = []
    stride = max(1, max_chunk_lines - max(0, chunk_overlap))
    start = 0
    while start < len(lines):
        end = min(start + max_chunk_lines, len(lines))
        spans.append(
            ChunkSpan(
                discriminator=f"line:{start + 1}",
                start_line=start + 1,
                end_line=end,
                content=_cap_chunk_content("\n".join(lines[start:end])),
            )
        )
        if end >= len(lines):
            break
        start += stride
    return spans


def _display_signature(chunk_content: str, relative_path: str, start_line: int, end_line: int) -> str:
    for raw_line in chunk_content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in _SECTION_COMMENT_PREFIXES):
            continue
        if len(line) > 140:
            return f"{line[:137]}..."
        return line
    return f"{relative_path}:{start_line}-{end_line}"


def _normalize_posix(path: str) -> str:
    normalized = posixpath.normpath(PurePosixPath(path).as_posix())
    return "" if normalized == "." else normalized


def _candidate_lookup_keys(relative_path: str) -> set[str]:
    path = PurePosixPath(_normalize_posix(relative_path))
    candidates = {path.as_posix()}
    stem = path.with_suffix("").as_posix()
    candidates.add(stem)
    if path.name.startswith("index.") or path.name == "__init__.py":
        candidates.add(_normalize_posix(path.parent.as_posix()))
    if path.suffix == ".py":
        if path.name == "__init__.py":
            candidates.add(_normalize_posix(path.parent.as_posix()).replace("/", "."))
        else:
            candidates.add(_normalize_posix(stem).replace("/", "."))
    return {item for item in candidates if item and item != "."}


def _extract_module_specifiers(content: str, language: str) -> list[str]:
    specs: list[str] = []
    if language in _JS_LIKE_LANGUAGES:
        specs.extend(_IMPORT_RE.findall(content))
        specs.extend(_REQUIRE_RE.findall(content))
        specs.extend(_DYNAMIC_IMPORT_RE.findall(content))
    elif language == "python":
        specs.extend(_PY_FROM_RE.findall(content))
        for group in _PY_IMPORT_RE.findall(content):
            specs.extend(part.strip() for part in group.split(",") if part.strip())
    elif language in _C_LIKE_LANGUAGES:
        specs.extend(_C_INCLUDE_RE.findall(content))
    elif language == "ruby":
        specs.extend(_RUBY_REQUIRE_RE.findall(content))
    elif language == "php":
        specs.extend(_PHP_INCLUDE_RE.findall(content))
    elif language == "dart":
        specs.extend(_DART_IMPORT_RE.findall(content))
    elif language == "lua":
        specs.extend(_LUA_REQUIRE_RE.findall(content))
        specs.extend(_LUA_LOAD_RE.findall(content))
    elif language == "shell":
        specs.extend(_SHELL_SOURCE_RE.findall(content))
    return [spec.strip() for spec in specs if spec.strip()]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _path_candidates(current_relative_path: str, specifier: str, language: str) -> list[str]:
    normalized = specifier.strip()
    if not normalized:
        return []
    if language == "dart" and normalized.startswith("package:"):
        return []
    if language == "lua":
        normalized = normalized.replace(".", "/")
    if language == "shell" and normalized.startswith("$"):
        return []

    base_dir = PurePosixPath(current_relative_path).parent
    candidates: list[str] = []
    if normalized.startswith("./") or normalized.startswith("../"):
        candidates.append(_normalize_posix(str(base_dir / normalized)))
    elif normalized.startswith("/"):
        candidates.append(_normalize_posix(normalized.lstrip("/")))
    elif "/" in normalized or normalized.endswith(tuple(SUPPORTED_EXTENSIONS)):
        candidates.append(_normalize_posix(normalized))
        candidates.append(_normalize_posix(str(base_dir / normalized)))
    elif language in _C_LIKE_LANGUAGES:
        candidates.append(_normalize_posix(str(base_dir / normalized)))
    return _dedupe(item for item in candidates if item and item != ".")


def _python_candidates(current_relative_path: str, specifier: str) -> list[str]:
    if not specifier:
        return []
    leading_dots = len(specifier) - len(specifier.lstrip("."))
    module_name = specifier.lstrip(".")
    if leading_dots == 0:
        return [module_name]

    base_dir = PurePosixPath(current_relative_path).parent
    for _ in range(max(leading_dots - 1, 0)):
        base_dir = base_dir.parent
    suffix = PurePosixPath(module_name.replace(".", "/")) if module_name else PurePosixPath(".")
    target = _normalize_posix((base_dir / suffix).as_posix())
    return _dedupe(item for item in [target, target.replace("/", ".")] if item and item != ".")


def _resolve_internal_targets(
    current_relative_path: str,
    language: str,
    specifier: str,
    lookup: dict[str, list[str]],
) -> list[str]:
    candidates: list[str] = []
    if language == "python":
        candidates.extend(_python_candidates(current_relative_path, specifier))
    candidates.extend(_path_candidates(current_relative_path, specifier, language))

    matches: list[str] = []
    for candidate in candidates:
        lookup_key = candidate.rstrip("/")
        matches.extend(lookup.get(lookup_key, []))
        stem_key = PurePosixPath(lookup_key).with_suffix("").as_posix()
        if stem_key != lookup_key:
            matches.extend(lookup.get(stem_key, []))
    return [item for item in _dedupe(matches) if item != current_relative_path]


def build_fallback_bundle(
    project_id: str,
    project_root: Path,
    files: Iterable[Path],
    covered_paths: set[str],
    contains_weight: float,
    import_weight: float,
    max_chunk_lines: int,
    chunk_overlap: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> FallbackBundle:
    file_records: list[FileRecord] = []
    symbols: list[SymbolRecord] = []
    edges: list[EdgeRecord] = []
    chunks: list[ChunkRecord] = []
    module_symbol_ids: dict[str, str] = {}
    file_payloads: dict[str, tuple[str, str]] = {}

    pending_files = [
        path for path in files
        if path.relative_to(project_root).as_posix() not in covered_paths
    ]

    for index, path in enumerate(pending_files, start=1):
        relative_path = path.relative_to(project_root).as_posix()
        content = path.read_text(encoding="utf-8", errors="ignore")
        language = get_language_from_extension(path.suffix.lower())
        file_symbol_id = f"file::{relative_path}"
        module_symbol_id = f"module::{relative_path}"
        content_hash = sha256(content.encode("utf-8")).hexdigest()
        line_count = max(1, len(content.splitlines()))

        file_records.append(
            FileRecord(
                project_id=project_id,
                relative_path=relative_path,
                abs_path=str(path.resolve()),
                language=language,
                content_hash=content_hash,
                content=content,
            )
        )
        symbols.append(
            SymbolRecord(
                project_id=project_id,
                symbol_id=file_symbol_id,
                scip_symbol=file_symbol_id,
                display_name=path.name,
                kind="File",
                relative_path=relative_path,
                start_line=1,
                end_line=line_count,
                signature=relative_path,
                docstring="",
                snippet="\n".join(content.splitlines()[: min(40, line_count)]),
            )
        )
        symbols.append(
            SymbolRecord(
                project_id=project_id,
                symbol_id=module_symbol_id,
                scip_symbol=module_symbol_id,
                display_name=path.stem or path.name,
                kind="Module",
                relative_path=relative_path,
                start_line=1,
                end_line=line_count,
                signature=relative_path,
                docstring="",
                snippet="\n".join(content.splitlines()[: min(80, line_count)]),
                enclosing_symbol_id=file_symbol_id,
            )
        )
        edges.append(
            EdgeRecord(
                project_id=project_id,
                source_symbol_id=file_symbol_id,
                target_symbol_id=module_symbol_id,
                edge_type="contains",
                weight=contains_weight,
            )
        )
        module_symbol_ids[relative_path] = module_symbol_id
        file_payloads[relative_path] = (language, content)

        for span in chunk_file_content(content, max_chunk_lines=max_chunk_lines, chunk_overlap=chunk_overlap):
            section_symbol_id = f"section::{relative_path}:{span.discriminator}"
            signature = _display_signature(span.content, relative_path, span.start_line, span.end_line)
            display_name = f"{Path(relative_path).stem}:{span.start_line}"
            symbols.append(
                SymbolRecord(
                    project_id=project_id,
                    symbol_id=section_symbol_id,
                    scip_symbol=section_symbol_id,
                    display_name=display_name,
                    kind="Section",
                    relative_path=relative_path,
                    start_line=span.start_line,
                    end_line=span.end_line,
                    signature=signature,
                    docstring="",
                    snippet=span.content,
                    enclosing_symbol_id=module_symbol_id,
                    source_kind="fallback",
                )
            )
            edges.append(
                EdgeRecord(
                    project_id=project_id,
                    source_symbol_id=module_symbol_id,
                    target_symbol_id=section_symbol_id,
                    edge_type="contains",
                    weight=contains_weight,
                )
            )
            chunk_content = "\n".join(item for item in [relative_path, language, span.content] if item)
            chunks.append(
                ChunkRecord(
                    project_id=project_id,
                    chunk_id=sha256(
                        f"{relative_path}:{span.discriminator}:{span.start_line}:{span.end_line}".encode("utf-8")
                    ).hexdigest()[:32],
                    relative_path=relative_path,
                    symbol_id=section_symbol_id,
                    symbol_name=display_name,
                    artifact_name=None,
                    chunk_kind="fallback-section",
                    start_line=span.start_line,
                    end_line=span.end_line,
                    content=chunk_content,
                    content_hash=sha256(chunk_content.encode("utf-8")).hexdigest(),
                )
            )
        if progress_callback is not None:
            progress_callback(index, len(pending_files), relative_path)

    lookup: dict[str, list[str]] = defaultdict(list)
    for relative_path in module_symbol_ids:
        for candidate in _candidate_lookup_keys(relative_path):
            lookup[candidate].append(relative_path)

    seen_edges = {(edge.source_symbol_id, edge.target_symbol_id, edge.edge_type) for edge in edges}
    for relative_path, module_symbol_id in module_symbol_ids.items():
        language, content = file_payloads[relative_path]
        for specifier in _extract_module_specifiers(content, language):
            for target_path in _resolve_internal_targets(relative_path, language, specifier, lookup):
                target_symbol_id = module_symbol_ids.get(target_path)
                if target_symbol_id is None:
                    continue
                edge_key = (module_symbol_id, target_symbol_id, "imports")
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append(
                    EdgeRecord(
                        project_id=project_id,
                        source_symbol_id=module_symbol_id,
                        target_symbol_id=target_symbol_id,
                        edge_type="imports",
                        weight=import_weight,
                    )
                )

    return FallbackBundle(files=file_records, symbols=symbols, edges=edges, chunks=chunks)
