"""Documentation ingestion pipeline for Markdown and OpenAPI files.

This module provides the main entry point for ingesting documentation files
into the ultimate-indexer's existing graph infrastructure. It:

1. Discovers Markdown and OpenAPI files in a directory
2. Parses files using MarkdownParser and OpenAPIParser
3. Chunks content with context preservation
4. Resolves cross-file and intra-file links
5. Builds a document graph with weighted edges
6. Produces FileRecord, SymbolRecord, EdgeRecord, and ChunkRecord objects
   compatible with the existing Storage schema

The module follows patterns from socraticode.py and fallback.py for integration
with the main indexer pipeline.
"""

from __future__ import annotations

import logging
import os
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from ..constants import DEFAULT_IGNORED_DIR_NAMES
from ..embeddings import hash_text
from ..models import ChunkRecord, EdgeRecord, FileRecord, SymbolRecord
from .chunker import DocumentChunk, DocumentChunker
from .doc_graph import DocumentGraph, DEFAULT_LINK_WEIGHTS
from .link_resolver import LinkResolver
from .markdown_parser import MarkdownParser, SectionHeader
from .openapi_parser import OpenAPIParser, OpenAPISection

logger = logging.getLogger(__name__)

# Supported file extensions
DOCUMENT_EXTENSIONS = {".md", ".markdown", ".yaml", ".yml", ".json"}

# Chunk kind constants
CHUNK_KINDS = {
    "markdown_section": "doc-markdown-section",
    "markdown_text": "doc-markdown-text",
    "openapi_endpoint": "doc-openapi-endpoint",
    "openapi_schema": "doc-openapi-schema",
    "openapi_description": "doc-openapi-description",
}


def _path_identity_key(path: Path) -> tuple[int, int] | str:
    try:
        stat = path.stat()
    except OSError:
        return str(path.resolve()).casefold()
    return (stat.st_dev, stat.st_ino)


# Files that should be handled by fallback, not documentation ingestion.
_SKIP_DOC_FILES = {
    ".gitignore",
    ".dockerignore",
    ".socraticodeignore",
    ".cgcignore",
    ".env",
    ".env.example",
    ".env.local",
    ".env.production",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    ".eslintrc.json",
    "composer.json",
    "composer.lock",
}

# Top-level OpenAPI/Swagger discriminator (JSON `"openapi"` or YAML `openapi:`).
_OPENAPI_MARKER = re.compile(r'(?m)^\s*["\']?(openapi|swagger)["\']?\s*:')


def _looks_like_openapi(path: Path) -> bool:
    """Content sniff so only real OpenAPI/Swagger specs — not arbitrary YAML/JSON
    config (CI files, package manifests, …) — are treated as documentation.
    Reads only the head; the full validation still happens in OpenAPIParser."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(8192)
    except OSError:
        return False
    return bool(_OPENAPI_MARKER.search(head))


def _discover_document_files(
    project_root: Path,
    doc_dirs: list[str] | None = None,
) -> list[Path]:
    """Discover documentation files anywhere in the project, honoring ignore dirs.

    Markdown (``.md``/``.markdown``) is unambiguous documentation, so every such
    file outside ignored directories is included regardless of location.
    ``.yaml``/``.yml``/``.json`` files are included only when they content-sniff
    as an OpenAPI/Swagger spec, so config files stay on the code/fallback path.

    A non-empty ``doc_dirs`` restricts the walk to those subdirectories
    (preserved for callers that scope ingestion explicitly); the default (None)
    searches the whole tree so specs/READMEs are found wherever they live.
    """
    files: list[Path] = []
    seen_files: set[tuple[int, int] | str] = set()

    def _add(path: Path) -> None:
        if not path.is_file():
            return
        name = path.name
        if name in _SKIP_DOC_FILES:
            return
        ext = path.suffix.lower()
        if ext in (".md", ".markdown"):
            pass  # markdown is always documentation
        elif ext in (".yaml", ".yml", ".json"):
            if not _looks_like_openapi(path):
                return
        else:
            return
        # Skip hidden files except markdown docs (e.g. keep a .changes.md note).
        if name.startswith(".") and ext not in (".md", ".markdown"):
            return
        identity = _path_identity_key(path)
        if identity in seen_files:
            return
        seen_files.add(identity)
        files.append(path)

    if doc_dirs:
        roots = [project_root / d for d in doc_dirs if (project_root / d).is_dir()]
    else:
        roots = [project_root]

    for base in roots:
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune ignored + hidden directories in place so we never descend
            # into node_modules, .git, build output, the index state dir, etc.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in DEFAULT_IGNORED_DIR_NAMES and not d.startswith(".")
            ]
            for fn in filenames:
                _add(Path(dirpath) / fn)

    return sorted(files)


def _chunk_to_record(
    project_id: str,
    chunk: DocumentChunk,
    file_symbol_id: str,
    chunk_symbol_id: str,
) -> ChunkRecord:
    """Convert a DocumentChunk to a ChunkRecord.

    ``chunk_symbol_id`` is the symbol the chunk binds to in search results — a
    ``doc-section::`` symbol when the chunk belongs to a section, else the
    ``doc::`` document root.
    """
    # Build rich content with breadcrumb context
    content_parts: list[str] = []
    if chunk.breadcrumb:
        content_parts.append(f"[{chunk.breadcrumb}]")
    content_parts.append(f"file: {chunk.file_path}")
    content_parts.append(f"type: {chunk.chunk_type}")

    # Add header context
    if chunk.headers:
        header_context = " > ".join(h.text for h in chunk.headers)
        content_parts.append(f"path: {header_context}")

    content_parts.append("")
    content_parts.append(chunk.content)

    full_content = "\n".join(content_parts)

    return ChunkRecord(
        project_id=project_id,
        chunk_id=chunk.chunk_id
        or sha256(
            f"{chunk.file_path}:{chunk.chunk_type}:{chunk.anchor or chunk.breadcrumb or f'line:{chunk.start_line}-{chunk.end_line}'}:{chunk.metadata.get('sub_chunk', 0) if chunk.metadata else 0}".encode()
        ).hexdigest()[:32],
        relative_path=chunk.file_path,
        symbol_id=chunk_symbol_id,
        symbol_name=Path(chunk.file_path).stem,
        artifact_name=None,
        chunk_kind=CHUNK_KINDS.get(chunk.chunk_type, "doc-section"),
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        content=full_content,
        content_hash=hash_text(full_content),
    )


def _create_doc_symbols(
    project_id: str,
    file_path: str,
    file_record: FileRecord,
    sections: list[Any],
) -> tuple[list[SymbolRecord], list[EdgeRecord]]:
    """Create SymbolRecord and EdgeRecord objects for a document.

    Creates:
    - Document symbol (using doc:: prefix to avoid conflicts with code)
    - Document section symbols for each major section

    Note: We don't create a file:: symbol here since it may already exist
    from the FileRecord. We use doc:: prefix for all documentation symbols
    to avoid conflicts with code symbols.
    """
    symbols: list[SymbolRecord] = []
    edges: list[EdgeRecord] = []

    # Use doc:: prefix for documentation to avoid conflicts with code file symbols
    doc_symbol_id = f"doc::{file_path}"

    # Create a document-level symbol
    doc_symbol = SymbolRecord(
        project_id=project_id,
        symbol_id=doc_symbol_id,
        scip_symbol=doc_symbol_id,
        display_name=Path(file_path).stem,
        kind="Document",
        relative_path=file_path,
        start_line=1,
        end_line=file_record.content.count("\n") + 1,
        signature=file_path,
        docstring=f"Documentation: {file_path}",
        snippet=file_record.content[:500]
        if len(file_record.content) > 500
        else file_record.content,
        enclosing_symbol_id=None,  # No parent for doc root
        source_kind="documentation",
    )
    symbols.append(doc_symbol)

    # Create section symbols
    for idx, section in enumerate(sections):
        header = section.get("header")
        anchor = section.get("anchor")
        if header is None and anchor is None:
            continue

        section_anchor = anchor or (header.anchor if header else f"section-{idx}")
        section_symbol_id = f"doc-section::{file_path}:{section_anchor}"

        section_symbol = SymbolRecord(
            project_id=project_id,
            symbol_id=section_symbol_id,
            scip_symbol=section_symbol_id,
            display_name=header.text if header else (anchor or f"Section {idx}"),
            # OpenAPI operations and schemas are the meaningful, rankable units of
            # an API spec (like a struct/endpoint in code), so they get first-class
            # kinds; prose sections stay non-rankable "Section".
            kind={
                "openapi_endpoint": "ApiEndpoint",
                "openapi_schema": "ApiSchema",
            }.get(section.get("chunk_type", ""), "Section"),
            relative_path=file_path,
            start_line=(header.line_number + 1)
            if header
            else section.get("start_line", 0),
            end_line=section.get("end_line", section.get("start_line", 0) + 10),
            signature=header.text if header else (anchor or f"Section {idx}"),
            docstring=section.get("content", "")[:200]
            if section.get("content")
            else "",
            snippet=section.get("content", "")[:300] if section.get("content") else "",
            enclosing_symbol_id=doc_symbol_id,
            source_kind="documentation",
        )
        symbols.append(section_symbol)

        # Edge from document to section
        edges.append(
            EdgeRecord(
                project_id=project_id,
                source_symbol_id=doc_symbol_id,
                target_symbol_id=section_symbol_id,
                edge_type="contains",
                weight=0.55,
            )
        )

    return symbols, edges


def ingest_documentation(
    project_id: str,
    project_root: Path,
    doc_dirs: list[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[FileRecord], list[SymbolRecord], list[EdgeRecord], list[ChunkRecord]]:
    """Ingest all documentation files from a project.

    This is the main entry point for documentation ingestion. It discovers,
    parses, chunks, and links documentation files, producing records compatible
    with the existing Storage schema.

    Args:
        project_id: The project identifier
        project_root: Root directory of the project
        doc_dirs: Optional list of documentation directories to search
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (files, symbols, edges, chunks)
    """
    files: list[FileRecord] = []
    symbols: list[SymbolRecord] = []
    edges: list[EdgeRecord] = []
    chunks: list[ChunkRecord] = []

    # Track processed files to avoid duplicates
    processed_files: set[str] = set()

    # Discover documentation files
    doc_files = _discover_document_files(project_root, doc_dirs)
    if not doc_files:
        logger.info("No documentation files found")
        return files, symbols, edges, chunks

    # Deduplicate files by relative path
    unique_files: list[Path] = []
    for f in doc_files:
        rel = str(f.relative_to(project_root))
        if rel not in processed_files:
            processed_files.add(rel)
            unique_files.append(f)
    doc_files = unique_files

    logger.info(f"Found {len(doc_files)} documentation files")

    # Initialize chunker and link resolver
    chunker = DocumentChunker()
    link_resolver = LinkResolver(str(project_root))
    doc_graph = DocumentGraph()

    # First pass: parse and chunk all files
    file_data: list[tuple[str, list[DocumentChunk], list[dict], dict[str, int]]] = []

    for index, doc_path in enumerate(doc_files, start=1):
        rel_path = str(doc_path.relative_to(project_root))
        try:
            content = doc_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Failed to read {rel_path}: {e}")
            continue

        # Create file record
        language = doc_path.suffix.lower().lstrip(".") or "text"
        if language in ("md", "markdown"):
            language = "markdown"
        elif language in ("yaml", "yml"):
            language = "openapi"

        file_record = FileRecord(
            project_id=project_id,
            relative_path=rel_path,
            abs_path=str(doc_path.resolve()),
            language=language,
            content_hash=hash_text(content),
            content=content,
            source_kind="documentation",
        )
        files.append(file_record)

        # Note: We don't create a file:: symbol here since documentation files
        # are already represented by FileRecord. We only create doc:: symbols
        # to avoid conflicts with code file symbols that may already exist.

        # Parse based on file type
        sections_parsed: list[dict[str, Any]] = []
        links: list[dict] = []
        anchors: dict[str, int] = {}

        if doc_path.suffix.lower() in (".md", ".markdown"):
            # Markdown parsing
            parser = MarkdownParser(rel_path, content)
            headers, parsed_links, anchors = parser.parse()
            sections = parser.get_sections()
            sections_parsed = [
                {
                    "header": s.header,
                    "level": s.level,
                    "content": s.content,
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                    "anchor": s.anchor,
                    "anchors_in_range": s.anchors_in_range,
                    "chunk_type": "markdown_section",
                }
                for s in sections
            ]
            links = parsed_links  # Keep as ParsedLink objects
        elif doc_path.suffix.lower() in (".yaml", ".yml", ".json"):
            # OpenAPI parsing
            parser = OpenAPIParser(rel_path, content)
            openapi_sections, parsed_links, anchors = parser.parse()
            if openapi_sections:  # Valid OpenAPI spec
                sections_parsed = [
                    {
                        "header": s.header,
                        "level": s.level,
                        "content": s.content,
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                        "anchor": s.anchor,
                        "anchors_in_range": s.anchors_in_range,
                        "chunk_type": s.chunk_type,
                        "metadata": s.metadata,
                    }
                    for s in openapi_sections
                ]
                links = parsed_links  # Keep as ParsedLink objects
            else:
                # Not a valid OpenAPI spec, treat as generic YAML/JSON
                sections_parsed = [
                    {
                        "header": None,
                        "level": 0,
                        "content": content,
                        "start_line": 0,
                        "end_line": content.count("\n") + 1,
                        "anchor": None,
                        "anchors_in_range": [],
                        "chunk_type": "markdown_text",
                    }
                ]

        # Create document and section symbols
        doc_symbols, doc_edges = _create_doc_symbols(
            project_id, rel_path, file_record, sections_parsed
        )
        symbols.extend(doc_symbols)
        edges.extend(doc_edges)

        # Chunk the sections
        doc_chunks = chunker.chunk_sections(rel_path, sections_parsed)

        # Generate chunk IDs using anchor (section name) when available
        for chunk in doc_chunks:
            if not chunk.chunk_id:
                section_key = chunk.anchor or chunk.breadcrumb or f"line:{chunk.start_line}-{chunk.end_line}"
                sub_chunk = chunk.metadata.get("sub_chunk", 0) if chunk.metadata else 0
                chunk.chunk_id = sha256(
                    f"{chunk.file_path}:{chunk.chunk_type}:{section_key}:{sub_chunk}".encode()
                ).hexdigest()[:32]

        file_data.append((rel_path, doc_chunks, links, anchors))

        if progress_callback:
            progress_callback(index, len(doc_files), rel_path)

    # Second pass: register chunks and resolve links. Chunks bind to their
    # SECTION symbol when one exists so section-level results can surface in
    # search output; the doc-root symbol is only the fallback.
    logger.info("Resolving documentation links...")
    created_symbol_ids = {symbol.symbol_id for symbol in symbols}
    chunk_symbol_map: dict[str, str] = {}
    for rel_path, doc_chunks, links, anchors in file_data:
        # Register with link resolver
        link_resolver.register_file(rel_path, doc_chunks, anchors)

        # Add chunks to graph
        for chunk in doc_chunks:
            doc_graph.add_node(
                chunk.chunk_id,
                file_path=chunk.file_path,
                chunk_type=chunk.chunk_type,
                breadcrumb=chunk.breadcrumb,
                anchor=chunk.anchor,
            )

            # Create chunk record
            file_symbol_id = f"file::{rel_path}"
            doc_symbol_id = f"doc::{rel_path}"
            section_anchor = chunk.anchor or (
                chunk.headers[-1].anchor if chunk.headers else None
            )
            chunk_symbol_id = doc_symbol_id
            if section_anchor:
                candidate = f"doc-section::{rel_path}:{section_anchor}"
                if candidate in created_symbol_ids:
                    chunk_symbol_id = candidate
            chunk_record = _chunk_to_record(
                project_id, chunk, file_symbol_id, chunk_symbol_id
            )
            chunk_symbol_map[chunk_record.chunk_id] = chunk_symbol_id
            chunks.append(chunk_record)

    # Third pass: resolve links and build graph edges. Link and hierarchy
    # structure is persisted as symbol-level EdgeRecords (translated through the
    # chunk->symbol map) so it feeds PageRank; the in-memory doc_graph mirrors it
    # for chunk-level inspection.
    seen_doc_edges: set[tuple[str, str, str]] = set()

    def _persist_doc_edge(edge: dict) -> None:
        source_symbol = chunk_symbol_map.get(edge["source_chunk_id"])
        target_symbol = chunk_symbol_map.get(edge["target_chunk_id"])
        if not source_symbol or not target_symbol or source_symbol == target_symbol:
            return
        key = (source_symbol, target_symbol, str(edge["edge_type"]))
        if key in seen_doc_edges:
            return
        seen_doc_edges.add(key)
        edges.append(
            EdgeRecord(
                project_id=project_id,
                source_symbol_id=source_symbol,
                target_symbol_id=target_symbol,
                edge_type=str(edge["edge_type"]),
                weight=float(edge["weight"]),
            )
        )

    for rel_path, doc_chunks, links, anchors in file_data:
        # Resolve explicit links
        resolved_edges = link_resolver.resolve_links(
            links, rel_path, DEFAULT_LINK_WEIGHTS
        )
        for edge in resolved_edges:
            doc_graph.add_edge(
                edge["source_chunk_id"],
                edge["target_chunk_id"],
                edge["edge_type"],
                edge["weight"],
            )
            _persist_doc_edge(edge)

        # Build hierarchy edges
        hierarchy_edges = link_resolver.build_hierarchy_edges(
            rel_path, doc_chunks, DEFAULT_LINK_WEIGHTS.get("hierarchy", 0.3)
        )
        for edge in hierarchy_edges:
            doc_graph.add_edge(
                edge["source_chunk_id"],
                edge["target_chunk_id"],
                edge["edge_type"],
                edge["weight"],
            )
            _persist_doc_edge(edge)

    logger.info(
        f"Ingested {len(files)} documentation files, "
        f"{len(symbols)} symbols, "
        f"{len(edges)} edges, "
        f"{len(chunks)} chunks"
    )

    return files, symbols, edges, chunks
