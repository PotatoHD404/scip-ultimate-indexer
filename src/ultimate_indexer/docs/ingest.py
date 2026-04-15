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
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

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


def _discover_document_files(
    project_root: Path,
    doc_dirs: list[str] | None = None,
) -> list[Path]:
    """Discover documentation files in the project.

    Args:
        project_root: Root directory of the project
        doc_dirs: Optional list of documentation directories to search.
                  If None, searches common doc directories and root.

    Returns:
        List of paths to documentation files
    """
    files: list[Path] = []

    # Files that should be handled by fallback, not documentation ingestion
    SKIP_FILES = {
        ".gitignore",
        ".dockerignore",
        ".socraticodeignore",
        ".cgcignore",
        ".env",
        ".env.example",
        ".env.local",
        ".env.production",
    }

    # Default documentation directories to search
    if doc_dirs is None:
        doc_dirs = [
            "docs",
            "documentation",
            "doc",
            "Docs",
            "Documentation",
            "api-docs",
            "api_docs",
            "openapi",
            "specs",
        ]

    # Collect all candidate directories
    candidate_dirs: list[Path] = []
    for dir_name in doc_dirs:
        candidate = project_root / dir_name
        if candidate.exists() and candidate.is_dir():
            candidate_dirs.append(candidate)

    # Remove duplicates while preserving order
    seen: set[tuple[int, int] | str] = set()
    unique_dirs: list[Path] = []
    for d in candidate_dirs:
        identity = _path_identity_key(d)
        if identity not in seen:
            seen.add(identity)
            unique_dirs.append(d)

    # Walk directories and find document files
    seen_files: set[tuple[int, int] | str] = set()
    for base_dir in unique_dirs:
        for ext in DOCUMENT_EXTENSIONS:
            for path in base_dir.rglob(f"*{ext}"):
                if not path.is_file():
                    continue
                # Skip hidden files (except markdown docs)
                if path.name.startswith(".") and not path.name.endswith(".md"):
                    continue
                # Skip special config files
                if path.name in SKIP_FILES:
                    continue
                # Skip common non-doc files
                if path.name in ("package.json", "tsconfig.json", ".eslintrc.json"):
                    continue
                identity = _path_identity_key(path)
                if identity in seen_files:
                    continue
                seen_files.add(identity)
                files.append(path)

    return sorted(files)


def _chunk_to_record(
    project_id: str,
    chunk: DocumentChunk,
    file_symbol_id: str,
    doc_symbol_id: str,
) -> ChunkRecord:
    """Convert a DocumentChunk to a ChunkRecord."""
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
        symbol_id=doc_symbol_id,
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
            kind="Section",
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

    # Second pass: register chunks and resolve links
    logger.info("Resolving documentation links...")
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
            chunk_record = _chunk_to_record(
                project_id, chunk, file_symbol_id, doc_symbol_id
            )
            chunks.append(chunk_record)

    # Third pass: resolve links and build graph edges
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

            # Note: Cross-file symbol edges are handled at the chunk level
            # through the document graph. We don't create additional symbol edges
            # here to avoid duplication.

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

    logger.info(
        f"Ingested {len(files)} documentation files, "
        f"{len(symbols)} symbols, "
        f"{len(edges)} edges, "
        f"{len(chunks)} chunks"
    )

    return files, symbols, edges, chunks
