"""Documentation ingestion module for Markdown and OpenAPI specifications.

This module provides parsers and processors for documentation files that integrate
with the ultimate-indexer's existing graph infrastructure. It supports:

- Markdown (.md, .markdown) files with cross-file and intra-file link resolution
- OpenAPI 3.x specifications (.yaml, .yml, .json) with $ref resolution
- Smart chunking with context preservation (breadcrumbs, header hierarchy)
- Document graph construction with weighted edges for different link types
- Personalized PageRank for query-time relevance boosting

The module follows the same patterns as scip_parser.py and socraticode.py,
producing FileRecord, SymbolRecord, EdgeRecord, and ChunkRecord objects
that are stored in the existing SQLite schema.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .markdown_parser import MarkdownParser, slugify_heading
from .openapi_parser import OpenAPIParser
from .chunker import DocumentChunker
from .link_resolver import LinkResolver
from .doc_graph import DocumentGraph
from .ingest import ingest_documentation

__all__ = [
    "MarkdownParser",
    "OpenAPIParser",
    "slugify_heading",
    "DocumentChunker",
    "LinkResolver",
    "DocumentGraph",
    "ingest_documentation",
]
