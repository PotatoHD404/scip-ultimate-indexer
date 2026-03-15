from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .formatter import format_groups, format_top_symbols
from .indexer import UltimateIndexer


def build_mcp() -> FastMCP:
    server = FastMCP("scip-ultimate-indexer")

    @server.tool()
    def index_project(project_path: str, force: bool = False, embedding_backend: str = "auto") -> str:
        indexer = UltimateIndexer(Path(project_path), embedding_backend=embedding_backend)
        try:
            summary = indexer.index(force=force)
            return (
                f"Indexed {summary.indexed_files} files, {summary.indexed_symbols} symbols, "
                f"{summary.indexed_edges} edges, {summary.indexed_chunks} chunks"
            )
        finally:
            indexer.close()

    @server.tool()
    def query_project(
        project_path: str,
        query: str,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = UltimateIndexer(Path(project_path), embedding_backend=embedding_backend)
        try:
            groups = indexer.query(query, limit=limit)
            return format_groups(indexer.storage, indexer.project_id, groups)
        finally:
            indexer.close()

    @server.tool()
    def top_project_symbols(project_path: str, limit: int = 10, embedding_backend: str = "auto") -> str:
        indexer = UltimateIndexer(Path(project_path), embedding_backend=embedding_backend)
        try:
            return format_top_symbols(indexer.top_symbols(limit=limit))
        finally:
            indexer.close()

    @server.tool()
    def visualize_project(
        project_path: str,
        query: str,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = UltimateIndexer(Path(project_path), embedding_backend=embedding_backend)
        try:
            groups = indexer.query(query, limit=limit)
            path = indexer.visualize(groups, title=f"Results for: {query}")
            return str(path)
        finally:
            indexer.close()

    return server


def run_mcp() -> None:
    build_mcp().run(transport="stdio")
