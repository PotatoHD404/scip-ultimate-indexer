from __future__ import annotations

import atexit
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .formatter import format_groups_compact, format_top_symbols, truncate_text
from .indexer import UltimateIndexer


def build_mcp() -> FastMCP:
    server = FastMCP("scip-ultimate-indexer", log_level="WARNING")
    indexers: dict[tuple[str, str], UltimateIndexer] = {}

    def get_indexer(project_path: str, embedding_backend: str) -> UltimateIndexer:
        resolved_path = str(Path(project_path).resolve())
        cache_key = (resolved_path, embedding_backend)
        indexer = indexers.get(cache_key)
        if indexer is None:
            indexer = UltimateIndexer(Path(resolved_path), embedding_backend=embedding_backend)
            indexers[cache_key] = indexer
        return indexer

    def close_indexers() -> None:
        while indexers:
            _, indexer = indexers.popitem()
            try:
                indexer.close()
            except Exception:
                pass

    atexit.register(close_indexers)

    @server.tool()
    def index_project(project_path: str, force: bool = False, embedding_backend: str = "auto") -> str:
        indexer = get_indexer(project_path, embedding_backend)
        summary = indexer.index(force=force)
        return (
            f"Indexed {summary.indexed_files} files, {summary.indexed_symbols} symbols, "
            f"{summary.indexed_edges} edges, {summary.indexed_chunks} chunks"
        )

    @server.tool()
    def query_project(
        project_path: str,
        query: str,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project_path, embedding_backend)
        groups = indexer.query(query, limit=limit)
        return format_groups_compact(indexer.storage, indexer.project_id, groups)

    @server.tool()
    def top_project_symbols(project_path: str, limit: int = 10, embedding_backend: str = "auto") -> str:
        indexer = get_indexer(project_path, embedding_backend)
        return truncate_text(format_top_symbols(indexer.top_symbols(limit=limit), include_scores=False), 4_000)

    @server.tool()
    def visualize_project(
        project_path: str,
        query: str,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project_path, embedding_backend)
        groups = indexer.query(query, limit=limit)
        path = indexer.visualize(groups, title=f"Results for: {query}")
        return str(path)

    return server


def run_mcp() -> None:
    build_mcp().run(transport="stdio")
