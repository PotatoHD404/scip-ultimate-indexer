from __future__ import annotations

import atexit
import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .formatter import format_groups_compact, format_top_symbols, truncate_text
from .indexer import UltimateIndexer


def _normalized_kind(kind: str) -> str:
    return kind.replace("_", "").replace("-", "").lower()


def _registry_path(cache_dir: Path) -> Path:
    return cache_dir / "projects.json"


def _load_registry(cache_dir: Path) -> list[dict[str, str]]:
    path = _registry_path(cache_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        project_path = str(item.get("path", "")).strip()
        if not project_path:
            continue
        name = str(item.get("name", "")).strip() or Path(project_path).name
        entries.append({"name": name, "path": project_path})
    return entries


def _save_registry(cache_dir: Path, entries: list[dict[str, str]]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _registry_path(cache_dir).write_text(
        json.dumps(entries, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _register_project(cache_dir: Path, project_path: Path) -> None:
    resolved = str(project_path.resolve())
    name = project_path.resolve().name
    entries = _load_registry(cache_dir)
    by_path = {entry["path"]: entry for entry in entries}
    by_path[resolved] = {"name": name, "path": resolved}
    merged = sorted(by_path.values(), key=lambda item: (item["name"].lower(), item["path"]))
    _save_registry(cache_dir, merged)


def _resolve_project_path(cache_dir: Path, project: str | None) -> Path:
    if project:
        candidate = Path(project).expanduser()
        if candidate.exists():
            return candidate.resolve()
    entries = _load_registry(cache_dir)
    if project:
        for entry in entries:
            if entry["name"] == project or entry["path"] == project:
                return Path(entry["path"]).resolve()
    if len(entries) == 1:
        return Path(entries[0]["path"]).resolve()
    raise ValueError("Unknown project. Use list_projects or pass an absolute project path.")


def build_mcp(
    *,
    cache_dir: Path | str = ".scip_indexes",
    embedding_model: str = "models/coderankembed-q8_0.gguf",
    embedding_n_ctx: int = 2048,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> FastMCP:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    server = FastMCP(
        "scip-ultimate-indexer",
        log_level="WARNING",
        host=host,
        port=port,
    )
    indexers: dict[tuple[str, str], UltimateIndexer] = {}

    def _backend_for_request(embedding_backend: str) -> str:
        if embedding_backend and embedding_backend != "auto":
            return embedding_backend
        model_candidate = Path(embedding_model).expanduser()
        if model_candidate.exists():
            return "local"
        return "auto"

    def get_indexer(project: str | None, embedding_backend: str) -> UltimateIndexer:
        resolved_path = _resolve_project_path(cache_dir, project)
        backend = _backend_for_request(embedding_backend)
        cache_key = (str(resolved_path), backend)
        indexer = indexers.get(cache_key)
        if indexer is None:
            indexer = UltimateIndexer(resolved_path, embedding_backend=backend)
            if embedding_model:
                indexer.settings.model_path = embedding_model
            if embedding_n_ctx > 0:
                import os

                os.environ["ULTIMATE_INDEXER_LLAMA_N_CTX"] = str(embedding_n_ctx)
                os.environ["ULTIMATE_INDEXER_LLAMA_N_BATCH"] = str(embedding_n_ctx)
                os.environ["ULTIMATE_INDEXER_LLAMA_N_UBATCH"] = str(embedding_n_ctx)
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
    def index_project(
        project_path: str,
        force: bool = False,
        embedding_backend: str = "auto",
    ) -> str:
        resolved_path = Path(project_path).expanduser().resolve()
        _register_project(cache_dir, resolved_path)
        indexer = get_indexer(str(resolved_path), embedding_backend)
        summary = indexer.index(force=force)
        return (
            f"Indexed {summary.indexed_files} files, {summary.indexed_symbols} symbols, "
            f"{summary.indexed_edges} edges, {summary.indexed_chunks} chunks"
        )

    @server.tool()
    def list_projects() -> str:
        projects = _load_registry(cache_dir)
        if not projects:
            return "// No projects found. Use index_project first."
        lines = ["// Available projects:"]
        for project in projects:
            lines.append(f"//   {project['name']}  ({project['path']})")
        return "\n".join(lines)

    @server.tool()
    def search_symbols(
        query: str,
        project: str | None = None,
        count: int = 10,
        kind: str | None = None,
        hybrid: bool = True,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        groups = indexer.query(query, limit=count)
        if kind:
            normalized = _normalized_kind(kind)
            filtered_groups = []
            for group in groups:
                selected = [
                    symbol
                    for symbol in group.symbols
                    if _normalized_kind(symbol.kind) == normalized
                ]
                if not selected:
                    continue
                group.symbols = selected
                filtered_groups.append(group)
            groups = filtered_groups
        if not hybrid:
            groups = groups[:count]
        return format_groups_compact(indexer.storage, indexer.project_id, groups)

    @server.tool()
    def get_important_symbols(
        project: str | None = None,
        count: int = 20,
        metric: str = "pagerank",
        kind: str | None = None,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        rows = indexer.important_symbols(limit=count, metric=metric, kind_filter=kind)
        return truncate_text(format_top_symbols(rows, include_scores=True), 4_000)

    @server.tool()
    def get_project_overview(
        project: str | None = None,
        max_per_kind: int = 15,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        return truncate_text(indexer.project_overview(max_per_kind=max_per_kind), 8_000)

    @server.tool()
    def get_stats(
        project: str | None = None,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        return indexer.project_stats()

    @server.tool()
    def scored_project_tree(
        project: str | None = None,
        embedding_backend: str = "auto",
        max_chars: int = 12_000,
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        return indexer.scored_tree(max_chars=max_chars)

    @server.tool()
    def visualize_project(
        query: str,
        project: str | None = None,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        groups = indexer.query(query, limit=limit)
        path = indexer.visualize(groups, title=f"Results for: {query}")
        return str(path)

    return server


def run_mcp(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    cache_dir: Path | str = ".scip_indexes",
    embedding_model: str = "models/coderankembed-q8_0.gguf",
    embedding_n_ctx: int = 2048,
) -> None:
    server = build_mcp(
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        embedding_n_ctx=embedding_n_ctx,
        host=host,
        port=port,
    )
    server.run(transport=transport)
