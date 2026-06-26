from __future__ import annotations

import atexit
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# CRITICAL: Configure logging to stderr BEFORE any other imports or code execution
# This prevents any logging from contaminating stdout (which must contain ONLY JSON-RPC)
_logging_configured = False
if not _logging_configured:
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr,
        force=True,
    )
    _logging_configured = True

from .formatter import (
    format_context_window,
    format_important_symbols_codegraph,
    format_search_symbols_codegraph,
)
from .indexer import UltimateIndexer
from .scip_runner import StructuredIndexingRequiredError


def _normalized_kind(kind: str) -> str:
    return kind.replace("_", "").replace("-", "").lower()


def _not_indexed_message(indexer: UltimateIndexer) -> str | None:
    if indexer.storage.get_project_signature(indexer.project_id) is None:
        return (
            f"// No index found for {indexer.project_id}. "
            "Run the index_project tool for this project first."
        )
    return None


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
    cache_dir: Path | str | None = None,
    embedding_model: str = "models/coderankembed-q8_0.gguf",
    embedding_n_ctx: int = 2048,
    host: str = "127.0.0.1",
    port: int = 8000,
    embedding_api_key: str | None = None,
    embedding_api_endpoint: str | None = None,
    embedding_api_model: str | None = None,
) -> FastMCP:
    # When cache_dir is None, per-project SQLite indexes live at
    # <project>/.ultimate_indexer (index_cache_dir=None) — matching the CLI so a
    # project indexed via `ultimate-indexer index` is visible here.  The shared
    # project registry then needs a stable home of its own.
    if cache_dir is None:
        index_cache_dir: Path | None = None
        registry_dir = Path.home() / ".cache" / "ultimate_indexer"
    else:
        index_cache_dir = Path(cache_dir).resolve()
        registry_dir = index_cache_dir
    registry_dir.mkdir(parents=True, exist_ok=True)
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
        # If API embedding is configured, use API backend
        if embedding_api_endpoint and embedding_api_model:
            return "api"
        model_candidate = Path(embedding_model).expanduser()
        if model_candidate.exists():
            return "local"
        return "auto"

    def get_indexer(
        project: str | None,
        embedding_backend: str,
        *,
        auto_refresh: bool = True,
    ) -> UltimateIndexer:
        resolved_path = _resolve_project_path(registry_dir, project)
        backend = _backend_for_request(embedding_backend)
        cache_key = (str(resolved_path), backend)
        indexer = indexers.get(cache_key)
        if indexer is None:
            indexer = UltimateIndexer(
                resolved_path,
                embedding_backend=backend,
                cache_base_dir=index_cache_dir,
            )
            # Configure API embedding if provided
            if embedding_api_key:
                indexer.settings.embedding_api_key = embedding_api_key
            if embedding_api_endpoint:
                indexer.settings.embedding_api_endpoint = embedding_api_endpoint
            if embedding_api_model:
                indexer.settings.embedding_api_model = embedding_api_model
            # Configure local embedding if provided
            if embedding_model:
                indexer.settings.model_path = embedding_model
            if embedding_n_ctx > 0:
                import os

                # setdefault: honour any explicit operator-set values; the server
                # flag only fills in defaults rather than mutating global state
                # differently per indexer.
                os.environ.setdefault("ULTIMATE_INDEXER_LLAMA_N_CTX", str(embedding_n_ctx))
                os.environ.setdefault("ULTIMATE_INDEXER_LLAMA_N_BATCH", str(embedding_n_ctx))
                os.environ.setdefault("ULTIMATE_INDEXER_LLAMA_N_UBATCH", str(embedding_n_ctx))
            indexers[cache_key] = indexer
        # Auto-refresh: re-index incrementally if any files changed since the
        # last index run.  Skipped for index_project (auto_refresh=False) to
        # avoid a redundant pre-scan before the explicit index call.
        if auto_refresh:
            indexer.refresh_if_stale()
        return indexer

    def close_indexers() -> None:
        while indexers:
            _, indexer = indexers.popitem()
            try:
                indexer.close()
            except Exception:
                pass

    atexit.register(close_indexers)

    def _render_search(
        *,
        query: str,
        project: str | None,
        count: int,
        kind: str | None,
        hybrid: bool,
        embedding_backend: str,
        scope: str,
        focus: list[str] | None = None,
    ) -> str:
        indexer = get_indexer(project, embedding_backend)
        not_indexed = _not_indexed_message(indexer)
        if not_indexed is not None:
            return not_indexed
        groups = indexer.query(
            query,
            limit=count,
            scope=scope,  # type: ignore[arg-type]
            focus_paths=tuple(focus or ()),
        )
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
        return format_search_symbols_codegraph(
            indexer.storage,
            indexer.project_id,
            query,
            groups,
            max_results=count,
            max_tokens=0,
        )

    @server.tool()
    def index_project(
        project_path: str,
        force: bool = False,
        embedding_backend: str = "auto",
    ) -> str:
        """Index or re-index a project so it can be queried by other tools.

        Runs SCIP indexers for supported languages (Python, TypeScript, Go,
        Rust, …), builds a symbol graph with weighted edges, embeds chunks, and
        computes global PageRank scores.  Incremental re-indexing only
        re-processes files whose content hash has changed.  The project is
        registered in the server registry so subsequent calls can reference it
        by name instead of full path.

        Parameters
        ----------
        project_path:
            Absolute (or ``~``-relative) path to the repository root.  Must be
            a directory that contains source files.
        force:
            ``True`` — delete all cached data and re-index from scratch.
            ``False`` (default) — skip unchanged files; fastest for routine
            refreshes after small edits.
        embedding_backend:
            Which embedding provider to use.
            ``"auto"`` — pick the best available (API > local GGUF > hash).
            ``"local"`` — llama-cpp GGUF model from the models/ directory.
            ``"api"`` — remote HTTP embedding endpoint configured via env vars.
            ``"hash"`` — deterministic hash vectors; no model required
            (degrades semantic search quality).

        When to use
        -----------
        Call once before using any search or overview tool.  Re-call after
        adding files or changing the embedding backend.
        """
        resolved_path = Path(project_path).expanduser().resolve()
        _register_project(registry_dir, resolved_path)
        # auto_refresh=False: the explicit index() call below handles everything;
        # running refresh_if_stale first would redundantly scan files twice.
        indexer = get_indexer(str(resolved_path), embedding_backend, auto_refresh=False)
        try:
            summary = indexer.index(force=force)
        except StructuredIndexingRequiredError as exc:
            return exc.render_message()
        return (
            f"Indexed {summary.indexed_files} files, {summary.indexed_symbols} symbols, "
            f"{summary.indexed_edges} edges, {summary.indexed_chunks} chunks"
        )

    @server.tool()
    def list_projects() -> str:
        """List all projects registered with this MCP server.

        Returns a comment-prefixed list of project names and their absolute
        paths.  A project is registered automatically the first time
        ``index_project`` is called for it.

        When to use
        -----------
        Use to discover which project names can be passed to the ``project``
        parameter of other tools, or to check whether a project has been
        indexed yet.
        """
        projects = _load_registry(registry_dir)
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
        focus: list[str] | None = None,
    ) -> str:
        """Search both code and documentation (legacy alias for ``search_all``).

        Prefer ``search_all`` for new code; this tool exists for backwards
        compatibility.

        Parameters
        ----------
        query:
            Natural-language or identifier search string.
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        count:
            Maximum number of file groups to return (default 10).
        kind:
            Filter results to a single symbol kind, e.g. ``"Function"``,
            ``"Class"``, ``"Interface"``.  Case-insensitive.  ``None`` returns
            all kinds.
        hybrid:
            ``True`` (default) — blend BM25 lexical search with dense
            embedding similarity and re-rank with Personalized PageRank.
            ``False`` — return only the top ``count`` results without
            re-ranking.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.
        """
        return _render_search(
            query=query,
            project=project,
            count=count,
            kind=kind,
            hybrid=hybrid,
            embedding_backend=embedding_backend,
            scope="all",
            focus=focus,
        )

    @server.tool()
    def search_code(
        query: str,
        project: str | None = None,
        count: int = 10,
        kind: str | None = None,
        hybrid: bool = True,
        embedding_backend: str = "auto",
        focus: list[str] | None = None,
    ) -> str:
        """Search only code symbols and source files (excludes documentation).

        Runs BM25 + dense embedding search over function metadata, function
        bodies, and generic symbol chunks, then re-ranks with Personalized
        PageRank seeded by the top query matches.  Results are grouped by file.

        Parameters
        ----------
        query:
            Natural-language or identifier search string.
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        count:
            Maximum number of file groups to return (default 10).
        kind:
            Restrict to one symbol kind: ``"Function"``, ``"Class"``,
            ``"Struct"``, ``"Interface"``, ``"Method"``, ``"Enum"``, etc.
            Case-insensitive.  ``None`` returns all code kinds.
        hybrid:
            ``True`` (default) — full hybrid BM25 + dense + PPR pipeline.
            ``False`` — top-``count`` results without PPR re-ranking.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use when the query is about source code behaviour, APIs, or
        implementation details and you do not need documentation results.
        Use ``search_all`` when you want both code and docs combined.
        """
        return _render_search(
            query=query,
            project=project,
            count=count,
            kind=kind,
            hybrid=hybrid,
            embedding_backend=embedding_backend,
            scope="code",
            focus=focus,
        )

    @server.tool()
    def search_docs(
        query: str,
        project: str | None = None,
        count: int = 10,
        kind: str | None = None,
        hybrid: bool = True,
        embedding_backend: str = "auto",
        focus: list[str] | None = None,
    ) -> str:
        """Search only documentation sources (Markdown, OpenAPI specs).

        Restricts retrieval to symbols whose ``source_kind`` is
        ``"documentation"`` or whose kind is ``"Document"`` / ``"Section"``.

        Parameters
        ----------
        query:
            Natural-language search string describing the documentation topic.
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        count:
            Maximum number of file groups to return (default 10).
        kind:
            Filter to a doc kind such as ``"Document"`` or ``"Section"``.
            Usually left as ``None``.
        hybrid:
            ``True`` (default) — BM25 + dense + PPR pipeline.
            ``False`` — pure ranking without PPR.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use when looking for explanations, guides, or API specs rather than
        source code.  Use ``search_all`` to get both code and docs together.
        """
        return _render_search(
            query=query,
            project=project,
            count=count,
            kind=kind,
            hybrid=hybrid,
            embedding_backend=embedding_backend,
            scope="docs",
            focus=focus,
        )

    @server.tool()
    def search_all(
        query: str,
        project: str | None = None,
        count: int = 10,
        kind: str | None = None,
        hybrid: bool = True,
        embedding_backend: str = "auto",
        focus: list[str] | None = None,
    ) -> str:
        """Search code and documentation, then merge results with RRF ranking.

        Runs separate code and docs searches (each with BM25 + dense + PPR),
        then merges the two ranked lists using Reciprocal Rank Fusion (k=60)
        so the most relevant results from either source surface first.

        Parameters
        ----------
        query:
            Natural-language or identifier search string.
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        count:
            Maximum number of file groups in the merged result (default 10).
            Each sub-search retrieves up to ``count * 2`` candidates before
            merging.
        kind:
            Optionally restrict to one symbol kind.  Applied after merging.
        hybrid:
            ``True`` (default) — full pipeline.
            ``False`` — skip PPR re-ranking.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Default search tool.  Use when you are unsure whether the answer lives
        in code or documentation, or when both are relevant.
        """
        return _render_search(
            query=query,
            project=project,
            count=count,
            kind=kind,
            hybrid=hybrid,
            embedding_backend=embedding_backend,
            scope="all",
            focus=focus,
        )

    @server.tool()
    def get_important_symbols(
        project: str | None = None,
        count: int = 20,
        kind: str | None = None,
        embedding_backend: str = "auto",
    ) -> str:
        """Return the top symbols by global graph PageRank importance.

        Uses pre-computed global PageRank scores (with kind boosts applied)
        rather than a query-driven search.  Results reflect the architectural
        centrality of each symbol in the codebase — heavily called or
        implemented symbols rank higher.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        count:
            Number of symbols to return (default 20).  All returned symbols
            have ``global_rank > 0``.
        kind:
            Restrict to a single kind, e.g. ``"Interface"``, ``"Struct"``,
            ``"Function"``.  Case-insensitive.  ``None`` returns all kinds.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use to understand the most architecturally significant parts of the
        codebase without a specific search query.  Useful for orientation,
        code review, or determining where to focus a refactoring effort.
        For a richer token-budget-aware view use ``get_context`` instead.
        """
        indexer = get_indexer(project, embedding_backend)
        not_indexed = _not_indexed_message(indexer)
        if not_indexed is not None:
            return not_indexed
        rows = indexer.important_symbols(limit=count, kind_filter=kind)
        return format_important_symbols_codegraph(
            indexer.storage,
            indexer.project_id,
            rows,
            max_tokens=0,
        )

    @server.tool()
    def get_project_overview(
        project: str | None = None,
        max_per_kind: int = 15,
        embedding_backend: str = "auto",
    ) -> str:
        """Return a categorized overview of the most important project symbols.

        Groups symbols into four buckets — Interfaces, Structs/Classes,
        Functions/Methods, Constants — and lists up to ``max_per_kind`` entries
        from each bucket ordered by global PageRank.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        max_per_kind:
            Maximum symbols per category (default 15).  Increase for larger
            codebases; decrease for tighter output.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use for a quick orientation at the start of a session.  Gives a
        categorized table-of-contents without requiring a query.  For a
        token-budget-aware snapshot prefer ``get_context``; for a ranked flat
        list prefer ``get_important_symbols``.
        """
        indexer = get_indexer(project, embedding_backend)
        not_indexed = _not_indexed_message(indexer)
        if not_indexed is not None:
            return not_indexed
        return indexer.project_overview(max_per_kind=max_per_kind)

    @server.tool()
    def get_stats(
        project: str | None = None,
        embedding_backend: str = "auto",
    ) -> str:
        """Return counts and high-level statistics for an indexed project.

        Includes total files, symbols, edges, embedded chunks, symbol kind
        breakdown, and top folders by file count.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use to verify that a project has been indexed correctly, check coverage
        after adding new files, or diagnose unexpected search behaviour.
        """
        indexer = get_indexer(project, embedding_backend)
        return indexer.project_stats()

    @server.tool()
    def scored_project_tree(
        project: str | None = None,
        embedding_backend: str = "auto",
        max_tokens: int = 3_000,
        top_k: int | None = None,
    ) -> str:
        """Render a project file tree scored by symbol usefulness.

        Each file receives a score that blends the sum and maximum of its
        symbols' global PageRank values plus small bonuses for symbol count and
        chunk count.  Directory scores roll up all descendant files.  The tree
        is sorted so the most useful directories/files appear at the top.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.
        max_tokens:
            Maximum output size in tokens as counted by ``count_tokens``
            (default 3 000).  Lines are removed from the end until the output
            fits; a ``// ... truncated`` note is appended when trimmed.
            Pass 0 for unlimited output.
        top_k:
            When set, show only the top *k* files by score rather than the full
            tree.  Useful for focusing on the most important entry points.

        When to use
        -----------
        Use to understand which files are the structural core of a project
        before diving into code.  Differs from ``sorted_project_tree`` only in
        the header text; both use the same ranking algorithm.
        """
        indexer = get_indexer(project, embedding_backend)
        return indexer.scored_tree(max_tokens=max_tokens, top_k=top_k)

    @server.tool()
    def sorted_project_tree(
        project: str | None = None,
        embedding_backend: str = "auto",
        max_tokens: int = 3_000,
        top_k: int | None = None,
    ) -> str:
        """Render a project file tree sorted by accumulated descendant score.

        Identical algorithm to ``scored_project_tree`` — directories are sorted
        by accumulated descendant PageRank, files by their direct score — but
        this variant's header makes the sorting criterion explicit (accumulated
        value vs. direct value) which helps when explaining the tree to users.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.
        max_tokens:
            Maximum output size in tokens (default 3 000).  Pass 0 for
            unlimited.
        top_k:
            Limit to the top *k* highest-scoring files.

        When to use
        -----------
        Prefer over ``scored_project_tree`` when you want to explain to a user
        why specific directories or files rank highly.
        """
        indexer = get_indexer(project, embedding_backend)
        return indexer.sorted_tree(max_tokens=max_tokens, top_k=top_k)

    @server.tool()
    def visualize_project(
        query: str,
        project: str | None = None,
        limit: int = 10,
        embedding_backend: str = "auto",
    ) -> str:
        """Generate an interactive HTML graph for query results and return its path.

        Runs ``search_all`` for the given query, then renders the top-``limit``
        file groups as a force-directed node-link diagram (D3.js) written to
        the project's ``.ultimate_indexer/visuals/`` directory.

        Parameters
        ----------
        query:
            Search query whose results become the graph nodes.
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.
        limit:
            Number of file groups (and their symbols/edges) to include in the
            graph (default 10).  Higher values produce denser graphs.
        embedding_backend:
            Embedding backend; see ``index_project`` for accepted values.

        When to use
        -----------
        Use when you want to visually explore how the symbols matching a query
        relate to each other through the call/type/import graph.  The returned
        file path can be opened in a browser.
        """
        indexer = get_indexer(project, embedding_backend)
        groups = indexer.query(query, limit=limit, scope="all")
        path = indexer.visualize(groups, title=f"Results for: {query}")
        return str(path)

    @server.tool()
    def get_context(
        project: str | None = None,
        symbol_tokens: int = 8192,
        doc_tokens: int = 2048,
        embedding_backend: str = "auto",
    ) -> str:
        """Return a compact code context window packed to a Qwen-token budget.

        Produces two sections assembled by greedy packing — never plain
        truncation — so the output fits as many meaningful symbols and docs as
        possible within the token budget:

        **Symbols section** (up to *symbol_tokens* Qwen tokens)
        Functions, methods, interfaces, structs, classes, traits, enums, type
        aliases, constants, and properties ordered by graph PageRank.  Each
        entry is one signature line preceded by a location comment::

            // QualifiedName  (Kind)  path:line
            func DoSomething(x int) error

        **Docs section** (up to *doc_tokens* Qwen tokens)
        Top-ranked documentation symbols (markdown files, OpenAPI specs) with
        a one-line summary.

        Parameters
        ----------
        project:
            Project name or absolute path.  Optional when only one project is
            indexed.  Use ``list_projects`` to see available names.
        symbol_tokens:
            Maximum token budget for the symbols section (default 8 192).
            Counted via tiktoken cl100k_base (Qwen2-compatible BPE) when
            tiktoken is installed, otherwise approximated at 3.5 chars/token.
        doc_tokens:
            Maximum Qwen-token budget for the documentation section (default
            2 048).  Set to 0 to omit docs entirely.
        embedding_backend:
            Embedding backend for the indexer instance.  ``"auto"`` selects
            the best available backend (API > local GGUF > hash fallback).

        When to use
        -----------
        Use ``get_context`` when you need a broad structural overview of the
        whole project — for example at the start of a coding session — and want
        a single, token-efficient snapshot rather than a query-driven result.
        For targeted searches prefer ``search_code`` or ``search_all``.
        """
        indexer = get_indexer(project, embedding_backend)
        not_indexed = _not_indexed_message(indexer)
        if not_indexed is not None:
            return not_indexed
        return format_context_window(
            indexer.storage,
            indexer.project_id,
            symbol_tokens=symbol_tokens,
            doc_tokens=doc_tokens,
        )

    return server


def run_mcp(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    cache_dir: Path | str | None = None,
    embedding_model: str = "models/coderankembed-q8_0.gguf",
    embedding_n_ctx: int = 2048,
    embedding_api_key: str | None = None,
    embedding_api_endpoint: str | None = None,
    embedding_api_model: str | None = None,
) -> None:
    server = build_mcp(
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        embedding_n_ctx=embedding_n_ctx,
        host=host,
        port=port,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    server.run(transport=transport)
