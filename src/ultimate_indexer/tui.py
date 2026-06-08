from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .formatter import format_groups, format_top_symbols
from .indexer import UltimateIndexer
from .scip_runner import StructuredIndexingRequiredError

console = Console()


def _build_indexer(
    project_path: Path,
    embedding_backend: str | None,
    cache_dir: Path | None,
    embedding_api_key: str | None,
    embedding_api_endpoint: str | None,
    embedding_api_model: str | None,
) -> UltimateIndexer:
    indexer = UltimateIndexer(
        project_path,
        embedding_backend=embedding_backend,
        cache_base_dir=cache_dir.resolve() if cache_dir is not None else None,
    )
    if embedding_api_key:
        indexer.settings.embedding_api_key = embedding_api_key
    if embedding_api_endpoint:
        indexer.settings.embedding_api_endpoint = embedding_api_endpoint
    if embedding_api_model:
        indexer.settings.embedding_api_model = embedding_api_model
    return indexer


def run_tui(
    *,
    project_path: Path,
    embedding_backend: str | None,
    cache_dir: Path | None,
    embedding_api_key: str | None,
    embedding_api_endpoint: str | None,
    embedding_api_model: str | None,
) -> None:
    indexer = _build_indexer(
        project_path=project_path,
        embedding_backend=embedding_backend,
        cache_dir=cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        while True:
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"Project: {project_path.resolve()}",
                            "1) Index",
                            "2) Search code",
                            "3) Search docs",
                            "4) Search all",
                            "5) Top symbols",
                            "6) Stats",
                            "7) Tree",
                            "0) Exit",
                        ]
                    ),
                    title="Ultimate Indexer TUI",
                )
            )
            action = IntPrompt.ask("Choose action", default=0)
            if action == 0:
                return
            if action == 1:
                force = Prompt.ask("Force reindex?", choices=["y", "n"], default="n") == "y"
                try:
                    summary = indexer.index(force=force)
                except StructuredIndexingRequiredError as exc:
                    console.print(f"[red]{exc.render_message()}[/red]")
                    continue
                table = Table(title="Index Summary")
                table.add_column("Files")
                table.add_column("Symbols")
                table.add_column("Edges")
                table.add_column("Chunks")
                table.add_row(
                    str(summary.indexed_files),
                    str(summary.indexed_symbols),
                    str(summary.indexed_edges),
                    str(summary.indexed_chunks),
                )
                console.print(table)
                continue
            if action in {2, 3, 4}:
                query = Prompt.ask("Query")
                limit = IntPrompt.ask("Limit", default=10)
                scope = "code" if action == 2 else "docs" if action == 3 else "all"
                groups = indexer.query(query, limit=limit, scope=scope)  # type: ignore[arg-type]
                console.print(format_groups(indexer.storage, indexer.project_id, groups))
                continue
            if action == 5:
                limit = IntPrompt.ask("Limit", default=10)
                console.print(format_top_symbols(indexer.top_symbols(limit=limit)))
                continue
            if action == 6:
                console.print(indexer.project_stats())
                continue
            if action == 7:
                max_tokens = IntPrompt.ask("Max tokens", default=3_000)
                console.print(indexer.sorted_tree(max_tokens=max_tokens))
                continue
            console.print("[yellow]Unknown action[/yellow]")
    finally:
        indexer.close()
