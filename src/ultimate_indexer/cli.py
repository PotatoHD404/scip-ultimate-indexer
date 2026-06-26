from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .formatter import format_groups, format_top_symbols
from .indexer import UltimateIndexer
from .mcp_server import run_mcp
from .models import IndexProgress
from .scip_runner import StructuredIndexingRequiredError
from .tui import run_tui


app = typer.Typer(add_completion=False)
console = Console()


def _build_indexer(
    project_path: Path,
    embedding_backend: str | None,
    cache_dir: Path | None,
    embedding_api_key: str | None = None,
    embedding_api_endpoint: str | None = None,
    embedding_api_model: str | None = None,
) -> UltimateIndexer:
    indexer = UltimateIndexer(
        project_path,
        embedding_backend=embedding_backend,
        cache_base_dir=cache_dir.resolve() if cache_dir is not None else None,
    )
    # Override API settings if provided via CLI
    if embedding_api_key:
        indexer.settings.embedding_api_key = embedding_api_key
    if embedding_api_endpoint:
        indexer.settings.embedding_api_endpoint = embedding_api_endpoint
    if embedding_api_model:
        indexer.settings.embedding_api_model = embedding_api_model
    return indexer


def _ensure_indexed(indexer: UltimateIndexer, project_path: Path) -> None:
    if indexer.storage.get_project_signature(indexer.project_id) is None:
        console.print(
            f"[red]No index found for {project_path}. Run:  "
            f"ultimate-indexer index {project_path}[/red]"
        )
        raise typer.Exit(code=1)


class _IndexProgressDisplay:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
            disable=not enabled,
        )
        self._task: int | None = None

    def __enter__(self) -> "_IndexProgressDisplay":
        self.progress.start()
        self._task = self.progress.add_task("Indexing 0/0", total=1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.stop()

    def update(self, event: IndexProgress) -> None:
        if self._task is None:
            return
        detail = event.detail or event.stage.replace("-", " ").title()
        if event.total > 0:
            detail = f"{detail} ({min(event.completed, event.total)}/{event.total} {event.unit})"
        if event.total > 0:
            stage_fraction = event.completed / event.total
        else:
            stage_fraction = 0.0
        overall_completed = (event.stage_index - 1) + stage_fraction
        self.progress.update(
            self._task,
            total=float(event.stage_total),
            completed=overall_completed,
            description=f"Indexing {event.stage_index}/{event.stage_total}: {detail}",
            refresh=True,
        )


@app.command()
def index(
    project_path: Path,
    force: bool = typer.Option(False, "--force"),
    scip_path: Path | None = typer.Option(None, "--scip-path"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key", help="API key for remote embedding service"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint", help="API endpoint URL for remote embedding service"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model", help="Model name for remote embedding service"),
) -> None:
    indexer = _build_indexer(
        project_path,
        embedding_backend,
        cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        try:
            with _IndexProgressDisplay(enabled=progress and console.is_terminal) as display:
                summary = indexer.index(
                    scip_path=scip_path,
                    force=force,
                    progress_callback=display.update if progress else None,
                )
        except StructuredIndexingRequiredError as exc:
            console.print(f"[red]{exc.render_message()}[/red]")
            raise typer.Exit(code=1) from exc
        table = Table(title="Index Summary")
        table.add_column("Files")
        table.add_column("Symbols")
        table.add_column("Edges")
        table.add_column("Chunks")
        table.add_column("Artifacts")
        table.add_row(
            str(summary.indexed_files),
            str(summary.indexed_symbols),
            str(summary.indexed_edges),
            str(summary.indexed_chunks),
            str(summary.artifact_files),
        )
        console.print(table)
        if summary.warnings:
            console.print("[yellow]SCIP warnings; fallback coverage was used where needed:[/yellow]")
            for warning in summary.warnings:
                console.print(f"[yellow]- {warning}[/yellow]")
    finally:
        indexer.close()


@app.command()
def query(
    project_path: Path,
    text: str,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
    scope: str = typer.Option("all", "--scope", help="Search scope: all, code, docs"),
    focus: list[str] = typer.Option(
        None,
        "--focus",
        help="Bias results toward these files and their git co-change neighbours (repeatable).",
    ),
) -> None:
    indexer = _build_indexer(
        project_path,
        embedding_backend,
        cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        normalized_scope = scope.strip().lower()
        if normalized_scope not in {"all", "code", "docs"}:
            raise typer.BadParameter("scope must be one of: all, code, docs")
        _ensure_indexed(indexer, project_path)
        groups = indexer.query(
            text,
            limit=limit,
            scope=normalized_scope,  # type: ignore[arg-type]
            focus_paths=tuple(focus or ()),
        )
        console.print(format_groups(indexer.storage, indexer.project_id, groups))
    finally:
        indexer.close()


@app.command("top-symbols")
def top_symbols(
    project_path: Path,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
) -> None:
    indexer = _build_indexer(
        project_path,
        embedding_backend,
        cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        _ensure_indexed(indexer, project_path)
        console.print(format_top_symbols(indexer.top_symbols(limit=limit)))
    finally:
        indexer.close()


@app.command()
def visualize(
    project_path: Path,
    query: str,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
) -> None:
    indexer = _build_indexer(
        project_path,
        embedding_backend,
        cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        groups = indexer.query(query, limit=limit)
        output_path = indexer.visualize(groups, title=f"Results for: {query}")
        console.print(str(output_path))
    finally:
        indexer.close()


@app.command()
def tree(
    project_path: Path,
    top_k: int | None = typer.Option(None, "--top-k"),
    max_tokens: int | None = typer.Option(3_000, "--max-tokens"),
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
) -> None:
    indexer = _build_indexer(
        project_path,
        embedding_backend,
        cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )
    try:
        _ensure_indexed(indexer, project_path)
        console.print(indexer.sorted_tree(max_tokens=max_tokens, top_k=top_k))
    finally:
        indexer.close()


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_model: str = typer.Option("models/coderankembed-q8_0.gguf", "--embedding-model"),
    embedding_n_ctx: int = typer.Option(2048, "--embedding-n-ctx"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
) -> None:
    run_mcp(
        transport=transport,
        host=host,
        port=port,
        cache_dir=cache_dir,
        embedding_model=embedding_model,
        embedding_n_ctx=embedding_n_ctx,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )


@app.command()
def tui(
    project_path: Path,
    embedding_backend: str | None = typer.Option(None, "--embedding-backend"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    embedding_api_key: str | None = typer.Option(None, "--embedding-api-key"),
    embedding_api_endpoint: str | None = typer.Option(None, "--embedding-api-endpoint"),
    embedding_api_model: str | None = typer.Option(None, "--embedding-api-model"),
) -> None:
    run_tui(
        project_path=project_path,
        embedding_backend=embedding_backend,
        cache_dir=cache_dir,
        embedding_api_key=embedding_api_key,
        embedding_api_endpoint=embedding_api_endpoint,
        embedding_api_model=embedding_api_model,
    )


if __name__ == "__main__":
    app()
