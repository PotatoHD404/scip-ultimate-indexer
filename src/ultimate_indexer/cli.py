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


app = typer.Typer(add_completion=False)
console = Console()


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
    embedding_backend: str = typer.Option("auto", "--embedding-backend"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    indexer = UltimateIndexer(project_path, embedding_backend=embedding_backend)
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
    finally:
        indexer.close()


@app.command()
def query(
    project_path: Path,
    text: str,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str = typer.Option("auto", "--embedding-backend"),
) -> None:
    indexer = UltimateIndexer(project_path, embedding_backend=embedding_backend)
    try:
        groups = indexer.query(text, limit=limit)
        console.print(format_groups(indexer.storage, indexer.project_id, groups))
    finally:
        indexer.close()


@app.command("top-symbols")
def top_symbols(
    project_path: Path,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str = typer.Option("auto", "--embedding-backend"),
) -> None:
    indexer = UltimateIndexer(project_path, embedding_backend=embedding_backend)
    try:
        console.print(format_top_symbols(indexer.top_symbols(limit=limit)))
    finally:
        indexer.close()


@app.command()
def visualize(
    project_path: Path,
    query: str,
    limit: int = typer.Option(10, "--limit"),
    embedding_backend: str = typer.Option("auto", "--embedding-backend"),
) -> None:
    indexer = UltimateIndexer(project_path, embedding_backend=embedding_backend)
    try:
        groups = indexer.query(query, limit=limit)
        output_path = indexer.visualize(groups, title=f"Results for: {query}")
        console.print(str(output_path))
    finally:
        indexer.close()


@app.command()
def mcp() -> None:
    run_mcp()
