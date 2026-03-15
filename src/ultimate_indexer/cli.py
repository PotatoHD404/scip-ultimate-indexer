from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .formatter import format_groups, format_top_symbols
from .indexer import UltimateIndexer
from .mcp_server import run_mcp


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def index(
    project_path: Path,
    force: bool = typer.Option(False, "--force"),
    scip_path: Path | None = typer.Option(None, "--scip-path"),
    embedding_backend: str = typer.Option("auto", "--embedding-backend"),
) -> None:
    indexer = UltimateIndexer(project_path, embedding_backend=embedding_backend)
    try:
        summary = indexer.index(scip_path=scip_path, force=force)
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
