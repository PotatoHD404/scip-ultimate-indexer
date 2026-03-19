from __future__ import annotations

import sys
from pathlib import Path

import anyio

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.python_scip import emit_python_scip


def test_query_project_stays_connected_and_returns_compact_output(fixture_project: Path) -> None:
    python_files = sorted(fixture_project.rglob("*.py"))
    scip_path = fixture_project / ".ultimate_indexer" / "cache" / "fixture.scip"
    scip_path.parent.mkdir(parents=True, exist_ok=True)
    emit_python_scip(fixture_project, python_files, scip_path)

    indexer = UltimateIndexer(fixture_project, embedding_backend="hash")
    try:
        indexer.index(force=True, scip_path=scip_path)
    finally:
        indexer.close()

    async def exercise_mcp() -> None:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ultimate_indexer", "mcp"],
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        async with stdio_client(server) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                first = await session.call_tool(
                    "query_project",
                    {
                        "project_path": str(fixture_project),
                        "query": "greeting service",
                        "limit": 5,
                        "embedding_backend": "hash",
                    },
                )
                second = await session.call_tool(
                    "query_project",
                    {
                        "project_path": str(fixture_project),
                        "query": "greeting service",
                        "limit": 5,
                        "embedding_backend": "hash",
                    },
                )

                assert first.isError is False
                assert second.isError is False
                assert first.content
                assert second.content
                assert first.content[0].type == "text"
                assert second.content[0].type == "text"
                assert len(first.content[0].text) <= 4_000
                assert len(second.content[0].text) <= 4_000

    anyio.run(exercise_mcp)
