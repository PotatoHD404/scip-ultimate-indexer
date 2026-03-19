from __future__ import annotations

import sys
from pathlib import Path

import anyio

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

def test_query_project_stays_connected_and_returns_compact_output(fixture_project: Path) -> None:
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
                indexed = await session.call_tool(
                    "index_project",
                    {
                        "project_path": str(fixture_project),
                        "force": True,
                        "embedding_backend": "hash",
                    },
                )
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

                assert indexed.isError is False
                assert first.isError is False
                assert second.isError is False
                assert first.content
                assert second.content
                assert first.content[0].type == "text"
                assert second.content[0].type == "text"
                assert len(first.content[0].text) <= 4_000
                assert len(second.content[0].text) <= 4_000

                top = await session.call_tool(
                    "top_project_symbols",
                    {
                        "project_path": str(fixture_project),
                        "limit": 5,
                        "embedding_backend": "hash",
                    },
                )
                assert top.isError is False
                assert top.content
                assert "score=" not in top.content[0].text

    anyio.run(exercise_mcp)
