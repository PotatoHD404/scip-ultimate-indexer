from __future__ import annotations

import sys
from pathlib import Path

import anyio

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

def test_query_project_stays_connected_and_returns_compact_output(fixture_project: Path) -> None:
    async def exercise_mcp() -> None:
        cache_dir = fixture_project / ".scip_indexes-registry"
        server = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "ultimate_indexer",
                "mcp",
                "--cache-dir",
                str(cache_dir),
                "--embedding-model",
                str(Path(__file__).resolve().parents[1] / "graph-indexer" / "models" / "coderankembed-q8_0.gguf"),
            ],
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
                    "search_symbols",
                    {
                        "project": str(fixture_project),
                        "query": "greeting service",
                        "count": 5,
                        "embedding_backend": "hash",
                    },
                )
                second = await session.call_tool(
                    "search_symbols",
                    {
                        "project": str(fixture_project),
                        "query": "greeting service",
                        "count": 5,
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
                assert "// Search: greeting service" in first.content[0].text
                assert "score=" in first.content[0].text
                assert any(path.name == "index.sqlite3" for path in (cache_dir / "indexes").rglob("index.sqlite3"))

                listed = await session.call_tool(
                    "list_projects",
                    {},
                )
                assert listed.isError is False
                assert "sample_project" in listed.content[0].text

                top = await session.call_tool(
                    "get_important_symbols",
                    {
                        "project": str(fixture_project),
                        "count": 5,
                        "embedding_backend": "hash",
                    },
                )
                assert top.isError is False
                assert top.content
                assert "score=" in top.content[0].text

                overview = await session.call_tool(
                    "get_project_overview",
                    {
                        "project": str(fixture_project),
                        "embedding_backend": "hash",
                    },
                )
                assert overview.isError is False
                assert "Project overview for" in overview.content[0].text

                stats = await session.call_tool(
                    "get_stats",
                    {
                        "project": str(fixture_project),
                        "embedding_backend": "hash",
                    },
                )
                assert stats.isError is False
                assert "Files:" in stats.content[0].text

                tree = await session.call_tool(
                    "scored_project_tree",
                    {
                        "project": str(fixture_project),
                        "embedding_backend": "hash",
                    },
                )
                assert tree.isError is False
                assert tree.content
                assert "Project tree scored by usefulness." in tree.content[0].text
                assert "sample_project/" in tree.content[0].text
                assert "pkg/" in tree.content[0].text
                assert "docs/" in tree.content[0].text
                assert "app.py" in tree.content[0].text

    anyio.run(exercise_mcp)
