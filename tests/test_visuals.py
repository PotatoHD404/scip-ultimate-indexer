from __future__ import annotations

from pathlib import Path

from ultimate_indexer.models import FileGroup, RankedSymbol
from ultimate_indexer.visuals import write_query_visualization


class _FakeStorage:
    def __init__(self, symbol_rows: dict[str, dict], edges: list[dict[str, str]]) -> None:
        self._symbol_rows = symbol_rows
        self._edges = edges

    def get_symbol_rows(self, project_id: str) -> dict[str, dict]:
        return self._symbol_rows

    def get_edges(self, project_id: str) -> list[dict[str, str]]:
        return self._edges


def _seed_group(symbol_id: str, relative_path: str = "pkg/services.py") -> FileGroup:
    return FileGroup(
        relative_path=relative_path,
        score=1.0,
        symbols=[
            RankedSymbol(
                symbol_id=symbol_id,
                relative_path=relative_path,
                display_name=symbol_id,
                kind="Function",
                score=1.0,
                signature="",
                docstring="",
                snippet="",
            )
        ],
    )


def test_write_query_visualization_trims_edges_and_reports_notice(tmp_path: Path) -> None:
    symbol_rows: dict[str, dict] = {}
    for index in range(10):
        symbol_id = f"s{index}"
        symbol_rows[symbol_id] = {
            "kind": "Function",
            "relative_path": "pkg/services.py",
            "display_name": f"fn{index}",
            "enclosing_symbol_id": None,
            "global_rank": float(100 - index),
        }
    edges = []
    for index in range(9):
        edges.append(
            {
                "source_symbol_id": f"s{index}",
                "target_symbol_id": f"s{index + 1}",
                "edge_type": "calls",
            }
        )
    for index in range(7):
        edges.append(
            {
                "source_symbol_id": "s0",
                "target_symbol_id": f"s{index + 2}",
                "edge_type": "uses",
            }
        )

    output_path = tmp_path / "query_graph.html"
    write_query_visualization(
        storage=_FakeStorage(symbol_rows, edges),
        project_id="project",
        groups=[_seed_group("s0")],
        output_path=output_path,
        title="Trimmed Graph",
        max_nodes=0,
        max_edges=6,
    )
    html = output_path.read_text(encoding="utf-8")
    assert "Edges: 6" in html
    assert "edges trimmed for browser performance" in html


def test_write_query_visualization_enables_performance_mode_for_large_graphs(tmp_path: Path) -> None:
    node_count = 1305
    symbol_rows: dict[str, dict] = {}
    edges = []
    for index in range(node_count):
        symbol_id = f"s{index}"
        symbol_rows[symbol_id] = {
            "kind": "Function",
            "relative_path": "pkg/large.py",
            "display_name": f"fn{index}",
            "enclosing_symbol_id": None,
            "global_rank": float(node_count - index),
        }
        if index > 0:
            edges.append(
                {
                    "source_symbol_id": f"s{index - 1}",
                    "target_symbol_id": symbol_id,
                    "edge_type": "calls",
                }
            )

    output_path = tmp_path / "query_graph.html"
    write_query_visualization(
        storage=_FakeStorage(symbol_rows, edges),
        project_id="project",
        groups=[_seed_group("s0", relative_path="pkg/large.py")],
        output_path=output_path,
        title="Large Graph",
        max_nodes=0,
        max_edges=0,
    )
    html = output_path.read_text(encoding="utf-8")
    assert "Performance mode enabled" in html
    assert "pkg/large.py::fn0" in html
    assert "pkg/large.py::fn1304" not in html


def test_write_query_visualization_does_not_trim_by_default(tmp_path: Path) -> None:
    node_count = 250
    symbol_rows: dict[str, dict] = {}
    edges = []
    for index in range(node_count):
        symbol_id = f"s{index}"
        symbol_rows[symbol_id] = {
            "kind": "Function",
            "relative_path": "pkg/full.py",
            "display_name": f"fn{index}",
            "enclosing_symbol_id": None,
            "global_rank": float(node_count - index),
        }
        if index > 0:
            edges.append(
                {
                    "source_symbol_id": f"s{index - 1}",
                    "target_symbol_id": symbol_id,
                    "edge_type": "calls",
                }
            )

    output_path = tmp_path / "query_graph.html"
    write_query_visualization(
        storage=_FakeStorage(symbol_rows, edges),
        project_id="project",
        groups=[_seed_group("s0", relative_path="pkg/full.py")],
        output_path=output_path,
        title="Full Graph",
    )
    html = output_path.read_text(encoding="utf-8")
    assert "Nodes: 250" in html
    assert "Edges: 249" in html
    assert "trimmed for browser performance" not in html
