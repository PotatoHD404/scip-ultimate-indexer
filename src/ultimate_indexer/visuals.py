from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import FileGroup
from .storage import Storage


NODE_COLORS = {
    "File": {"background": "#E0E7FF", "border": "#6366F1"},
    "Module": {"background": "#F3E8FF", "border": "#A855F7"},
    "Section": {"background": "#FEE2E2", "border": "#DC2626"},
    "Function": {"background": "#D1FAE5", "border": "#10B981"},
    "Method": {"background": "#CFFAFE", "border": "#0891B2"},
    "Class": {"background": "#DBEAFE", "border": "#2563EB"},
    "Artifact": {"background": "#FDE68A", "border": "#D97706"},
    "default": {"background": "#F1F5F9", "border": "#64748B"},
}


def _node_color(kind: str) -> dict[str, str]:
    return NODE_COLORS.get(kind, NODE_COLORS["default"])


def _template_environment() -> Environment:
    template_dir = Path(__file__).with_name("templates")
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"]),
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def generate_visualization_html(
    nodes: list[dict],
    edges: list[dict],
    title: str,
    description: str,
    *,
    performance_mode: bool = False,
    dropped_node_count: int = 0,
    dropped_edge_count: int = 0,
    edge_chunk_size: int = 1200,
) -> str:
    template = _template_environment().get_template("query_graph.html")
    notices: list[str] = []
    if dropped_node_count > 0:
        notices.append(f"{dropped_node_count} nodes trimmed for browser performance.")
    if dropped_edge_count > 0:
        notices.append(f"{dropped_edge_count} edges trimmed for browser performance.")
    if performance_mode:
        notices.append("Performance mode enabled: simplified styling and chunked rendering are active.")
    return template.render(
        title=title,
        description=description,
        performance_notice=" ".join(notices),
        node_count=len(nodes),
        edge_count=len(edges),
        performance_mode=performance_mode,
        edge_chunk_size=max(200, edge_chunk_size),
        nodes_json=json.dumps(nodes).replace("</", "<\\/"),
        edges_json=json.dumps(edges).replace("</", "<\\/"),
    )


def _qualified_label(symbol_rows: dict[str, dict], symbol_id: str) -> str:
    row = symbol_rows.get(symbol_id)
    if row is None:
        return symbol_id
    relative_path = str(row["relative_path"])
    if str(row["kind"]) == "File":
        return relative_path

    parts: list[str] = []
    current_id = symbol_id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        current = symbol_rows.get(current_id)
        if current is None:
            break
        if str(current["kind"]) == "File":
            break
        display_name = str(current["display_name"]).strip()
        if display_name:
            parts.append(display_name)
        parent_id = current["enclosing_symbol_id"]
        current_id = str(parent_id) if parent_id else ""
    qualified_name = ".".join(reversed(parts))
    return f"{relative_path}::{qualified_name}" if qualified_name else relative_path


def _rank_value(symbol_rows: dict[str, dict], symbol_id: str) -> float:
    row = symbol_rows.get(symbol_id)
    if row is None:
        return 0.0
    try:
        return float(row["global_rank"])
    except Exception:
        return 0.0


def _trim_edges_for_budget(
    edges: list[dict[str, str]],
    core_ids: set[str],
    max_edges: int,
) -> tuple[list[dict[str, str]], int]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return edges, 0

    degree: Counter[str] = Counter()
    for edge in edges:
        source = edge["from"]
        target = edge["to"]
        degree[source] += 1
        degree[target] += 1

    ranked = sorted(
        edges,
        key=lambda edge: (
            edge["from"] not in core_ids and edge["to"] not in core_ids,
            -(degree[edge["from"]] + degree[edge["to"]]),
            edge["from"],
            edge["to"],
            edge.get("label", ""),
        ),
    )
    kept = ranked[:max_edges]
    return kept, len(edges) - len(kept)


def write_query_visualization(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    output_path: Path,
    title: str,
    *,
    max_nodes: int | None = None,
    max_edges: int | None = None,
) -> Path:
    if max_nodes is None:
        max_nodes = _env_int("ULTIMATE_INDEXER_VISUAL_MAX_NODES", 0)
    if max_edges is None:
        max_edges = _env_int("ULTIMATE_INDEXER_VISUAL_MAX_EDGES", 0)
    perf_node_threshold = _env_int("ULTIMATE_INDEXER_VISUAL_PERF_NODES", 1200)
    perf_edge_threshold = _env_int("ULTIMATE_INDEXER_VISUAL_PERF_EDGES", 3000)
    label_budget = _env_int("ULTIMATE_INDEXER_VISUAL_LABEL_BUDGET", 300)
    edge_chunk_size = _env_int("ULTIMATE_INDEXER_VISUAL_EDGE_CHUNK", 1200)

    symbol_rows = storage.get_symbol_rows(project_id)
    all_edges = storage.get_edges(project_id)
    selected_ids: set[str] = set()
    core_ids: set[str] = set()
    for group in groups:
        for symbol in group.symbols[:3]:
            selected_ids.add(symbol.symbol_id)
            core_ids.add(symbol.symbol_id)
            row = symbol_rows.get(symbol.symbol_id)
            if row and row["enclosing_symbol_id"]:
                parent_id = str(row["enclosing_symbol_id"])
                selected_ids.add(parent_id)
                core_ids.add(parent_id)
    for edge in all_edges:
        source = str(edge["source_symbol_id"])
        target = str(edge["target_symbol_id"])
        if source in selected_ids or target in selected_ids:
            selected_ids.add(source)
            selected_ids.add(target)

    if max_nodes > 0 and len(selected_ids) > max_nodes:
        ranked_ids = sorted(
            selected_ids,
            key=lambda symbol_id: (
                symbol_id not in core_ids,
                -_rank_value(symbol_rows, symbol_id),
                symbol_id,
            ),
        )
        dropped_node_count = len(selected_ids) - max_nodes
        selected_ids = set(ranked_ids[:max_nodes])
    else:
        dropped_node_count = 0

    edges: list[dict[str, str]] = []
    for edge in all_edges:
        source = str(edge["source_symbol_id"])
        target = str(edge["target_symbol_id"])
        if source not in selected_ids or target not in selected_ids:
            continue
        edges.append(
            {
                "from": source,
                "to": target,
                "label": str(edge["edge_type"]),
            }
        )

    edges, dropped_edge_count = _trim_edges_for_budget(edges, core_ids, max_edges=max_edges)
    performance_mode = len(selected_ids) >= perf_node_threshold or len(edges) >= perf_edge_threshold

    if performance_mode:
        ranked_ids = sorted(
            selected_ids,
            key=lambda symbol_id: (
                symbol_id not in core_ids,
                -_rank_value(symbol_rows, symbol_id),
                symbol_id,
            ),
        )
        labeled_ids = set(ranked_ids[: max(0, label_budget)])
        labeled_ids.update(core_ids)
        for edge in edges:
            edge.pop("label", None)
    else:
        labeled_ids = selected_ids

    nodes = []
    for symbol_id in sorted(selected_ids):
        row = symbol_rows.get(symbol_id)
        if row is None:
            continue
        color = _node_color(str(row["kind"]))
        label = _qualified_label(symbol_rows, symbol_id) if symbol_id in labeled_ids else ""
        nodes.append(
            {
                "id": symbol_id,
                "label": label,
                "group": str(row["kind"]),
                "title": f"{row['kind']}\\n{row['relative_path']}",
                "color": color,
            }
        )

    output_path.write_text(
        generate_visualization_html(
            nodes=nodes,
            edges=edges,
            title=title,
            description="CodeGraphContext-inspired interactive symbol graph",
            performance_mode=performance_mode,
            dropped_node_count=dropped_node_count,
            dropped_edge_count=dropped_edge_count,
            edge_chunk_size=edge_chunk_size,
        ),
        encoding="utf-8",
    )
    return output_path
