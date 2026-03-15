from __future__ import annotations

import json
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


def generate_visualization_html(nodes: list[dict], edges: list[dict], title: str, description: str) -> str:
    template = _template_environment().get_template("query_graph.html")
    return template.render(
        title=title,
        description=description,
        node_count=len(nodes),
        edge_count=len(edges),
        nodes_json=json.dumps(nodes).replace("</", "<\\/"),
        edges_json=json.dumps(edges).replace("</", "<\\/"),
    )


def write_query_visualization(
    storage: Storage,
    project_id: str,
    groups: list[FileGroup],
    output_path: Path,
    title: str,
) -> Path:
    symbol_rows = storage.get_symbol_rows(project_id)
    all_edges = storage.get_edges(project_id)
    selected_ids: set[str] = set()
    for group in groups:
        for symbol in group.symbols[:3]:
            selected_ids.add(symbol.symbol_id)
            row = symbol_rows.get(symbol.symbol_id)
            if row and row["enclosing_symbol_id"]:
                selected_ids.add(str(row["enclosing_symbol_id"]))
    for edge in all_edges:
        source = str(edge["source_symbol_id"])
        target = str(edge["target_symbol_id"])
        if source in selected_ids or target in selected_ids:
            selected_ids.add(source)
            selected_ids.add(target)

    nodes = []
    for symbol_id in sorted(selected_ids):
        row = symbol_rows.get(symbol_id)
        if row is None:
            continue
        color = _node_color(str(row["kind"]))
        nodes.append(
            {
                "id": symbol_id,
                "label": str(row["display_name"]),
                "group": str(row["kind"]),
                "title": f"{row['kind']}\\n{row['relative_path']}",
                "color": color,
            }
        )

    edges = []
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

    output_path.write_text(
        generate_visualization_html(
            nodes=nodes,
            edges=edges,
            title=title,
            description="CodeGraphContext-inspired interactive symbol graph",
        ),
        encoding="utf-8",
    )
    return output_path
