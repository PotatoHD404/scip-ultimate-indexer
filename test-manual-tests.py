#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.python_scip import emit_python_scip


PROJECT_PATH = ROOT / "test-manual-project"
SCIP_PATH = ROOT / ".manual-test-python.scip"
RESULTS_PATH = ROOT / "manual-test-results.json"


def heading(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def count_duplicates(rows: list[tuple]) -> dict[str, int]:
    counts = Counter(rows)
    return {json.dumps(list(key)): count for key, count in counts.items() if count > 1}


def build_python_scip(project_path: Path) -> Path:
    py_files = sorted(project_path.rglob("*.py"))
    emit_python_scip(project_path, py_files, SCIP_PATH)
    return SCIP_PATH


def run_manual_checks(project_path: Path) -> dict:
    scip_path = build_python_scip(project_path)
    indexer = UltimateIndexer(project_path, embedding_backend="hash")
    try:
        summary = indexer.index(scip_path=scip_path, force=True)
        conn = indexer.storage.connection
        project_id = indexer.project_id

        files = conn.execute(
            "SELECT relative_path, source_kind, language FROM files WHERE project_id = ? ORDER BY relative_path",
            (project_id,),
        ).fetchall()
        symbols = conn.execute(
            "SELECT symbol_id, kind, relative_path, source_kind FROM symbols WHERE project_id = ? ORDER BY symbol_id",
            (project_id,),
        ).fetchall()
        edges = conn.execute(
            "SELECT source_symbol_id, target_symbol_id, edge_type, weight FROM edges WHERE project_id = ? ORDER BY source_symbol_id, target_symbol_id, edge_type, weight",
            (project_id,),
        ).fetchall()
        chunks = conn.execute(
            "SELECT chunk_id, relative_path, symbol_id, chunk_kind FROM chunks WHERE project_id = ? ORDER BY chunk_id",
            (project_id,),
        ).fetchall()

        symbol_ids = {str(row[0]) for row in symbols}
        missing_edge_endpoints = [
            {
                "source": str(source),
                "target": str(target),
                "edge_type": str(edge_type),
            }
            for source, target, edge_type, _ in edges
            if str(source) not in symbol_ids or str(target) not in symbol_ids
        ]

        outgoing = Counter(str(source) for source, _, _, _ in edges)
        incoming = Counter(str(target) for _, target, _, _ in edges)
        isolated_symbols = []
        for symbol_id, kind, relative_path, source_kind in symbols:
            if str(kind) in {"File", "Module"}:
                continue
            degree = outgoing[str(symbol_id)] + incoming[str(symbol_id)]
            if degree == 0:
                isolated_symbols.append(
                    {
                        "symbol_id": str(symbol_id),
                        "kind": str(kind),
                        "relative_path": str(relative_path),
                        "source_kind": str(source_kind),
                    }
                )

        docs_files = [
            {
                "relative_path": str(relative_path),
                "source_kind": str(source_kind),
                "language": str(language),
            }
            for relative_path, source_kind, language in files
            if str(relative_path).endswith(".md")
            or "/docs/" in f"/{relative_path}"
            or str(relative_path).startswith("docs/")
        ]

        queries = [
            "authentication",
            "database connection pooling",
            "installation pip install",
            "pull request contributing",
        ]
        query_results = {}
        for query in queries:
            groups = indexer.query(query, limit=6)
            rendered = []
            doc_hits = 0
            for group in groups:
                is_doc = group.relative_path.endswith(
                    ".md"
                ) or group.relative_path.startswith("docs/")
                if is_doc:
                    doc_hits += 1
                rendered.append(
                    {
                        "relative_path": group.relative_path,
                        "score": round(group.score, 6),
                        "doc_hit": is_doc,
                        "symbols": [
                            {
                                "display_name": symbol.display_name,
                                "kind": symbol.kind,
                                "score": round(symbol.score, 6),
                            }
                            for symbol in group.symbols[:3]
                        ],
                    }
                )
            query_results[query] = {
                "doc_hits_in_top_6": doc_hits,
                "results": rendered,
            }

        scored_tree = indexer.scored_tree(top_k=10)
        top_symbols = [
            {
                "display_name": str(row["display_name"]),
                "kind": str(row["kind"]),
                "relative_path": str(row["relative_path"]),
                "global_rank": round(float(row["global_rank"]), 6),
            }
            for row in indexer.top_symbols(limit=10)
        ]

        exact_edge_duplicates = count_duplicates(
            [
                (str(source), str(target), str(edge_type), float(weight))
                for source, target, edge_type, weight in edges
            ]
        )
        semantic_edge_duplicates = count_duplicates(
            [
                (str(source), str(target), str(edge_type))
                for source, target, edge_type, _ in edges
            ]
        )

        report = {
            "project_path": str(project_path),
            "summary": {
                "indexed_files": summary.indexed_files,
                "indexed_symbols": summary.indexed_symbols,
                "indexed_edges": summary.indexed_edges,
                "indexed_chunks": summary.indexed_chunks,
                "documentation_files": summary.documentation_files,
                "warnings": summary.warnings,
            },
            "counts": {
                "files": len(files),
                "symbols": len(symbols),
                "edges": len(edges),
                "chunks": len(chunks),
                "documentation_like_files": len(docs_files),
            },
            "duplicates": {
                "files_by_relative_path": count_duplicates(
                    [(str(relative_path),) for relative_path, _, _ in files]
                ),
                "symbols_by_symbol_id": count_duplicates(
                    [(str(symbol_id),) for symbol_id, _, _, _ in symbols]
                ),
                "chunks_by_chunk_id": count_duplicates(
                    [(str(chunk_id),) for chunk_id, _, _, _ in chunks]
                ),
                "edges_exact": exact_edge_duplicates,
                "edges_same_endpoints_and_type": semantic_edge_duplicates,
            },
            "graph_checks": {
                "missing_edge_endpoints": missing_edge_endpoints,
                "self_loops": [
                    {
                        "symbol_id": str(source),
                        "edge_type": str(edge_type),
                    }
                    for source, target, edge_type, _ in edges
                    if str(source) == str(target)
                ],
                "isolated_non_file_symbols": isolated_symbols,
            },
            "docs_files": docs_files,
            "query_results": query_results,
            "top_symbols": top_symbols,
            "scored_tree": scored_tree,
        }
        return report
    finally:
        indexer.close()


def print_report(report: dict) -> None:
    heading("Manual Graph / Docs Test Report")
    print(json.dumps(report["summary"], indent=2))

    heading("Duplicate Checks")
    for name, payload in report["duplicates"].items():
        status = "PASS" if not payload else "FAIL"
        print(f"{status}: {name}")
        if payload:
            print(json.dumps(payload, indent=2))

    heading("Graph Checks")
    graph_checks = report["graph_checks"]
    print(f"Missing edge endpoints: {len(graph_checks['missing_edge_endpoints'])}")
    print(f"Self loops: {len(graph_checks['self_loops'])}")
    print(
        f"Isolated non-file symbols: {len(graph_checks['isolated_non_file_symbols'])}"
    )
    if graph_checks["isolated_non_file_symbols"]:
        print(json.dumps(graph_checks["isolated_non_file_symbols"][:10], indent=2))

    heading("Indexed Docs")
    print(json.dumps(report["docs_files"], indent=2))

    heading("Query Results")
    for query, payload in report["query_results"].items():
        print(f"Query: {query}")
        print(json.dumps(payload, indent=2))

    heading("Top Symbols")
    print(json.dumps(report["top_symbols"], indent=2))

    heading("Scored Tree")
    print(report["scored_tree"])


if __name__ == "__main__":
    report = run_manual_checks(PROJECT_PATH)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_report(report)
    print(f"\nSaved raw report to {RESULTS_PATH}")
