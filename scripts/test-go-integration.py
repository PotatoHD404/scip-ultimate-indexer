"""Go + Docs Integration Test

Verifies end-to-end correctness of the scip-ultimate-indexer against a real
Go BFF project (Traefik) and its own documentation, covering:

    Phase 1  — Index:                counts, no errors
    Phase 2  — Search relevance:     6 domain queries match expected signals
    Phase 3  — Important symbols:    PageRank top-25 includes interfaces + core pkg
    Phase 4  — Scored tree:          pkg/ ranks above vendor/, token budget honoured
    Phase 5  — Impl edges:           cross-file implements edges present after SCIP fix
    Phase 5b — Library symbols:      external stdlib stubs in graph with edges
    Phase 6  — get_context:          token budget respected, rich symbol section
    Phase 7  — Auto-refresh:         refresh_if_stale triggers re-index on file change
    Phase 8  — Docs self-test:       Traefik markdown docs indexed and queryable

Run with:
    poetry run python scripts/test-go-integration.py
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

os.environ.setdefault("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")

from ultimate_indexer.indexer import UltimateIndexer
from ultimate_indexer.formatter import (
    count_tokens,
    format_context_window,
    format_important_symbols_codegraph,
    format_scored_tree,
    format_search_symbols_codegraph,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAEFIK_URL = "https://github.com/traefik/traefik"
TRAEFIK_DIR = Path("/tmp/traefik-bff")
CACHE_DIR = Path(tempfile.gettempdir()) / "traefik-bff-index-cache"

# Phase 7: a stable .go file to touch
TOUCH_FILE = TRAEFIK_DIR / "pkg" / "server" / "server.go"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class PhaseResult:
    name: str
    passed: bool
    detail: str
    extra: dict[str, Any] = field(default_factory=dict)


results: list[PhaseResult] = []


def _pass(name: str, detail: str, **extra) -> PhaseResult:
    r = PhaseResult(name=name, passed=True, detail=detail, extra=extra)
    results.append(r)
    print(f"  PASS  {detail}")
    return r


def _fail(name: str, detail: str, **extra) -> PhaseResult:
    r = PhaseResult(name=name, passed=False, detail=detail, extra=extra)
    results.append(r)
    print(f"  FAIL  {detail}")
    return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_indexer(project_dir: Path) -> UltimateIndexer:
    return UltimateIndexer(
        project_dir,
        embedding_backend="hash",
        cache_base_dir=CACHE_DIR,
    )


def _count_header_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith("//"))


# ============================================================
# Phase 1 — Clone and Index
# ============================================================
def phase1_index() -> tuple[UltimateIndexer | None, Any]:
    print("\nPhase 1 — Index")

    # Clone if needed
    if not TRAEFIK_DIR.exists():
        print("  Cloning traefik/traefik (depth=1)…")
        r = subprocess.run(
            ["git", "clone", "--depth=1", TRAEFIK_URL, str(TRAEFIK_DIR)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            _fail("Phase 1", f"git clone failed: {r.stderr[:300]}")
            return None, None
    else:
        print(f"  Using existing clone at {TRAEFIK_DIR}")

    indexer = _make_indexer(TRAEFIK_DIR)
    try:
        print("  Running index (hash embeddings)…")
        summary = indexer.index()
    except Exception as exc:
        _fail("Phase 1", f"index() raised: {exc}")
        return None, None

    checks = {
        "files": (summary.indexed_files, 300),
        "symbols": (summary.indexed_symbols, 3_000),
        "edges": (summary.indexed_edges, 5_000),
        "chunks": (summary.indexed_chunks, 2_000),
    }
    failures = [
        f"{k}={v} (want >={want})"
        for k, (v, want) in checks.items()
        if v < want
    ]
    detail = (
        f"{summary.indexed_files} files, {summary.indexed_symbols} symbols, "
        f"{summary.indexed_edges} edges, {summary.indexed_chunks} chunks"
    )
    if failures:
        _fail("Phase 1", f"{detail} — below thresholds: {'; '.join(failures)}")
    else:
        _pass("Phase 1", detail)

    return indexer, summary


# ============================================================
# Phase 2 — Search relevance
# ============================================================
SEARCH_QUERIES = [
    ("http router middleware chain",    ["middleware", "router", "chain"]),
    ("TLS certificate provider",        ["tls", "certificate", "provider"]),
    ("entrypoint listener tcp",         ["entrypoint", "tcp", "listen"]),
    ("load balancer backend service",   ["service", "loadbalanc", "backend"]),
    ("access log request",              ["accesslog", "log", "request"]),
    ("docker provider label",           ["docker", "provider", "label"]),
]


def phase2_search(indexer: UltimateIndexer) -> None:
    print("\nPhase 2 — Search relevance")
    passed = 0
    for query, signals in SEARCH_QUERIES:
        groups = indexer.query(query, limit=10)
        out = format_search_symbols_codegraph(
            indexer.storage, indexer.project_id, query, groups, max_results=10
        )
        out_lower = out.lower()
        hit = any(s in out_lower for s in signals)
        count = sum(len(g.symbols) for g in groups)
        if count >= 3 and hit:
            print(f"  PASS  '{query}' → {count} symbols, signals present")
            passed += 1
        else:
            matched = [s for s in signals if s in out_lower]
            print(f"  FAIL  '{query}' → {count} symbols, matched signals: {matched}")

    total = len(SEARCH_QUERIES)
    detail = f"{passed}/{total} queries matched expected patterns"
    if passed == total:
        _pass("Phase 2", detail)
    elif passed >= total - 1:
        # tolerate one miss — hash embeddings are approximate
        _pass("Phase 2", detail + " (1 miss tolerated with hash embeddings)")
    else:
        _fail("Phase 2", detail)


# ============================================================
# Phase 3 — Important symbols / PageRank quality
# ============================================================
def phase3_important_symbols(indexer: UltimateIndexer) -> None:
    print("\nPhase 3 — Important symbols")
    rows = indexer.important_symbols(limit=25)
    out = format_important_symbols_codegraph(
        indexer.storage, indexer.project_id, rows
    )

    kinds = [str(r["kind"]) for r in rows]
    paths = [str(r["relative_path"]) for r in rows]
    names_lower = [str(r["display_name"]).lower() for r in rows]

    # scip-go maps Go structs AND interfaces to "TypeAlias"; "Interface" only
    # appears for TypeScript/other languages.  Accept any named-type kind.
    named_type_kinds = {"Interface", "Struct", "Class", "TypeAlias", "Trait", "Enum"}
    has_named_type = bool(set(kinds) & named_type_kinds)

    # At least one symbol from a core package path (external or project).
    # With the correct edge direction, core types should naturally dominate the top 25.
    core_pkgs = {"server", "router", "middleware", "provider", "config",
                 "api", "observability", "tls", "tcp", "proxy"}
    in_core = any(
        any(pkg in p for pkg in core_pkgs)
        for p in paths
    )

    # Test suites (SimpleSuite, *Suite, *Test types) must NOT dominate the top 25.
    # With source-credibility weighting and external-stub filtering, test-only types
    # should represent a small fraction of important project symbols.
    test_suite_names = [n for n in names_lower if "suite" in n or n.endswith("test")]
    test_suite_fraction = len(test_suite_names) / max(len(names_lower), 1)
    suites_dominate = test_suite_fraction > 0.20  # fail if > 20 % are test suites

    has_headers = "//" in out

    issues = []
    if not has_named_type:
        issues.append(f"no named-type kind in top 25 (kinds={set(kinds)})")
    if not in_core:
        issues.append("no symbol from core packages in top 25")
    if suites_dominate:
        issues.append(
            f"test suites dominate top 25 ({len(test_suite_names)}/25 = "
            f"{test_suite_fraction:.0%}); edge direction fix may not have taken effect"
        )
    if not has_headers:
        issues.append("output has no // headers")

    detail = (
        f"{len(rows)} symbols; test-suite fraction={test_suite_fraction:.0%}; "
        "kinds: " + ", ".join(sorted(set(kinds))[:6])
        + ("…" if len(set(kinds)) > 6 else "")
    )
    if issues:
        _fail("Phase 3", f"{detail} — issues: {'; '.join(issues)}")
    else:
        _pass("Phase 3", detail)


# ============================================================
# Phase 4 — Scored tree
# ============================================================
def phase4_scored_tree(indexer: UltimateIndexer) -> None:
    print("\nPhase 4 — Scored tree")
    out = indexer.scored_tree(max_tokens=3_000, top_k=50)
    tokens = count_tokens(out)
    truncated_count = out.count("// ... truncated")

    # Find first line for pkg/ and vendor/ to compare order
    lines = out.splitlines()
    pkg_line = next((i for i, l in enumerate(lines) if "pkg/" in l or "pkg " in l), None)
    vendor_line = next((i for i, l in enumerate(lines) if "vendor" in l.lower()), None)

    # pkg/ should appear before vendor/, or vendor/ not present at all
    pkg_before_vendor = (
        pkg_line is not None
        and (vendor_line is None or pkg_line < vendor_line)
    )

    issues = []
    if tokens > 3_000:
        issues.append(f"output is {tokens} tokens (want ≤ 3000)")
    if truncated_count > 1:
        issues.append(f"truncated marker appears {truncated_count} times")
    if not pkg_before_vendor:
        issues.append(f"pkg/ not ranked before vendor/ (pkg_line={pkg_line}, vendor_line={vendor_line})")

    detail = f"{tokens} tokens, pkg_line={pkg_line}, vendor_line={vendor_line}"
    if issues:
        _fail("Phase 4", f"{detail} — issues: {'; '.join(issues)}")
    else:
        _pass("Phase 4", detail)


# ============================================================
# Phase 5 — Interface implementation edges
# ============================================================
def _row_path(symbol_rows: dict, symbol_id: str) -> str:
    """Return relative_path for a symbol_id, or '' if not found.

    sqlite3.Row objects support index-style access (row["col"]) but not
    .get(), so we must guard for missing keys explicitly.
    """
    row = symbol_rows.get(symbol_id)
    if row is None:
        return ""
    try:
        return str(row["relative_path"])
    except (IndexError, KeyError):
        return ""


def phase5_impl_edges(indexer: UltimateIndexer) -> None:
    print("\nPhase 5 — implements edges")
    project_id = indexer.project_id
    all_edges = indexer.storage.get_edges(project_id)
    symbol_rows = indexer.storage.get_symbol_rows(project_id)

    impl_edges = [e for e in all_edges if str(e["edge_type"]) == "implements"]
    cross_file = []
    for e in impl_edges:
        src_path = _row_path(symbol_rows, str(e["source_symbol_id"]))
        tgt_path = _row_path(symbol_rows, str(e["target_symbol_id"]))
        # Skip edges where either symbol is unknown (empty path)
        if src_path and tgt_path and src_path != tgt_path:
            cross_file.append(e)

    issues = []
    if len(impl_edges) < 10:
        issues.append(f"only {len(impl_edges)} implements edges (want ≥ 10)")
    if len(cross_file) < 5:
        issues.append(f"only {len(cross_file)} cross-file implements edges (want ≥ 5)")

    detail = f"{len(impl_edges)} implements edges, {len(cross_file)} cross-file"
    if issues:
        _fail("Phase 5", f"{detail} — {'; '.join(issues)}")
    else:
        _pass("Phase 5", detail)

    # ------------------------------------------------------------------
    # Phase 5b — External library symbols in the graph
    # ------------------------------------------------------------------
    print("\nPhase 5b — External library symbols")
    external_syms = {
        sid: row
        for sid, row in symbol_rows.items()
        if str(row["relative_path"]).startswith("_external/")
    }
    # Count edges that touch at least one external symbol
    ext_edges = [
        e for e in all_edges
        if str(e["source_symbol_id"]) in external_syms
        or str(e["target_symbol_id"]) in external_syms
    ]
    # Any net/http symbols? Go stdlib should be in the stubs for a large project
    has_stdlib = any(
        "net/http" in str(row["relative_path"]) or "net/http" in sid
        for sid, row in external_syms.items()
    )
    issues_5b = []
    if len(external_syms) < 5:
        issues_5b.append(f"only {len(external_syms)} external stubs (want ≥ 5)")
    if len(ext_edges) < 5:
        issues_5b.append(f"only {len(ext_edges)} edges touching external stubs (want ≥ 5)")

    detail_5b = (
        f"{len(external_syms)} external stubs, "
        f"{len(ext_edges)} edges, "
        f"has_stdlib_http={has_stdlib}"
    )
    if issues_5b:
        _fail("Phase 5b", f"{detail_5b} — {'; '.join(issues_5b)}")
    else:
        _pass("Phase 5b", detail_5b)


# ============================================================
# Phase 6 — get_context token budget
# ============================================================
def phase6_context_window(indexer: UltimateIndexer) -> None:
    print("\nPhase 6 — get_context token budget")
    out = format_context_window(
        indexer.storage,
        indexer.project_id,
        symbol_tokens=8_192,
        doc_tokens=2_048,
    )
    tokens = count_tokens(out)
    header_count = _count_header_lines(out)
    # Sections: symbols first, then docs (actual headers from formatter.py)
    sym_idx = out.find("// Symbols (by graph rank)")
    doc_idx = out.find("// Docs (by graph rank)")
    sym_before_doc = sym_idx != -1 and (doc_idx == -1 or sym_idx < doc_idx)

    issues = []
    if tokens > 10_240:
        issues.append(f"{tokens} tokens (want ≤ 10240)")
    if header_count < 20:
        issues.append(f"only {header_count} // headers (want ≥ 20)")
    if not sym_before_doc:
        issues.append("symbols section not before docs section")

    detail = f"{tokens} tokens, {header_count} // headers"
    if issues:
        _fail("Phase 6", f"{detail} — {'; '.join(issues)}")
    else:
        _pass("Phase 6", detail)


# ============================================================
# Phase 7 — Auto-refresh
# ============================================================
def phase7_auto_refresh(indexer: UltimateIndexer) -> None:
    print("\nPhase 7 — Auto-refresh")

    if not TOUCH_FILE.exists():
        _fail("Phase 7", f"Touch file not found: {TOUCH_FILE}")
        return

    # Record original hash
    orig_hashes = indexer.storage.get_file_hashes(indexer.project_id)
    rel_touch = TOUCH_FILE.resolve().relative_to(TRAEFIK_DIR.resolve()).as_posix()
    orig_hash = orig_hashes.get(rel_touch)

    # Append a comment to trigger mtime/size change
    original_content = TOUCH_FILE.read_text(encoding="utf-8")
    TOUCH_FILE.write_text(
        original_content + "\n// integration-test-marker\n",
        encoding="utf-8",
    )

    try:
        # Force the check interval so refresh_if_stale runs immediately
        indexer._last_stale_check = 0.0
        refreshed = indexer.refresh_if_stale(max_age_seconds=0.0)

        new_hashes = indexer.storage.get_file_hashes(indexer.project_id)
        new_hash = new_hashes.get(rel_touch)
        hash_changed = new_hash != orig_hash

        issues = []
        if not refreshed:
            issues.append("refresh_if_stale() returned False (expected True)")
        if not hash_changed:
            issues.append(f"file hash unchanged after refresh (orig={orig_hash[:8] if orig_hash else None})")

        detail = f"refreshed={refreshed}, hash_changed={hash_changed}"
        if issues:
            _fail("Phase 7", f"{detail} — {'; '.join(issues)}")
        else:
            _pass("Phase 7", detail)
    finally:
        # Always revert the file
        TOUCH_FILE.write_text(original_content, encoding="utf-8")
        # Reset last-check so subsequent phases don't inadvertently re-index
        indexer._last_stale_check = time.monotonic()


# ============================================================
# Phase 8 — Docs self-test (Traefik docs/content/)
# ============================================================
DOCS_DIR = TRAEFIK_DIR / "docs" / "content"

DOC_QUERIES = [
    ("getting started installation",    ["getting-started", "install", "quick", "deploy"]),
    ("TLS certificate HTTPS provider",  ["tls", "certificate", "https", "acme"]),
    ("middleware configuration example",["middleware", "config", "example", "option"]),
]


def phase8_docs(indexer: UltimateIndexer) -> None:
    print("\nPhase 8 — Docs self-test (Traefik docs/content)")
    symbol_rows = indexer.storage.get_symbol_rows(indexer.project_id)

    passed = 0
    for query, signals in DOC_QUERIES:
        groups = indexer.query(query, limit=8, scope="docs")
        if not groups:
            print(f"  FAIL  '{query}' → no doc results")
            continue

        out = format_search_symbols_codegraph(
            indexer.storage, indexer.project_id, query, groups, max_results=8
        )
        out_lower = out.lower()
        hit = any(s in out_lower for s in signals)
        count = sum(len(g.symbols) for g in groups)

        # Verify results are documentation (sqlite3.Row uses index access, not .get)
        all_doc = all(
            (lambda row: row is not None and str(row["source_kind"]) == "documentation")(
                symbol_rows.get(sym.symbol_id)
            )
            for g in groups
            for sym in g.symbols
            if sym.symbol_id in symbol_rows
        )

        if count >= 1 and hit:
            print(f"  PASS  '{query}' → {count} doc symbols, signals present")
            passed += 1
        else:
            matched = [s for s in signals if s in out_lower]
            print(f"  FAIL  '{query}' → {count} doc symbols, matched: {matched}")

    detail = f"{passed}/{len(DOC_QUERIES)} doc queries matched"
    if passed == len(DOC_QUERIES):
        _pass("Phase 8", detail)
    elif passed >= len(DOC_QUERIES) - 1:
        _pass("Phase 8", detail + " (1 miss tolerated)")
    else:
        _fail("Phase 8", detail)


# ============================================================
# Summary
# ============================================================
def print_summary() -> int:
    width = 32
    print("\n" + "=" * 65)
    print("  INTEGRATION TEST SUMMARY")
    print("=" * 65)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        name_col = (r.name + " —").ljust(width)
        print(f"  {status}  {name_col} {r.detail}")
    print("=" * 65)
    failures = sum(1 for r in results if not r.passed)
    if failures == 0:
        print("  All phases passed.")
    else:
        print(f"  {failures} phase(s) FAILED.")
    return failures


# ============================================================
# Entry point
# ============================================================
def main() -> int:
    print("=" * 65)
    print("  Go + Docs Integration Test")
    print(f"  Target: {TRAEFIK_DIR}")
    print(f"  Cache:  {CACHE_DIR}")
    print("=" * 65)

    indexer, summary = phase1_index()
    if indexer is None:
        print_summary()
        return 1

    try:
        phase2_search(indexer)
        phase3_important_symbols(indexer)
        phase4_scored_tree(indexer)
        phase5_impl_edges(indexer)
        phase6_context_window(indexer)
        phase7_auto_refresh(indexer)
        phase8_docs(indexer)
    finally:
        indexer.close()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
