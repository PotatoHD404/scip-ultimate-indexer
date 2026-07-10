# scip-ultimate-indexer — Agent Guide

**Python 3.12+, Poetry.** A self-contained SCIP-first code indexer with semantic search, BM25 + dense embeddings + PPR re-ranking, MCP server, and D3.js visualisation. Standalone, no external containers needed.

## Features

- SCIP-based indexing: Go, Python, TypeScript, Rust, Java, C/C++, and more
- Hybrid search: BM25 (lexical) + dense embeddings (semantic) + PPR (graph-aware)
- Three embedding backends: API (remote), local GGUF (`llama-cpp-python`), hash (deterministic fallback, no model)
- MCP server (`fastmcp`) for AI tool integration (OpenCode, Claude Desktop)
- Token-aware context packing for LLM consumption
- D3.js force-directed graph visualisation
- Query caching (content-hashed, SQLite)
- HyDE (Hypothetical Document Embeddings) for NL query quality
- Two-stage feature reranker
- Git history signals (recency, churn, co-change)

## Quick commands

```bash
ultimate-indexer index /path/to/project          # Index a project
ultimate-indexer query /path/to/project "query"   # Semantic search
ultimate-indexer top-symbols /path/to/project     # Top symbols by PageRank
ultimate-indexer tree /path/to/project            # Scored project tree
ultimate-indexer visualize /path/to/project "q"   # D3.js graph
ultimate-indexer mcp                              # Start MCP server (stdio)
```

## Installation

```bash
git clone <repo> && cd scip-ultimate-indexer

# Via Poetry
poetry install                                    # basic (hash fallback)
poetry install -E local-embeddings                 # with llama-cpp-python GGUF

# Via pip
pip install -e .                                  # basic
pip install -e ".[local-embeddings]"              # with GGUF

# Via uv
uv pip install -e .
```

## Key dependencies

`typer`, `rich`, `networkx`, `numpy`, `protobuf`, `mcp`, `jinja2`, `pyyaml`, `tiktoken`, `pathspec`, `llama-cpp-python` (optional)

## Embedding backends

| Backend | Env | Requires |
|---------|-----|----------|
| API | `ULTIMATE_INDEXER_EMBEDDING_BACKEND=api` + endpoint/model/key | Network |
| Local GGUF | `ULTIMATE_INDEXER_EMBEDDING_BACKEND=local` + model path | `llama-cpp-python` |
| Hash | `ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash` | Nothing (default fallback) |

Auto-detection order: API > local GGUF > hash.

## MCP tools (13 total)

| Tool | Parameters | Purpose |
|------|-----------|---------|
| `index_project` | `project_path`, `force`, `embedding_backend` | Index/re-index |
| `search_code` | `query`, `project`, `count`, `kind`, `hybrid`, `focus` | Code-only search |
| `search_docs` | `query`, `project`, `count`, `kind`, `hybrid`, `focus` | Docs-only search |
| `search_all` | `query`, `project`, `count`, `kind`, `hybrid`, `focus` | Combined (RRF merged) |
| `search_symbols` | (same as `search_all`) | Legacy alias |
| `get_important_symbols` | `project`, `count`, `kind` | Top PageRank symbols |
| `get_project_overview` | `project`, `max_per_kind` | Categorized overview |
| `get_stats` | `project`, `embedding_backend` | Stats; pass `""` to list projects |
| `get_context` | `project`, `symbol_tokens`, `doc_tokens` | Token-budgeted snapshot |
| `scored_project_tree` | `project`, `max_tokens`, `top_k` | File tree by symbol value |
| `sorted_project_tree` | (same as `scored_project_tree`) | Tree by descendant score |
| `visualize_project` | `query`, `project`, `limit` | D3.js query graph |
| `list_projects` | ❌ Removed | OpenCode bug workaround — use `get_stats(project="")` |

## OpenCode configuration

See `README.md` section "MCP Server (OpenCode / Claude Desktop)" for the full config block. Key env vars:

```json
"ULTIMATE_INDEXER_EMBEDDING_BACKEND": "api",
"ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT": "https://.../v1/embeddings",
"ULTIMATE_INDEXER_EMBEDDING_API_MODEL": "text-embedding-bge-m3",
"ULTIMATE_INDEXER_EMBEDDING_API_KEY": "sk-..."
```

Optional HyDE:
```json
"ULTIMATE_INDEXER_HYDE_API_ENDPOINT": "https://.../v1/chat/completions",
"ULTIMATE_INDEXER_HYDE_API_MODEL": "tgpt/t-pro-it-2-1-fp8",
"ULTIMATE_INDEXER_HYDE_API_KEY": "sk-..."
```

## Known issues

- **OpenCode 1.17.5 `{}""` bug**: OpenCode's MCP client sends `{}""` (trailing double quotes) for zero-parameter tool calls. `list_projects` was removed as a result. Use `get_stats(project="")` instead. A `json.loads` monkey-patch is applied to the MCP server to mitigate, but the error originates client-side before reaching our server.
- **First query is slow** (~3-6s for large projects): cold-start loads vectors from SQLite and runs PPR on 239k edges (BFF). Subsequent queries are cached and fast.
- **`refresh_if_stale` disabled**: was adding 2-5s latency on every search. Re-index explicitly via `index_project()` if needed.