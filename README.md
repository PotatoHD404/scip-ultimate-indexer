# SCIP Ultimate Indexer

A self-contained SCIP-first code indexer with semantic search, BM25 + dense embedding + PPR re-ranking, and an MCP server for AI agent integration.

## Quick Start

```bash
# Install dependencies
pip install -e .                            # basic (hash embeddings, no model download)
pip install -e ".[local-embeddings]"         # with local GGUF embeddings (llama-cpp-python)
# or via Poetry:
# poetry install && poetry shell

# Index a project
ultimate-indexer index /path/to/project

# Search
ultimate-indexer query /path/to/project "how is authentication handled?"

# Top symbols by PageRank importance
ultimate-indexer top-symbols /path/to/project --limit 20

# Scored project tree (most valuable files first)
ultimate-indexer tree /path/to/project

# Visualise query results as D3.js graph
ultimate-indexer visualize /path/to/project "auth"

# MCP server (stdio mode for AI tools like OpenCode)
ultimate-indexer mcp
```

## Installation

### Prerequisites

- Python 3.12+
- For local embeddings: a C++ compiler (to build `llama-cpp-python`)

### Via pip

```bash
# Basic (uses hash embeddings — no model download, always works)
pip install git+https://github.com/sourcegraph/scip-ultimate-indexer.git

# With local GGUF embeddings (better results offline)
pip install "scip-ultimate-indexer[local-embeddings] @ git+https://github.com/sourcegraph/scip-ultimate-indexer.git"
```

### Via Poetry

```bash
git clone <repo-url> && cd scip-ultimate-indexer
poetry install
# Optional: add GGUF local embeddings
poetry install -E local-embeddings
```

### Via uv (fast)

```bash
uv pip install -e .
# or with local embeddings:
uv pip install -e ".[local-embeddings]"
```

### Configure embedding backend

The indexer supports three backends, in order of preference:

1. **API** — remote embedding API (requires endpoint + model)
2. **Local GGUF** — `llama-cpp-python` with a local model file
3. **Hash** — deterministic hash vectors, no model required (default fallback)

```bash
# API embeddings (recommended for speed)
export ULTIMATE_INDEXER_EMBEDDING_BACKEND=api
export ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT=https://llm-proxy.example.com/v1/embeddings
export ULTIMATE_INDEXER_EMBEDDING_API_MODEL=text-embedding-bge-m3
export ULTIMATE_INDEXER_EMBEDDING_API_KEY=sk-...

# Local GGUF (offline)
export ULTIMATE_INDEXER_EMBEDDING_BACKEND=local
export ULTIMATE_INDEXER_MODEL_PATH=/path/to/coderankembed-q8_0.gguf

# Hash (zero setup, works everywhere)
export ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash
```

## MCP Server (OpenCode / Claude Desktop)

Run the MCP server for integration with AI coding tools:

```bash
ultimate-indexer mcp
```

### OpenCode config

Add to `~/.config/opencode/opencode.json`:

```json
"ultimateIndexer": {
  "type": "local",
  "command": ["<venv>/bin/python", "-m", "ultimate_indexer.cli", "mcp"],
  "environment": {
    "ULTIMATE_INDEXER_EMBEDDING_BACKEND": "api",
    "ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT": "https://llm-proxy.example.com/v1/embeddings",
    "ULTIMATE_INDEXER_EMBEDDING_API_MODEL": "text-embedding-bge-m3",
    "ULTIMATE_INDEXER_EMBEDDING_API_KEY": "sk-...",
    "ULTIMATE_INDEXER_EMBEDDING_API_TIMEOUT_SECONDS": "120",
    "ULTIMATE_INDEXER_EMBEDDING_API_MAX_RETRIES": "3",
    "ULTIMATE_INDEXER_EMBEDDING_API_BATCH_SIZE": "16",
    "ULTIMATE_INDEXER_CACHE_DIR": "/path/to/cache"
  },
  "enabled": true
}
```

Optionally enable HyDE for better natural-language queries:

```json
"ULTIMATE_INDEXER_HYDE_API_ENDPOINT": "https://llm-proxy.example.com/v1/chat/completions",
"ULTIMATE_INDEXER_HYDE_API_MODEL": "tgpt/t-pro-it-2-1-fp8",
"ULTIMATE_INDEXER_HYDE_API_KEY": "sk-..."
```

### MCP tools

| Tool | Description |
|------|-------------|
| `index_project` | Index or re-index a project |
| `search_code` | Semantic search over code only |
| `search_docs` | Semantic search over documentation only |
| `search_all` | Combined code + docs search (RRF merged) |
| `search_symbols` | Legacy alias for `search_all` |
| `get_important_symbols` | Top-ranked symbols by PageRank |
| `get_project_overview` | Categorized symbol overview |
| `get_stats` | Index statistics (use `project=""` to list projects) |
| `get_context` | Token-budgeted context snapshot |
| `scored_project_tree` | File tree ranked by symbol usefulness |
| `sorted_project_tree` | Same as scored, sorted by descendant score |
| `visualize_project` | D3.js force-directed graph of query results |

> **Note:** `list_projects` was removed due to an OpenCode 1.17.5 serialisation bug (sends `{}""` for zero-param tools). Use `get_stats` with `project=""` instead.

### HyDE (Hypothetical Document Embeddings)

HyDE generates a short hypothetical code snippet from your natural-language query and blends it into the query vector. This makes NL questions like "how is auth handled?" land closer to actual code.

```bash
# Enable HyDE (on by default)
export ULTIMATE_INDEXER_ENABLE_HYDE=true
export ULTIMATE_INDEXER_HYDE_BLEND=0.5

# Optional: LLM-backed HyDE for better generation
export ULTIMATE_INDEXER_HYDE_API_ENDPOINT=https://api.example.com/v1/chat/completions
export ULTIMATE_INDEXER_HYDE_API_MODEL=gpt-4o-mini
export ULTIMATE_INDEXER_HYDE_API_KEY=sk-...
```

If no HyDE API is configured, a fast deterministic template is used instead.

## Embeddings

The embedding backend is auto-detected:

1. If `ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT` + `_MODEL` are set → **API backend**
2. If a local GGUF model is found → **local backend**
3. Otherwise → **hash backend** (deterministic, no model)

### Debug env vars

| Env | Purpose |
|-----|---------|
| `ULTIMATE_INDEXER_MODEL_PATH` | Specify GGUF model path |
| `ULTIMATE_INDEXER_LLAMA_VERBOSE` | Show llama.cpp backend logs |
| `ULTIMATE_INDEXER_LLAMA_N_GPU_LAYERS` | GPU offload layers |
| `ULTIMATE_INDEXER_LLAMA_N_CTX` | Context size (default 2048) |
| `ULTIMATE_INDEXER_EMBEDDING_API_TIMEOUT_SECONDS` | API timeout (default 120) |
| `ULTIMATE_INDEXER_EMBEDDING_API_MAX_RETRIES` | API retries (default 3) |
| `ULTIMATE_INDEXER_EMBEDDING_API_RETRY_BASE_DELAY_MS` | Retry backoff base (default 500) |
| `ULTIMATE_INDEXER_EMBEDDING_API_BATCH_SIZE` | Batch size (default 16) |

## Search Quality Upgrades

All on by default, degrade gracefully (no models or git required):

- **Hybrid BM25 + dense embeddings** — lexical + semantic, PPR re-ranked
- **Vocabulary expansion** — identifier splitting, abbreviation bridging
- **Contextual embeddings** — file/module context prepended to each chunk
- **HyDE** — hypothetical code snippet blend for NL queries
- **Two-stage reranker** — feature-based rerank after fusion
- **Git history signals** — recency/churn/co-change boosts (when `.git` available)
- **Context-focused ranking** — bias results toward files you're working in

Toggle with env vars:

| Env | Default | Purpose |
|-----|---------|---------|
| `ULTIMATE_INDEXER_ENABLE_HYDE` | `true` | Hypothetical Document Embeddings |
| `ULTIMATE_INDEXER_ENABLE_RERANKER` | `true` | Two-stage feature reranker |
| `ULTIMATE_INDEXER_ENABLE_EXPANSION` | `true` | Vocabulary expansion |
| `ULTIMATE_INDEXER_ENABLE_CONTEXTUAL` | `true` | Contextual chunk headers |
| `ULTIMATE_INDEXER_ENABLE_GIT_SIGNALS` | `true` | Git-history boosts |

## SCIP Language Support

SCIP indexers are auto-detected from `PATH`. Supported languages:

| Language | Tool | Notes |
|----------|------|-------|
| Python | `scip-python` (external) + built-in emitter | Zero-config fallback always works |
| Go | `scip-go` | |
| TypeScript / JS | `scip-typescript` | |
| Rust | `rust-analyzer` | |
| Java | `scip-java` | Needs Maven/Gradle build |
| C / C++ | `scip-clang` | Needs `compile_commands.json` |

Set `ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP=1` to skip all external SCIP tools.

## What Gets Indexed

- Local source files via SCIP ingestion
- Python gets built-in structured symbols (function/class/method) even without `scip-python`
- Broad file discovery: Python, Go, JS/TS, Java, Kotlin, Scala, C/C++, C#, Rust, Ruby, PHP, Swift, Bash, HTML/CSS, Vue, Svelte, SQL, Dart, Lua, R, Elixir, Haskell, Perl, Dockerfile, Makefile
- Documentation files (Markdown, OpenAPI YAML)
- Fallback chunking for unsupported languages
- Config artifacts (`.socraticodecontextartifacts.json`)

### Ignore rules

Combines built-in defaults (deps, build output, caches, lockfiles, `.ultimate_indexer`) + `.gitignore` + optional `.socraticodeignore` / `.cgcignore`.

## Output Format

Query results grouped by file, with function signatures and docstrings:

```
// pkg/service.py
"""Greets users and prepares API payloads."""
def build_greeting(user: User, excited: bool = False) -> str:
    # skipped 12 rows
```

## Project Structure

```
src/ultimate_indexer/
├── cli.py          # CLI entry point (typer)
├── mcp_server.py   # MCP server (FastMCP)
├── indexer.py      # Main indexing + query orchestration
├── query.py        # Query engine (BM25 + dense + PPR)
├── storage.py      # SQLite storage layer
├── embeddings.py   # Embedding providers (API/local/hash)
├── scip_runner.py  # External SCIP tool runner
├── scip_parser.py  # SCIP binary parser
├── pagerank.py     # Weighted PageRank
├── hyde.py         # Hypothetical Document Embeddings
├── reranker.py     # Feature-based reranker
├── models.py       # Data models
├── config.py       # Configuration
├── formatter.py    # Output formatting
├── python_scip.py  # Built-in Python SCIP emitter
├── docs/           # Documentation ingestion
└── templates/      # Jinja2 templates (D3.js visuals)
```

## Tests

```bash
# All tests
pytest

# Smoke test (end-to-end, uses hash backend)
python scripts/smoke_test.py
```