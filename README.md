# SCIP Ultimate Indexer

`scip-ultimate-indexer` is a self-contained Python/Poetry indexer that combines:

- a SCIP-first indexing pipeline
- embedded SQLite storage instead of Neo4j/Qdrant containers
- SocratiCode-style artifact ingestion via `.socraticodecontextartifacts.json`
- hybrid lexical + vector retrieval
- weighted inverse personalized PageRank over connected symbols
- CodeGraphContext-inspired graph visuals
- an MCP server that returns code-readable grouped context

## Quick start

```bash
poetry install
poetry run ultimate-indexer index /path/to/project
poetry run ultimate-indexer query /path/to/project "how is authentication handled?"
poetry run ultimate-indexer top-symbols /path/to/project --limit 20
poetry run ultimate-indexer tree /path/to/project
poetry run ultimate-indexer visualize /path/to/project "auth"
poetry run ultimate-indexer tui /path/to/project
poetry run ultimate-indexer mcp
```

The MCP server also exposes a scored project tree view that ranks files and folders by usefulness using indexed symbol ranks plus lightweight structural fallbacks.

For value-oriented inspection, the CLI `tree` command and MCP `sorted_project_tree` tool show the same tree with folders sorted by accumulated descendant score and files annotated by direct value.

Its MCP tools now expose split retrieval scopes:
- `search_code` for code-only retrieval
- `search_docs` for documentation-only retrieval
- `search_all` to combine independently ranked code+docs results

`search_symbols` remains available as a backward-compatible alias to `search_all`. Other tools include `list_projects`, `get_important_symbols`, `get_project_overview`, `get_stats`, `index_project`, `visualize_project`, `scored_project_tree`, and `sorted_project_tree`. The `mcp` command also accepts graph-indexer-style flags such as `--cache-dir`, `--embedding-model`, `--embedding-n-ctx`, `--transport`, `--host`, and `--port`.

## Embeddings

The runtime now defaults to the local GGUF model path used by `graph-indexer`.
No Hugging Face download path is used anymore.

- the first choice is `ULTIMATE_INDEXER_MODEL_PATH` when set
- otherwise the indexer looks in `models/` and `graph-indexer/models/`
- the default committed filename is `coderankembed-q8_0.gguf`
- `auto` falls back to the deterministic `hash` backend when no local GGUF model is available **or** the `llama-cpp-python` runtime is not installed, so indexing and querying work with zero setup

Ranking and query scoring exclude low-signal generated artifacts such as `.next` outputs, protobuf-generated `proto/*` code, `*.pb.go`, and `*_pb2.*` files. Those files are still stored for inspection, but they no longer dominate `top-symbols` or semantic query results.

For CI and tests, set `ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash` to use a deterministic local embedding backend with no model download.

Useful debug envs:

- `ULTIMATE_INDEXER_MODEL_PATH` to point at a specific local GGUF file
- `ULTIMATE_INDEXER_LLAMA_VERBOSE=true` to expose backend/device logs
- `ULTIMATE_INDEXER_LLAMA_SUPPRESS_LOGS=false` to stop silencing `llama.cpp` stderr/stdout
- `ULTIMATE_INDEXER_LLAMA_N_GPU_LAYERS`, `ULTIMATE_INDEXER_LLAMA_N_CTX`, `ULTIMATE_INDEXER_LLAMA_N_BATCH`, and `ULTIMATE_INDEXER_LLAMA_N_UBATCH` to override runtime settings
- `ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT`, `ULTIMATE_INDEXER_EMBEDDING_API_MODEL`, and `ULTIMATE_INDEXER_EMBEDDING_API_KEY` for OpenAI-compatible embedding APIs
- `ULTIMATE_INDEXER_EMBEDDING_API_TIMEOUT_SECONDS`, `ULTIMATE_INDEXER_EMBEDDING_API_MAX_RETRIES`, and `ULTIMATE_INDEXER_EMBEDDING_API_RETRY_BASE_DELAY_MS` to control API resiliency
- `ULTIMATE_INDEXER_EMBEDDING_API_BATCH_SIZE` and `ULTIMATE_INDEXER_EMBEDDING_API_MAX_TOKENS` to tune API request sizing

## SCIP support

The built-in SCIP runner supports the same external toolchain family that CodeGraphContext wires up:

- `scip-python` for Python
- `scip-typescript` for TypeScript and JavaScript
- `scip-go` for Go
- `rust-analyzer` for Rust
- `scip-java` for Java
- `scip-clang` for C and C++

`SCIP_LANGUAGES` can be used to restrict auto-detection, and accepts CodeGraphContext-style values such as `python,javascript,go,rust,java,c`.

## What gets indexed

- local source files through a SCIP ingestion path
- built-in zero-config Python SCIP emission when no external `.scip` index is provided — this is also used automatically when `scip-python` is missing or fails at runtime, so Python projects get function/class/method symbols with no Node toolchain
- automatic external SCIP ingestion for detected languages when `scip-<lang>` tools are available
- broad SocratiCode-style file discovery for JavaScript, TypeScript, TSX, Python, Java, Kotlin, Scala, C, C++, C#, Go, Rust, Ruby, PHP, Swift, Bash, HTML/CSS, Vue, Svelte, config files, docs, SQL, Dart, Lua, R, Elixir, Haskell, Perl, and special files like `Dockerfile` and `Makefile`
- line and minified-content fallback chunking for non-SCIP files so unsupported languages stay searchable
- fallback import graph edges for path-based languages such as JS/TS, C/C++, Dart, Lua, Ruby, PHP, and shell scripts
- SocratiCode context artifacts from `.socraticodecontextartifacts.json`

## Ignore rules

The indexer combines:

- built-in defaults for dependency folders, build output, caches, lockfiles, editor folders, and `.ultimate_indexer`
- root and nested `.gitignore` files when `RESPECT_GITIGNORE` is not set to `false`
- optional `.socraticodeignore`
- optional upward `.cgcignore`
- `IGNORE_DIRS` for extra directory-name exclusions

It does not try to infer "unused" files semantically. If a path is not covered by those ignore sources, it is still eligible for indexing.

Use `EXTRA_EXTENSIONS=.tpl,.blade` to include project-specific plaintext extensions in discovery and fallback indexing.

## Output shape

Query results are grouped by file to reduce tokens and rendered like:

```text
// pkg/service.py
"""Greets users and prepares API payloads."""
def build_greeting(user: User, excited: bool = False) -> str:
    # skipped 12 rows
```

Struct/class-like results emit their interface plus direct method signatures.

Query ranking and `top-symbols` now follow the graph-indexer pattern more closely:

- `top-symbols` uses dependency-ordered PageRank plus symbol-kind boosts
- `query` uses lexical and semantic seeds, then expands with query-relative dependency ordering
- config/docs artifacts keep qualified keys, parent headings, and attribution in their indexed text

## Search quality upgrades

Retrieval combines several signals beyond hybrid BM25 + embeddings. All are on by
default and degrade gracefully (no models or git required):

- **Built-in zero-config Python SCIP.** Python projects get real Function/Method/Class
  symbols and call edges even when the external `scip-python` (Node) tool is missing or
  broken — the in-tree emitter is used as a fallback.
- **Vocabulary expansion** (doc2query / SPLADE-lite). Identifiers are split and common
  abbreviations bridged (e.g. a query for `authentication` matches `authnHandler`, `svc`
  matches `GreetingService`). Indexed into the lexical FTS only — never displayed.
  Toggle: `ULTIMATE_INDEXER_ENABLE_EXPANSION`.
- **Contextual embeddings.** Each chunk is embedded with a small structural context
  header (file, enclosing class/module, purpose) prepended — kept out of the BM25 text.
  Toggle: `ULTIMATE_INDEXER_ENABLE_CONTEXTUAL`.
- **HyDE** for natural-language queries. A hypothetical code snippet is generated from the
  query and blended into the dense query vector so NL questions land closer to real code.
  Toggle/tune: `ULTIMATE_INDEXER_ENABLE_HYDE`, `ULTIMATE_INDEXER_HYDE_BLEND` (default 0.5).
- **Two-stage reranking.** The fused candidate set is re-scored against the query (exact
  name/signature matches, term coverage) and blended with the first-stage score.
  Toggle/tune: `ULTIMATE_INDEXER_ENABLE_RERANKER`, `ULTIMATE_INDEXER_RERANK_BLEND`,
  `ULTIMATE_INDEXER_RERANK_TOP_K`.
- **Git history signals.** Recency and churn lift a file's symbols' global rank, and
  files that co-change in history couple together. Toggle/tune:
  `ULTIMATE_INDEXER_ENABLE_GIT_SIGNALS`, `ULTIMATE_INDEXER_GIT_SIGNAL_STRENGTH`,
  `ULTIMATE_INDEXER_GIT_HALF_LIFE_DAYS`, `ULTIMATE_INDEXER_GIT_HISTORY_LIMIT`.
- **Context-personalized ranking.** Pass `--focus <file>` (repeatable) to `query`, or the
  `focus` argument to the MCP `search_*` tools, to bias results toward the files you are
  working in and the files that historically co-change with them.
  Tune: `ULTIMATE_INDEXER_COCHANGE_WEIGHT`.

HyDE and the reranker default to fast deterministic implementations, but can be
upgraded to **model-backed variants** by pointing them at any OpenAI /
`llama.cpp`-server-compatible HTTP endpoint (they fall back automatically on any
error, so this is purely additive):

- LLM-generated HyDE via chat completions —
  `ULTIMATE_INDEXER_HYDE_API_ENDPOINT`, `ULTIMATE_INDEXER_HYDE_API_MODEL`,
  optional `ULTIMATE_INDEXER_HYDE_API_KEY` / `ULTIMATE_INDEXER_HYDE_MAX_TOKENS`.
- Cross-encoder reranker via a `/v1/rerank` endpoint (llama.cpp server, TEI,
  infinity, Jina, Cohere) — `ULTIMATE_INDEXER_RERANK_API_ENDPOINT`,
  `ULTIMATE_INDEXER_RERANK_API_MODEL`, optional `ULTIMATE_INDEXER_RERANK_API_KEY`.

```bash
poetry run ultimate-indexer query /path/to/project "how is auth handled?" --focus src/auth/login.py
```

Run `python scripts/smoke_test.py` for an end-to-end check of the whole pipeline
(uses the deterministic `hash` backend; no model download).

## Notes on coverage

This project now matches the original repos much more closely on ignore handling, extension coverage, retry/backoff behavior, and non-SCIP fallback indexing. Full symbol-level parity with every CodeGraphContext language parser still depends on either a real SCIP index or the relevant external SCIP tool being available for that language.
