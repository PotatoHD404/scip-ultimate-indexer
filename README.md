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
poetry run ultimate-indexer mcp
```

The MCP server also exposes a scored project tree view that ranks files and folders by usefulness using indexed symbol ranks plus lightweight structural fallbacks.

For value-oriented inspection, the CLI `tree` command and MCP `sorted_project_tree` tool show the same tree with folders sorted by accumulated descendant score and files annotated by direct value.

Its MCP tool names now align with the graph-indexer style as well: `list_projects`, `search_symbols`, `get_important_symbols`, `get_project_overview`, and `get_stats`, while keeping `index_project`, `visualize_project`, `scored_project_tree`, and `sorted_project_tree` as extra helpers. The `mcp` command also accepts graph-indexer-style flags such as `--cache-dir`, `--embedding-model`, `--embedding-n-ctx`, `--transport`, `--host`, and `--port`.

## Embeddings

The runtime now defaults to the local GGUF model path used by `graph-indexer`.
No Hugging Face download path is used anymore.

- the first choice is `ULTIMATE_INDEXER_MODEL_PATH` when set
- otherwise the indexer looks in `models/` and `graph-indexer/models/`
- the default committed filename is `coderankembed-q8_0.gguf`
- `auto` falls back to the deterministic `hash` backend only when no local GGUF model is available

Ranking and query scoring exclude low-signal generated artifacts such as `.next` outputs, protobuf-generated `proto/*` code, `*.pb.go`, and `*_pb2.*` files. Those files are still stored for inspection, but they no longer dominate `top-symbols` or semantic query results.

For CI and tests, set `ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash` to use a deterministic local embedding backend with no model download.

Useful debug envs:

- `ULTIMATE_INDEXER_MODEL_PATH` to point at a specific local GGUF file
- `ULTIMATE_INDEXER_LLAMA_VERBOSE=true` to expose backend/device logs
- `ULTIMATE_INDEXER_LLAMA_SUPPRESS_LOGS=false` to stop silencing `llama.cpp` stderr/stdout
- `ULTIMATE_INDEXER_LLAMA_N_GPU_LAYERS`, `ULTIMATE_INDEXER_LLAMA_N_CTX`, `ULTIMATE_INDEXER_LLAMA_N_BATCH`, and `ULTIMATE_INDEXER_LLAMA_N_UBATCH` to override runtime settings

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
- built-in zero-config Python SCIP emission when no external `.scip` index is provided
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

## Notes on coverage

This project now matches the original repos much more closely on ignore handling, extension coverage, retry/backoff behavior, and non-SCIP fallback indexing. Full symbol-level parity with every CodeGraphContext language parser still depends on either a real SCIP index or the relevant external SCIP tool being available for that language.
