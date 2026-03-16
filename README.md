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
poetry run ultimate-indexer visualize /path/to/project "auth"
poetry run ultimate-indexer mcp
```

## Embeddings

The runtime now defaults to `sentence-transformers` with [`nomic-ai/CodeRankEmbed`](https://huggingface.co/nomic-ai/CodeRankEmbed).

- on Apple Silicon, `auto` prefers the PyTorch `mps` device so inference runs on the Metal GPU
- on Apple Silicon, the default embedding path also uses a smaller batch size and caps sequence length to `512` tokens to keep Metal memory stable
- the query side uses the model's required prefix: `Represent this query for searching relevant code:`
- batching is enabled by default for faster indexing throughput
- `llama.cpp` remains available as a fallback/backend override

For CI and tests, set `ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash` to use a deterministic local embedding backend with no model download.

Useful debug envs:

- `ULTIMATE_INDEXER_ST_DEVICE` to force `mps`, `cuda`, or `cpu`
- `ULTIMATE_INDEXER_ST_BATCH_SIZE` to tune throughput
- `ULTIMATE_INDEXER_ST_MAX_SEQ_LENGTH` to trade context length against speed and memory use
- `ULTIMATE_INDEXER_ST_USE_FP16=true|false` to control half precision on GPU backends
- `ULTIMATE_INDEXER_ST_NORMALIZE=false` to disable normalized embeddings
- `ULTIMATE_INDEXER_LLAMA_VERBOSE=true` to expose backend/device logs
- `ULTIMATE_INDEXER_LLAMA_SUPPRESS_LOGS=false` to stop silencing `llama.cpp` stderr/stdout
- `ULTIMATE_INDEXER_LLAMA_MODEL_REPO_ID` to override the GGUF repo for the fallback `llama.cpp` backend
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

## Notes on coverage

This project now matches the original repos much more closely on ignore handling, extension coverage, retry/backoff behavior, and non-SCIP fallback indexing. Full symbol-level parity with every CodeGraphContext language parser still depends on either a real SCIP index or the relevant external SCIP tool being available for that language.
