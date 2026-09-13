# scip-ultimate-indexer — instructions

## What it is

Standalone sibling repo (not inside `deployka-infra`). Self-contained Python/Poetry SCIP-first code
indexer: SCIP parsing + built-in Python SCIP emission, embedded SQLite storage, hybrid BM25+vector
retrieval, weighted personalized PageRank, a Typer CLI, an MCP server, a TUI, and docs ingestion.
Public: `github.com/PotatoHD404/scip-ultimate-indexer`, `master`. Consumed as a RAG context engine
by `../gh-dataset-forge`'s D5 dataset and by that repo's curriculum-finetune PR enhancement — import
`ultimate_indexer.indexer.UltimateIndexer` as a library, or run `ultimate-indexer mcp`. No host, no
deploy target.

## Build / test / run

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install "mcp>=1.26.0"
.venv/bin/pip install -e . --no-deps
```

```bash
ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash .venv/bin/python -m pytest -q -W ignore::DeprecationWarning
.venv/bin/python scripts/smoke_test.py     # 15 checks
```

`llama_cpp` is not installed in the dev venv — set `ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash` (the
deterministic, zero-setup backend) for tests and local runs; `local` needs a GGUF model and
`llama-cpp-python` (the `local-embeddings` extra), `api` points at an OpenAI-compatible endpoint.

```bash
poetry install
poetry run ultimate-indexer index /path/to/project
poetry run ultimate-indexer query /path/to/project "how is auth handled?"
poetry run ultimate-indexer top-symbols /path/to/project --limit 20
poetry run ultimate-indexer mcp
```

Structural (SCIP-toolchain) indexing needs external tools per language: `scip-go`, `scip-python`,
`scip-typescript`, `rust-analyzer`, `scip-java` (via coursier bootstrap), `scip-clang`. Restrict with
`SCIP_LANGUAGES=<langs>`; disable entirely with `ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP=1` (tests do
this by default in `tests/conftest.py` — scip-python alone is 30-100s per project). Without a
language's tool, it falls back to tree-sitter/AST or char/section chunking, not an error.

## Deploy

None — a library/CLI/MCP server, installed via `poetry install` or `pip install -e`. No CI deploy
step beyond lint/test.

## Traps

- `index` caches to `<repo>/.ultimate_indexer/index.sqlite3`, keyed by config hash. After editing
  any ranking/scoring code, re-run with `index --force` — scores are computed and stored at index
  time, so a stale cache silently shows old rankings.
- Test-path damping (`_test_multiplier` in `indexer.py`) is `0.05`, not `0.25` — at `0.25` a single
  large test class could still dominate `top-symbols`, burying real product code. Don't raise it
  back without re-verifying against a repo with a big test suite.
- `scip-java` and `scip-clang` actually **compile/build** the target project — a repo whose own
  build fails (multi-module Maven reactors, old `<source>1.8>` bytecode mismatches, gpg-signed
  builds) will fall back to non-structural indexing even though the tool is correctly wired. That is
  the target repo's build problem, not a scip-ultimate-indexer bug.
- A repo with both Maven and Gradle needs `--build-tool=maven` explicitly, or scip-java aborts
  ("Multiple build tools detected").
- CMake C++ projects get `compile_commands.json` auto-generated into the cache
  (`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` + `--compdb-path`) — never written into the user's repo.
  Non-CMake C/C++ projects need a hand-provided `compile_commands.json` (e.g. via `bear`).
- `tiktoken` is pinned `>=0.9,<1.0` and the lock resolves to `0.13.0` specifically — `^0.9` on its
  own has no cp314 wheel and forces a Rust source build on Python 3.14.
- The tiktoken encoding must stay lazily loaded (module-level `get_encoding()` at import time
  downloads over the network and freezes the MCP stdio handshake on a cold cache).
- Java/Kotlin/openjdk on macOS via Homebrew are keg-only — `/usr/libexec/java_home -v 21` will not
  find them; point `JAVA_HOME` at
  `/opt/homebrew/opt/openjdk@<NN>/libexec/openjdk.jdk/Contents/Home` directly.
- `(project_id, symbol_id)` must be deduped before insert in `storage.py` — a duplicate raises the
  SQLite UNIQUE-constraint crash that was hit and fixed during the gh-dataset-forge retrieval
  benchmark.
- `poetry install` resolves from public PyPI by default (`[[tool.poetry.source]]` artifactory mirror
  is `priority = "explicit"`, opt-in only) — don't assume the internal mirror is reachable or needed.

## Deeper notes

- `project_gh_dataset_forge_and_code_indexers` — the 2026-06-08 26-agent audit, the Tier-1
  search-quality pass, the prod-readiness pass (packaging/formatter/scoring/docs-ingest fixes), the
  scip-clang/scip-java wiring, and the retrieval benchmark that validated this indexer against RAG.
