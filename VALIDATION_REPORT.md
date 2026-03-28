# Validation Report

## Scope

This pass validated three areas in the current implementation:

1. Whether documentation results actually appear in user-facing output.
2. Whether the current clustering/chunking behavior matches realistic cases.
3. Whether embedding usage is higher than necessary, especially for low-value or non-user-facing fields.

Validation used:

- Static inspection of indexing, query, formatting, and storage paths.
- Live indexing and query probes against `tests/fixtures/sample_project`.
- Synthetic probes for literal/tag retrieval and realistic function chunking cases.
- Full test run in a local `.venv`.

## Executive Summary

- Documentation does appear in user-facing output today, but mostly at document level rather than section level, and it can duplicate on case-insensitive filesystems.
- Dual-representation function search is not properly clustered with SCIP symbols. The same function can appear twice in results with different render styles.
- Embedding usage is materially higher than necessary. A large share of embedded text is boilerplate identity/signature prefix rather than retrieval-bearing content.
- Several metadata fields currently affect retrieval without being surfaced back to the user, which weakens result explainability.
- One real query-path bug remains: querying a hash-built index with `embedding_backend="auto"` can crash if `llama_cpp` is unavailable.

## Findings

### 1. Function result clustering is broken across SCIP and synthetic Python symbols

Severity: High

The Python dual-representation path creates synthetic function symbol IDs like `py:app:run:function`, while SCIP indexing creates separate symbol IDs for the same functions.

Relevant code:

- Function metadata symbol creation: `src/ultimate_indexer/function_indexer.py:364`
- Synthetic ranked symbol fallback: `src/ultimate_indexer/query.py:526`
- Grouping dedupe only by raw `symbol_id`: `src/ultimate_indexer/query.py:603`

Observed behavior:

- Query: `greeting service user`
  - `app.py::run` appeared twice:
    - SCIP symbol: `scip-python ... /run().`
    - Synthetic symbol: `py:app:run:function`
- Query: `tenants database schema`
  - Duplicates appeared for:
    - `serialize_user`
    - `run`
    - `build_demo_user`

Concrete example from live run:

- `app.py` grouped symbols included both:
  - `('run', 'Function', 'scip-python ... /run().')`
  - `('run', 'Function', 'py:app:run:function')`

Risk:

- User-facing output becomes noisy and inconsistent.
- Ranking is diluted across multiple IDs for the same logical function.
- Dual-representation boost is not being fully realized because metadata/body hits do not always merge onto the canonical code symbol.

Recommendation:

- Canonicalize function metadata/body records onto existing SCIP symbols before scoring.
- Primary join key should be:
  - `(relative_path, start_line, end_line, display_name)`
- Fallback join key:
  - `(relative_path, display_name, kind)`
- Group-level dedupe should be logical-symbol dedupe, not raw-ID dedupe.

### 2. Documentation does show up in user-facing output, but section visibility is weak

Severity: High

Documentation is present in search output today, but mostly as document-level matches. Section-level relevance is indexed, but not clearly preserved in the displayed symbol selection.

Relevant code:

- Doc chunk creation attaches all doc chunks to the doc root symbol:
  - `src/ultimate_indexer/docs/ingest.py:156`
- Section symbol chunks are skipped during general symbol chunk generation:
  - `src/ultimate_indexer/indexer.py:694`

Observed behavior:

- Query: `tenants database schema`
  - Output included `// docs/schema.md`
  - Rendered content clearly showed the schema markdown.
- Query: `Authentication joins users to tenants by tenant_id`
  - The correct doc file ranked first.
  - Output still rendered only a document-level block, not a targeted `Tables` or subsection-specific result.
- Query: `Tables`
  - Result still surfaced the whole document and artifact, not a section-specific user-facing block.

Concrete example:

- `docs/schema.md` rendered:
  - `# Database Schema`
  - `The sample project stores tenants and users in SQLite.`
  - `## Tables`
  - `- tenants(...)`

What is missing:

- The query path never clearly surfaces the section symbol as the winning displayed unit, even though section symbols exist in storage.

Risk:

- Users get correct files, but weaker locality and less precise explanation of why a doc matched.
- Broad document rendering increases clutter and hides exact relevant sections.

Recommendation:

- Attach documentation chunks to section symbols when a header/anchor exists.
- Keep doc-root symbol only for preambles and whole-document fallback.
- Stop excluding section symbols from chunk generation where section-level display matters.

### 3. Documentation can be duplicated as both `docs/...` and `Docs/...`

Severity: High

The documentation discovery path scans both `docs` and `Docs` and deduplicates by relative path string, not by realpath/inode or normalized path.

Relevant code:

- Candidate directories include both `docs` and `Docs`:
  - `src/ultimate_indexer/docs/ingest.py:77`
- Deduplication uses relative path string only:
  - `src/ultimate_indexer/docs/ingest.py:283`

Observed behavior on this machine:

- The same physical file was indexed twice:
  - `docs/schema.md`
  - `Docs/schema.md`

Concrete evidence:

- Stored doc symbols existed for both:
  - `doc::docs/schema.md`
  - `doc::Docs/schema.md`

Risk:

- Duplicate results in ranked output.
- Wasted embeddings and duplicate graph nodes.
- Behavior varies by filesystem case sensitivity.

Recommendation:

- Deduplicate by resolved realpath or inode.
- Alternatively normalize path case during discovery on case-insensitive platforms.

### 4. Embedding usage is materially higher than necessary

Severity: High

Function metadata and function body documents are always embedded, even though much of their embedded text is boilerplate.

Relevant code:

- Function metadata/body embeddings are always generated:
  - `src/ultimate_indexer/indexer.py:807`
- Metadata embedding text includes many prefixed fields:
  - `src/ultimate_indexer/function_indexer.py:74`
- Body embedding text includes identity and structural prefix:
  - `src/ultimate_indexer/function_indexer.py:145`

Measured on `tests/fixtures/sample_project`:

- `chunks_total = 39`
- `chunks_embedded = 17`
- `function_metadata_embedded = 6`
- `function_body_embedded = 6`

This means:

- 12 of 29 total embedded documents were function metadata/body docs.
- About 41% of embedding volume in the sample came from dual-representation function documents.

Measured metadata overhead:

- Total metadata chars: `1495`
- `signature:` chars: `761`
- `symbol:` + `kind:` chars: `281`
- Remaining chars: `453`

Interpretation:

- Around 70% of metadata embedding text was identity/signature boilerplate rather than higher-signal retrieval text.

Measured body overhead:

- Total body content chars: `1809`
- Actual body chars: `460`
- Prefix overhead chars: `1349`

Interpretation:

- Around 75% of body embedding text was prefix overhead, not code body.

Risk:

- More embedding cost than needed.
- Lower semantic density per embedded token.
- Larger indexes and slower reindexing.

Recommendation:

- Cache and reuse function metadata/body embeddings the same way regular chunks are cached.
- Reduce metadata text to the smallest useful retrieval shape.
- Strip body prefixes down to minimal tie-back context.

### 5. Low-value metadata fields influence retrieval without being shown back to the user

Severity: Medium

Function metadata full-text retrieval uses `metadata_content`, which includes `calls`, `types`, `tags`, and `literals`.

Relevant code:

- Function metadata search path:
  - `src/ultimate_indexer/storage.py:614`
- Metadata fields added to dense text:
  - `src/ultimate_indexer/function_indexer.py:92`

Synthetic validation:

Source included:

- `ORDER_WEBHOOK_TOKEN`
- `FEATURE_FLAG`
- `@lru_cache`
- API webhook docstring and URL

Observed queries:

- `ORDER_WEBHOOK_TOKEN`
  - Correct function `send_webhook` returned.
- `FEATURE_FLAG`
  - Correct function `load_flag` returned.
- `cache`
  - `load_flag` returned.

But rendered output did not expose the actual matching literal/tag:

- For `FEATURE_FLAG`, output showed only `@lru_cache(maxsize=64)` and a broad file section.
- For `ORDER_WEBHOOK_TOKEN`, output showed the function and file preview, not the literal that caused the match.

Risk:

- Retrieval appears “magical” or weakly justified.
- Low-value literals/tags can dominate relevance without improving what the user actually sees.

Recommendation:

- Move `called_functions`, `referenced_types`, `behavioral_tags`, and `literals` out of dense embeddings.
- Keep them for BM25-only retrieval or secondary reranking.
- If retained as retrieval features, surface matched fields in rendered output.

### 6. Logical body chunking works, but only for larger functions than many real cases

Severity: Medium

The split logic itself is reasonable, but the default `max_chunk_lines=50` means many realistic functions never get split.

Relevant code:

- Default body chunk threshold:
  - `src/ultimate_indexer/function_indexer.py:398`
- Split logic:
  - `src/ultimate_indexer/function_indexer.py:528`

Realistic case that did not split:

- `process_order(...)`
  - Included:
    - validation
    - loop
    - try/except
    - external webhook call
  - Result:
    - `CHUNKS 1`
    - chunk type: `main`

Realistic larger case that did split well:

- `large_pipeline(...)`
  - Split points found:
    - setup
    - validation
    - transform
    - try block
    - except block
    - notify
  - Result:
    - `DEFAULT_CHUNKS 6`
    - Types: `section`, `try_block`, `except_block`

Risk:

- Mid-sized functions get embedded as a single blob even when they have clear logical phases.
- Behavioral retrieval quality is weaker on common real-world functions in the 20-40 line range.

Recommendation:

- Lower the split trigger from 50 lines to a smaller threshold.
- Also split when multiple clear structural boundaries exist, even if total size is below the line limit.

### 7. Query path can crash on embedding backend mismatch

Severity: Medium

There is a real failure when querying an index built with `hash` using a second indexer configured with `embedding_backend="auto"` and no `llama_cpp` installed.

Relevant code:

- Provider resolution:
  - `src/ultimate_indexer/indexer.py:318`

Observed failing test:

- `tests/test_query.py::test_query_survives_embedding_backend_mismatch`

Observed failure:

- `ModuleNotFoundError: No module named 'llama_cpp'`

Impact:

- Query path is not robust to provider mismatch.
- Existing indexes can become unusable depending on local runtime state.

Recommendation:

- If dense query embedding cannot be created, degrade to sparse-only retrieval instead of crashing.
- Alternatively, skip dense paths when stored embeddings and active provider are incompatible.

## Concrete Examples

### Documentation showing up

Query:

- `tenants database schema`

User-facing output included:

- `// docs/schema.md`
- `# Database Schema`
- `The sample project stores tenants and users in SQLite.`
- `## Tables`

Conclusion:

- Documentation definitely reaches user-facing output today.

### Documentation duplication

Observed results included both:

- `docs/schema.md`
- `Docs/schema.md`

Conclusion:

- Same document can appear twice.

### Function duplication

Query:

- `greeting service user`

Observed duplicate logical symbol:

- `run` appeared twice in `app.py`

Conclusion:

- Function metadata/body hits are not clustered with canonical code symbols.

### Literal-driven retrieval without visible explanation

Synthetic queries:

- `ORDER_WEBHOOK_TOKEN`
- `FEATURE_FLAG`
- `cache`

Correct function was returned, but output did not clearly show the matching metadata field.

Conclusion:

- Retrieval is influenced by low-visibility fields.

## Embedding Reduction Opportunities

### Safe reductions

These are good candidates to remove from dense embeddings:

- `called_functions`
- `referenced_types`
- `behavioral_tags`
- `literals`
- body `symbol:` prefix
- body `chunk:` prefix
- body `type:` prefix

### Probably keep in dense metadata

- first meaningful docstring line
- short normalized signature
- parameter names
- return type
- function display name / FQN

### Notes on graph-only data

Graph structure already exists separately in storage and ranking:

- edges
- enclosing relationships
- pagerank

That graph information does not need to be repeated heavily inside embedded text unless it directly improves user-visible explanation.

## Recommended Thresholds

### Function body chunking

- Lower default body split threshold to `24-32` non-empty lines.
- Force splitting when `>= 2` structural boundaries exist:
  - section comment
  - `try`
  - `except`
  - significant loop

### Dense body embedding

- Skip dense body embedding for chunks with:
  - fewer than `4` non-empty lines, or
  - fewer than `120-160` chars after prefix removal

### Dense metadata embedding

- Skip dense metadata embedding when there is:
  - no docstring, and
  - no meaningful params/returns beyond a short signature

### Metadata fields

- Keep `literals`, `tags`, `calls`, and `types` sparse-only by default.
- Reintroduce them into dense text only if matched fields are rendered back to the user.

## Recommended Change List

1. Canonicalize Python metadata/body records onto SCIP symbols before scoring.
2. Change grouping dedupe from raw `symbol_id` to logical symbol identity.
3. Deduplicate documentation discovery by resolved path, not relative string alone.
4. Attach doc chunks to section symbols when possible.
5. Surface section-level documentation results in formatted output.
6. Add embedding cache/reuse for function metadata and body chunks.
7. Shrink metadata dense text to high-signal fields only.
8. Remove most body prefix boilerplate from dense embedding text.
9. Treat low-value metadata fields as sparse-only unless surfaced in output.
10. Make query fallback robust when dense embedding backend is unavailable.

## Test Status

Environment used for validation:

- Local `.venv`
- Package installed editable
- Runtime/test dependencies installed manually for this pass

Full test suite result during validation:

- Passed except:
  - `tests/test_query.py::test_query_survives_embedding_backend_mismatch`

This failing test reflects a real product issue, not just validation environment setup.

