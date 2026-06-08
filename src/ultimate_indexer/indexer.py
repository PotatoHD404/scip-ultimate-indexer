from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from .constants import MAX_FILE_BYTES, SPECIAL_FILES, is_indexable_filename
from .config import Settings
from .embeddings import (
    APIEmbeddingProvider,
    HashEmbeddingProvider,
    generate_embeddings,
    _provider_prepare_document_text,
    resolve_api_provider,
    resolve_llama_cpp_provider,
)
from .fallback import build_fallback_bundle
from .formatter import _pretty_signature, format_scored_tree
from .function_indexer import chunk_function_bodies, extract_function_metadata
from .ignore_rules import create_ignore_matcher
from .models import (
    ChunkRecord, EdgeRecord, FileRecord, FunctionBodyChunkRecord,
    FunctionMetadataRecord, IndexProgress, IndexSummary, SymbolRecord, TreeScoreNode,
)
from .query import QueryEngine, SearchScope, apply_kind_boost, dependency_ordered_pagerank
from .ranking_rules import is_external_symbol, is_queryable_symbol, is_rankable_symbol
from .scip_parser import ParsedScip, parse_scip_index
from .scip_runner import ScipRunFailure, StructuredIndexingRequiredError, run_scip_indexers
from .socraticode import ingest_socraticode_artifacts
from .storage import Storage
from .visuals import write_query_visualization
from .docs.ingest import _discover_document_files, ingest_documentation
from .expansions import expansion_text
from .contextual import build_context_header, contextual_embedding_text, first_doc_line
from . import git_signals as git_signals_module


INDEX_STAGES = (
    "discover",
    "scip",
    "artifacts",
    "docs",
    "fallback",
    "chunks",
    "embed",
    "store",
    "pagerank",
)
INDEX_FORMAT_VERSION = 7


def _discover_code_files(project_root: Path, extra_extensions: set[str]) -> list[Path]:
    matcher = create_ignore_matcher(project_root)
    files: list[Path] = []
    for current_root, dirs, filenames in os.walk(project_root):
        root_path = Path(current_root)
        rel_root = root_path.relative_to(project_root).as_posix() if root_path != project_root else ""
        kept_dirs: list[str] = []
        for dirname in dirs:
            rel_dir = f"{rel_root}/{dirname}".lstrip("/")
            if matcher.ignores(rel_dir):
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        for filename in filenames:
            if filename.startswith(".") and filename not in SPECIAL_FILES:
                continue
            full_path = root_path / filename
            rel_path = full_path.relative_to(project_root).as_posix()
            if matcher.ignores(rel_path):
                continue
            if not is_indexable_filename(filename, extra_extensions):
                continue
            try:
                if full_path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(full_path)
    return sorted(files)


def _project_signature(
    code_files: list[Path],
    doc_files: list[Path],
    project_root: Path,
    embedding_backend: str,
    model_path: str | None,
    model_filename: str,
    embedding_api_endpoint: str | None,
    embedding_api_model: str | None,
    extra_extensions: set[str],
    max_chunk_lines: int,
    chunk_overlap: int,
) -> tuple[str, str]:
    """Return (full_signature, config_signature).

    The full signature mixes per-file mtime/size with the config; it decides
    whether anything changed at all. The config signature covers only the
    non-file inputs (embedding backend/model, index format version, chunk params,
    ignore settings) so a config/format change forces a full rebuild even when no
    file content changed (otherwise stale-format or stale-model rows survive).
    """
    file_payload: list[str] = []
    for path in code_files:
        relative = path.relative_to(project_root)
        file_payload.append(f"{relative}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    for path in doc_files:
        relative = path.relative_to(project_root)
        file_payload.append(f"doc::{relative}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    config_payload: list[str] = []
    config_payload.append(f"embedding-backend:{embedding_backend}")
    config_payload.append(f"index-format-version:{INDEX_FORMAT_VERSION}")
    config_payload.append(f"model-path:{model_path or ''}")
    config_payload.append(f"model-file:{model_filename}")
    config_payload.append(f"embedding-api-endpoint:{embedding_api_endpoint or ''}")
    config_payload.append(f"embedding-api-model:{embedding_api_model or ''}")
    config_payload.append(f"extra-extensions:{','.join(sorted(extra_extensions))}")
    config_payload.append(f"respect-gitignore:{os.getenv('RESPECT_GITIGNORE', 'true')}")
    config_payload.append(f"ignore-dirs:{os.getenv('IGNORE_DIRS', '')}")
    config_payload.append(f"scip-languages:{os.getenv('SCIP_LANGUAGES', '')}")
    config_payload.append(f"max-chunk-lines:{max_chunk_lines}")
    config_payload.append(f"chunk-overlap:{chunk_overlap}")
    for special_name in (
        ".socraticodecontextartifacts.json",
        ".gitignore",
        ".socraticodeignore",
        ".cgcignore",
    ):
        candidate = project_root / special_name
        if candidate.exists():
            config_payload.append(f"{special_name}:{candidate.stat().st_mtime_ns}:{candidate.stat().st_size}")
    config_signature = sha256("\n".join(config_payload).encode("utf-8")).hexdigest()
    full_signature = sha256("\n".join(file_payload + config_payload).encode("utf-8")).hexdigest()
    return full_signature, config_signature


def _overview_chunk(project_id: str, file_symbol: SymbolRecord, symbols: list[SymbolRecord]) -> ChunkRecord:
    signatures = [symbol.signature for symbol in symbols[:20] if symbol.kind != "File"]
    content = "\n".join(
        item
        for item in [
            file_symbol.relative_path,
            file_symbol.docstring,
            *signatures,
        ]
        if item
    )
    return ChunkRecord(
        project_id=project_id,
        chunk_id=sha256(f"{file_symbol.relative_path}:file-overview".encode("utf-8")).hexdigest()[:32],
        relative_path=file_symbol.relative_path,
        symbol_id=file_symbol.symbol_id,
        symbol_name=file_symbol.display_name,
        artifact_name=None,
        chunk_kind="file-overview",
        start_line=1,
        end_line=min(file_symbol.end_line, 40),
        content=content,
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
    )


def _render_scip_warning(failure: ScipRunFailure, project_root: Path) -> str:
    try:
        rel_root = Path(failure.working_directory).resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        rel_root = failure.working_directory
    detail = " ".join(failure.detail.split())
    if len(detail) > 240:
        detail = detail[:240].rstrip() + " […]"
    return f"{failure.language} at {rel_root}: {detail}"


def _dedupe_parsed_scip(parsed: ParsedScip) -> ParsedScip:
    file_by_path: dict[str, FileRecord] = {}
    for record in parsed.files:
        file_by_path.setdefault(record.relative_path, record)

    symbol_by_id: dict[str, SymbolRecord] = {}
    for record in parsed.symbols:
        symbol_by_id.setdefault(record.symbol_id, record)

    unique_edges: list[EdgeRecord] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for edge in parsed.edges:
        edge_key = (edge.source_symbol_id, edge.target_symbol_id, edge.edge_type)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        unique_edges.append(edge)

    return ParsedScip(
        files=list(file_by_path.values()),
        symbols=list(symbol_by_id.values()),
        edges=unique_edges,
    )


def _first_meaningful_line(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("```"):
            return line
    return ""


def _summarize_text(text: str, limit: int = 240) -> str:
    compact = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _symbol_heading(
    symbol: SymbolRecord,
    symbol_lookup: dict[str, SymbolRecord],
) -> str:
    enclosing_display = ""
    if symbol.enclosing_symbol_id:
        parent = symbol_lookup.get(symbol.enclosing_symbol_id)
        if parent is not None and parent.kind not in {"File", "Module"}:
            enclosing_display = parent.display_name
    if enclosing_display:
        return f"{symbol.kind.lower()}: {enclosing_display}.{symbol.display_name}"
    return f"{symbol.kind.lower()}: {symbol.display_name}"


def _child_lines(
    symbol: SymbolRecord,
    child_symbols: list[SymbolRecord],
) -> list[str]:
    if not child_symbols:
        return []
    lines: list[str] = []
    for child in child_symbols[:30]:
        signature = _pretty_signature(
            kind=child.kind,
            display_name=child.display_name,
            signature=child.signature,
            docstring=child.docstring,
            snippet=child.snippet,
        ).strip()
        if not signature:
            continue
        line = f"  {signature}"
        summary = _summarize_text(_first_meaningful_line(child.docstring), limit=100)
        if summary and summary != signature:
            line += f"  // {summary}"
        lines.append(line)
    if len(child_symbols) > 30:
        lines.append(f"  ... and {len(child_symbols) - 30} more")
    return lines


def _symbol_chunk_content(
    symbol: SymbolRecord,
    *,
    symbol_lookup: dict[str, SymbolRecord],
    children_by_symbol_id: dict[str, list[SymbolRecord]],
) -> str:
    pretty_signature = _pretty_signature(
        kind=symbol.kind,
        display_name=symbol.display_name,
        signature=symbol.signature,
        docstring=symbol.docstring,
        snippet=symbol.snippet,
    ).strip()
    parts = [
        _symbol_heading(symbol, symbol_lookup),
        f"file: {symbol.relative_path}",
    ]
    if pretty_signature:
        parts.append(f"signature:\n{pretty_signature}")
    summary = _summarize_text(symbol.docstring, limit=500)
    if summary:
        parts.append(f"documentation: {summary}")
    child_lines = _child_lines(symbol, children_by_symbol_id.get(symbol.symbol_id, []))
    if child_lines:
        label = "fields and methods" if symbol.kind in {"Class", "Struct", "Interface", "TypeAlias", "Enum", "Trait"} else "members"
        parts.append(f"{label}:\n" + "\n".join(child_lines))
    if symbol.snippet.strip():
        snippet = symbol.snippet.strip()
        if len(snippet) > 800:
            snippet = snippet[:800].rstrip() + "\n..."
        parts.append(f"code:\n{snippet}")
    return "\n".join(part for part in parts if part)


def _file_usefulness_score(rank_sum: float, rank_max: float, useful_symbol_count: int, chunk_count: int) -> float:
    symbol_bonus = 0.005 * min(max(useful_symbol_count, 0), 10)
    chunk_bonus = 0.001 * min(max(chunk_count, 0), 10)
    return max(0.0, rank_sum) + max(0.0, rank_max) * 0.35 + symbol_bonus + chunk_bonus


def _normalize_tree_scores(node: TreeScoreNode, root_score: float) -> None:
    if root_score <= 0:
        node.score = 0.0
    else:
        node.score = (node.raw_score / root_score) * 100.0
    for child in node.children:
        _normalize_tree_scores(child, root_score)


def _normalized_kind(kind: str) -> str:
    return kind.replace("_", "").replace("-", "").lower()


class UltimateIndexer:
    def __init__(
        self,
        project_root: Path,
        embedding_backend: str | None = None,
        *,
        state_dir: Path | None = None,
        cache_base_dir: Path | None = None,
    ) -> None:
        resolved_root = project_root.resolve()
        resolved_state_dir = state_dir
        if resolved_state_dir is None and cache_base_dir is not None:
            project_hash = sha256(str(resolved_root).encode("utf-8")).hexdigest()[:12]
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved_root.name).strip("-") or "project"
            resolved_state_dir = cache_base_dir.resolve() / f"{safe_name}-{project_hash}"
        self.settings = Settings(project_root=resolved_root, state_dir_override=resolved_state_dir)
        if embedding_backend is not None:
            self.settings.embedding_backend = embedding_backend
        self.settings.ensure_directories()
        self.project_id = str(self.settings.project_root)
        self.storage = Storage(self.settings.database_path)
        self._provider = None
        self._query_engine: QueryEngine | None = None
        # Timestamp of the last file-staleness check; 0.0 = never checked.
        self._last_stale_check: float = 0.0
        # Git-history signals from the most recent index (recency/churn/co-change).
        self._git_signals: git_signals_module.GitSignals | None = None

    def close(self) -> None:
        self.storage.close()

    def _provider_instance(self):
        if self._provider is not None:
            return self._provider
        backend = self.settings.embedding_backend
        if backend == "hash":
            self._provider = HashEmbeddingProvider()
            return self._provider
        # Check if API embedding is configured
        if backend == "api" or (self.settings.embedding_api_endpoint and self.settings.embedding_api_model):
            self._provider = resolve_api_provider(
                api_endpoint=self.settings.embedding_api_endpoint,
                api_model=self.settings.embedding_api_model,
                api_key=self.settings.embedding_api_key,
                batch_size=self.settings.embedding_api_batch_size,
                timeout_seconds=self.settings.embedding_api_timeout_seconds,
                max_tokens=self.settings.embedding_api_max_tokens,
                max_retries=self.settings.embedding_api_max_retries,
                retry_base_delay_ms=self.settings.embedding_api_retry_base_delay_ms,
            )
            return self._provider
        # llama-cpp path. The native provider imports ``llama_cpp`` lazily at
        # embed time, so a present-but-unusable model file (e.g. a cached .gguf
        # with the runtime uninstalled) would otherwise crash mid-index. Verify
        # importability up front: degrade "auto" to hash, but fail loudly when
        # the backend was explicitly requested.
        explicit_llama = backend in {"llama-cpp", "local"}
        if importlib.util.find_spec("llama_cpp") is None:
            if explicit_llama:
                raise RuntimeError(
                    f"Embedding backend {backend!r} requires the 'llama-cpp-python' "
                    "package, which is not installed. Install it, or select another "
                    "backend (ULTIMATE_INDEXER_EMBEDDING_BACKEND=hash, or configure an "
                    "embedding API)."
                )
            self._provider = HashEmbeddingProvider()
            return self._provider
        try:
            self._provider = resolve_llama_cpp_provider(
                project_root=self.settings.project_root,
                model_cache_dir=self.settings.model_cache_dir,
                model_path=self.settings.model_path,
                filename=self.settings.model_filename,
            )
            return self._provider
        except Exception:
            if explicit_llama:
                raise
            self._provider = HashEmbeddingProvider()
            return self._provider

    def _emit_progress(
        self,
        callback: Callable[[IndexProgress], None] | None,
        stage: str,
        completed: int = 0,
        total: int = 0,
        unit: str = "items",
        detail: str = "",
    ) -> None:
        if callback is None:
            return
        try:
            stage_index = INDEX_STAGES.index(stage) + 1
        except ValueError:
            stage_index = len(INDEX_STAGES)
        callback(
            IndexProgress(
                stage=stage,
                stage_index=stage_index,
                stage_total=len(INDEX_STAGES),
                completed=completed,
                total=total,
                unit=unit,
                detail=detail,
            )
        )

    def _embed_chunks(
        self,
        chunks: list[ChunkRecord],
        symbol_lookup: dict[str, SymbolRecord],
        progress_callback: Callable[[IndexProgress], None] | None = None,
    ) -> list[ChunkRecord]:
        provider = self._provider_instance()
        existing_chunk_embeddings = self.storage.get_chunk_embeddings(self.project_id)
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        cached_count = 0
        skipped_count = 0
        for index, chunk in enumerate(chunks):
            # Skip embedding for non-queryable symbols (but keep them in the graph)
            if chunk.symbol_id and chunk.symbol_id in symbol_lookup:
                symbol = symbol_lookup[chunk.symbol_id]
                if not is_queryable_symbol(symbol.relative_path, symbol.kind):
                    skipped_count += 1
                    continue
            existing = existing_chunk_embeddings.get(chunk.chunk_id)
            if (
                existing is not None
                and str(existing["content_hash"]) == chunk.content_hash
                and str(existing["embedding_model_id"]) == provider.model_id
            ):
                chunk.embedding = bytes(existing["embedding"])
                chunk.embedding_dim = int(existing["embedding_dim"])
                chunk.embedding_model_id = provider.model_id
                cached_count += 1
                continue
            embedding_text = _provider_prepare_document_text(
                provider,
                contextual_embedding_text(chunk.content, chunk.context_header),
                chunk.relative_path,
            )
            cached = self.storage.get_or_create_embedding(provider.model_id, embedding_text)
            if cached is not None:
                chunk.embedding = cached.astype("float32").tobytes()
                chunk.embedding_dim = int(cached.shape[0])
                chunk.embedding_model_id = provider.model_id
                cached_count += 1
                continue
            pending_indices.append(index)
            pending_texts.append(embedding_text)
        if pending_texts:
            self._emit_progress(
                progress_callback,
                stage="embed",
                completed=0,
                total=1,
                unit="model",
                detail="Loading embedding model",
            )
        embeddable_chunks = len(chunks) - skipped_count
        self._emit_progress(
            progress_callback,
            stage="embed",
            completed=cached_count,
            total=max(embeddable_chunks, 1),
            unit="chunks",
            detail=f"Embedding chunks (skipped {skipped_count} non-queryable)",
        )
        if pending_texts:
            vectors = generate_embeddings(
                provider,
                pending_texts,
                on_batch_complete=lambda processed, total: self._emit_progress(
                    progress_callback,
                    stage="embed",
                    completed=cached_count + processed,
                    total=max(embeddable_chunks, 1),
                    unit="chunks",
                    detail=f"Embedding chunks (skipped {skipped_count} non-queryable)",
                ),
            )
            for list_index, vector in enumerate(vectors):
                chunk = chunks[pending_indices[list_index]]
                embedding_text = pending_texts[list_index]
                self.storage.store_embedding(provider.model_id, embedding_text, vector)
                chunk.embedding = vector.astype("float32").tobytes()
                chunk.embedding_dim = int(vector.shape[0])
                chunk.embedding_model_id = provider.model_id
        self._emit_progress(
            progress_callback,
            stage="embed",
            completed=max(embeddable_chunks, 1),
            total=max(embeddable_chunks, 1),
            unit="chunks",
            detail=f"Embedded chunks (skipped {skipped_count} non-queryable)",
        )
        return chunks

    def _global_ranks(self) -> None:
        symbol_rows = self.storage.get_symbol_rows(self.project_id)
        if not symbol_rows:
            return
        ranks = dependency_ordered_pagerank(
            symbol_rows,
            self.storage.get_edges(self.project_id),
            alpha=0.85,
        )
        signals = getattr(self, "_git_signals", None)
        strength = self.settings.git_signal_strength if self.settings.enable_git_signals else 0.0

        def _git_multiplier(symbol_id: str) -> float:
            # Recently-changed / frequently-churned code is more likely to matter;
            # lift its rank by up to `strength` (1.0 = no change).
            if signals is None or strength <= 0:
                return 1.0
            relpath = str(symbol_rows[symbol_id]["relative_path"])
            boost = git_signals_module.recency_churn_boost(
                signals,
                relpath,
                recency_weight=self.settings.git_recency_weight,
                churn_weight=self.settings.git_churn_weight,
            )
            return 1.0 + strength * boost

        boosted = {
            symbol_id: apply_kind_boost(symbol_rows[symbol_id], score) * _git_multiplier(symbol_id)
            for symbol_id, score in ranks.items()
            if symbol_id in symbol_rows
        }
        total = sum(boosted.values())
        if total > 0:
            boosted = {symbol_id: score / total for symbol_id, score in boosted.items()}
        else:
            boosted = {}
        self.storage.set_global_ranks(self.project_id, boosted)

    def refresh_if_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Re-index incrementally if any project files changed since last index.

        Rate-limited: the staleness check runs at most once per
        *max_age_seconds*.  When nothing has changed the call returns in
        milliseconds — ``index()`` detects matching mtime+size signatures and
        exits without reading any file contents.  When files were modified,
        added, or removed, an incremental re-index runs automatically.

        Returns ``True`` if a re-index was triggered, ``False`` otherwise.

        Called automatically by the MCP server before every query tool so the
        index stays fresh while the server is long-running.
        """
        now = time.monotonic()
        if now - self._last_stale_check < max_age_seconds:
            return False
        self._last_stale_check = now

        # Only auto-refresh if the project has been indexed at least once.
        if not self.storage.get_file_hashes(self.project_id):
            return False

        # index() computes _project_signature() from st_mtime_ns + st_size
        # (no file reads).  If nothing changed the signature matches the stored
        # one and index() returns immediately.  If it differs, incremental
        # processing kicks in automatically.
        try:
            self.index(progress_callback=None)
            return True
        except Exception:
            # Never let a background refresh crash a query.
            return False

    def index(
        self,
        scip_path: Path | None = None,
        force: bool = False,
        progress_callback: Callable[[IndexProgress], None] | None = None,
    ) -> IndexSummary:
        scip_warnings: list[str] = []
        self._emit_progress(progress_callback, stage="discover", detail="Scanning project files")
        code_files = _discover_code_files(self.settings.project_root, self.settings.extra_extensions)
        doc_files_for_signature = _discover_document_files(self.settings.project_root)
        self._emit_progress(
            progress_callback,
            stage="discover",
            completed=max(len(code_files), 1),
            total=max(len(code_files), 1),
            unit="files",
            detail="Discovered indexable files",
        )
        signature, config_signature = _project_signature(
            code_files,
            doc_files_for_signature,
            self.settings.project_root,
            self.settings.embedding_backend,
            self.settings.model_path,
            self.settings.model_filename,
            self.settings.embedding_api_endpoint,
            self.settings.embedding_api_model,
            self.settings.extra_extensions,
            self.settings.max_chunk_lines,
            self.settings.chunk_overlap,
        )
        if not force and self.storage.get_project_signature(self.project_id) == signature:
            self._emit_progress(
                progress_callback,
                stage="pagerank",
                completed=1,
                total=1,
                unit="status",
                detail="Index is up to date; reusing cached data",
            )
            top_count = len(self.storage.get_symbol_rows(self.project_id))
            return IndexSummary(
                project_id=self.project_id,
                indexed_files=len(self.storage.get_file_hashes(self.project_id)),
                indexed_symbols=top_count,
                indexed_edges=len(self.storage.get_edges(self.project_id)),
                indexed_chunks=len(self.storage.load_chunk_vectors(self.project_id)[1]),
                reused_files=len(code_files),
                artifact_files=0,
                warnings=[],
            )

        parsed_files: list[FileRecord] = []
        parsed_symbols: list[SymbolRecord] = []
        parsed_edges: list[EdgeRecord] = []
        self._emit_progress(progress_callback, stage="scip", detail="Collecting SCIP data")
        if scip_path is not None:
            parsed = parse_scip_index(
                project_id=self.project_id,
                project_root=self.settings.project_root,
                index_path=scip_path,
                edge_weights=self.settings.edge_weights,
            )
            parsed_files.extend(parsed.files)
            parsed_symbols.extend(parsed.symbols)
            parsed_edges.extend(parsed.edges)
            self._emit_progress(
                progress_callback,
                stage="scip",
                completed=1,
                total=1,
                unit="indexes",
                detail=f"Parsed SCIP index {scip_path.name}",
            )
        else:
            timeout_seconds = int(os.getenv("ULTIMATE_INDEXER_SCIP_TIMEOUT", "600"))
            scip_report = run_scip_indexers(
                self.settings.project_root,
                code_files,
                self.settings.cache_dir,
                timeout_seconds=timeout_seconds,
            )
            if scip_report.missing:
                raise StructuredIndexingRequiredError(scip_report.missing, [])
            scip_warnings = [
                _render_scip_warning(failure, self.settings.project_root)
                for failure in scip_report.failed
            ]
            scip_results = scip_report.results
            total_scip_steps = len(scip_results)
            completed_scip_steps = 0
            if total_scip_steps == 0:
                self._emit_progress(
                    progress_callback,
                    stage="scip",
                    completed=1,
                    total=1,
                    unit="indexes",
                    detail="No external SCIP indexes available" if not scip_warnings else "SCIP failed; using fallback coverage",
                )
            for result in scip_results:
                parsed = parse_scip_index(
                    project_id=self.project_id,
                    project_root=self.settings.project_root,
                    index_path=result.index_path,
                    edge_weights=self.settings.edge_weights,
                    source_root=result.source_root,
                )
                parsed_files.extend(parsed.files)
                parsed_symbols.extend(parsed.symbols)
                parsed_edges.extend(parsed.edges)
                completed_scip_steps += 1
                self._emit_progress(
                    progress_callback,
                    stage="scip",
                    completed=completed_scip_steps,
                    total=total_scip_steps,
                    unit="indexes",
                    detail=f"Parsed {result.language} SCIP",
                )
        parsed = _dedupe_parsed_scip(
            ParsedScip(files=parsed_files, symbols=parsed_symbols, edges=parsed_edges)
        )

        self._emit_progress(progress_callback, stage="artifacts", detail="Loading SocratiCode artifacts")
        artifact_bundle = ingest_socraticode_artifacts(self.project_id, self.settings.project_root)
        self._emit_progress(
            progress_callback,
            stage="artifacts",
            completed=max(len(artifact_bundle.files), 1),
            total=max(len(artifact_bundle.files), 1),
            unit="artifacts",
            detail="Loaded artifact files",
        )
        
        # Ingest documentation files (Markdown and OpenAPI) BEFORE fallback
        # This ensures documentation files are excluded from fallback processing
        self._emit_progress(progress_callback, stage="docs", detail="Ingesting documentation files")
        doc_files, doc_symbols, doc_edges, doc_chunks = ingest_documentation(
            project_id=self.project_id,
            project_root=self.settings.project_root,
            progress_callback=lambda completed, total, rel_path: self._emit_progress(
                progress_callback,
                stage="docs",
                completed=completed,
                total=total,
                unit="docs",
                detail=f"Doc: {rel_path}",
            ),
        )
        self._emit_progress(
            progress_callback,
            stage="docs",
            completed=max(len(doc_files), 1),
            total=max(len(doc_files), 1),
            unit="docs",
            detail="Ingested documentation files",
        )
        
        # Mark documentation files as covered so fallback doesn't process them
        covered_paths = {record.relative_path for record in parsed.files}
        covered_paths.update(record.relative_path for record in artifact_bundle.files)
        covered_paths.update(record.relative_path for record in doc_files)
        self._emit_progress(progress_callback, stage="fallback", detail="Building fallback index coverage")
        fallback_bundle = build_fallback_bundle(
            project_id=self.project_id,
            project_root=self.settings.project_root,
            files=code_files,
            covered_paths=covered_paths,
            contains_weight=self.settings.edge_weights["contains"],
            import_weight=self.settings.edge_weights["imports"],
            max_chunk_lines=self.settings.max_chunk_lines,
            chunk_overlap=self.settings.chunk_overlap,
            progress_callback=lambda completed, total, relative_path: self._emit_progress(
                progress_callback,
                stage="fallback",
                completed=completed,
                total=total,
                unit="files",
                detail=f"Fallback: {relative_path}",
            ),
        )
        if not fallback_bundle.files:
            self._emit_progress(
                progress_callback,
                stage="fallback",
                completed=1,
                total=1,
                unit="files",
                detail="No fallback files needed",
            )
        files: list[FileRecord] = [*parsed.files, *fallback_bundle.files, *artifact_bundle.files, *doc_files]
        symbols: list[SymbolRecord] = [*parsed.symbols, *fallback_bundle.symbols, *artifact_bundle.symbols, *doc_symbols]
        edges = [*parsed.edges, *fallback_bundle.edges, *artifact_bundle.edges, *doc_edges]

        symbols_by_file: dict[str, list[SymbolRecord]] = {}
        symbol_lookup: dict[str, SymbolRecord] = {}
        children_by_symbol_id: dict[str, list[SymbolRecord]] = {}
        for symbol in symbols:
            symbols_by_file.setdefault(symbol.relative_path, []).append(symbol)
            symbol_lookup[symbol.symbol_id] = symbol
            if symbol.enclosing_symbol_id:
                children_by_symbol_id.setdefault(symbol.enclosing_symbol_id, []).append(symbol)

        chunks: list[ChunkRecord] = [*artifact_bundle.chunks, *fallback_bundle.chunks, *doc_chunks]
        self._emit_progress(progress_callback, stage="chunks", detail="Assembling query chunks")
        for file_symbols in symbols_by_file.values():
            file_symbols.sort(key=lambda item: (item.start_line, item.display_name))
            if file_symbols and file_symbols[0].source_kind == "artifact":
                continue
            file_symbol = next((item for item in file_symbols if item.kind == "File"), None)
            if file_symbol is not None:
                chunks.append(_overview_chunk(self.project_id, file_symbol, file_symbols))
            for symbol in file_symbols:
                if symbol.kind in {"File", "Module", "Section"}:
                    continue
                chunk_content = _symbol_chunk_content(
                    symbol,
                    symbol_lookup=symbol_lookup,
                    children_by_symbol_id=children_by_symbol_id,
                )
                context_header = ""
                if self.settings.enable_contextual_embeddings:
                    enclosing = symbol_lookup.get(symbol.enclosing_symbol_id or "")
                    context_header = build_context_header(
                        relative_path=symbol.relative_path,
                        kind=symbol.kind,
                        enclosing_name=enclosing.display_name if enclosing is not None else "",
                        purpose=first_doc_line(symbol.docstring),
                    )
                # Fold the context header into content_hash so a contextual-setting
                # change re-embeds (the embedding text now differs even though the
                # raw code content does not).
                content_hash = sha256(
                    (chunk_content + "\x00" + context_header).encode("utf-8")
                ).hexdigest()
                chunks.append(
                    ChunkRecord(
                        project_id=self.project_id,
                        chunk_id=sha256(
                            f"{symbol.relative_path}:{symbol.symbol_id}:{symbol.start_line}:{symbol.end_line}".encode("utf-8")
                        ).hexdigest()[:32],
                        relative_path=symbol.relative_path,
                        symbol_id=symbol.symbol_id,
                        symbol_name=symbol.display_name,
                        artifact_name=None,
                        chunk_kind="symbol",
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        content=chunk_content,
                        content_hash=content_hash,
                        context_header=context_header,
                    )
                )

        self._emit_progress(
            progress_callback,
            stage="chunks",
            completed=max(len(chunks), 1),
            total=max(len(chunks), 1),
            unit="chunks",
            detail="Prepared chunks",
        )

        # Extract function metadata and body chunks from Python files
        function_metadata_records: list[FunctionMetadataRecord] = []
        function_body_records: list[FunctionBodyChunkRecord] = []
        self._emit_progress(progress_callback, stage="chunks", detail="Extracting function metadata")
        
        for file_record in files:
            if not file_record.relative_path.endswith(".py"):
                continue
            try:
                source = file_record.content
                metadata_list = extract_function_metadata(file_record.relative_path, source)
                
                for meta in metadata_list:
                    # Create metadata record
                    metadata_content = meta.to_index_text()
                    meta_record = FunctionMetadataRecord(
                        project_id=self.project_id,
                        symbol_id=meta.symbol_id,
                        relative_path=meta.relative_path,
                        display_name=meta.display_name,
                        kind=meta.kind,
                        fully_qualified_name=meta.fully_qualified_name,
                        signature=meta.signature,
                        normalized_signature=meta.normalized_signature,
                        docstring=meta.docstring,
                        params=json.dumps(meta.params),
                        param_types=json.dumps(meta.param_types),
                        return_type=meta.return_type,
                        decorators=json.dumps(meta.decorators),
                        referenced_types=json.dumps(meta.referenced_types),
                        called_functions=json.dumps(meta.called_functions),
                        raised_exceptions=json.dumps(meta.raised_exceptions),
                        literals=json.dumps(meta.literals),
                        behavioral_tags=json.dumps(meta.behavioral_tags),
                        start_line=meta.start_line,
                        end_line=meta.end_line,
                        metadata_content=metadata_content,
                        content_hash=sha256(metadata_content.encode("utf-8")).hexdigest(),
                    )
                    function_metadata_records.append(meta_record)
                
                # Create body chunk records
                body_chunks = chunk_function_bodies(metadata_list, source)
                for body_chunk in body_chunks:
                    body_content = body_chunk.to_index_text()
                    body_record = FunctionBodyChunkRecord(
                        project_id=self.project_id,
                        chunk_id=sha256(
                            f"{body_chunk.symbol_id}:{body_chunk.chunk_index}".encode("utf-8")
                        ).hexdigest()[:32],
                        symbol_id=body_chunk.symbol_id,
                        relative_path=body_chunk.relative_path,
                        display_name=body_chunk.display_name,
                        kind=body_chunk.kind,
                        signature=body_chunk.signature,
                        chunk_index=body_chunk.chunk_index,
                        total_chunks=body_chunk.total_chunks,
                        body=body_chunk.body,
                        chunk_type=body_chunk.chunk_type,
                        start_line=body_chunk.start_line,
                        end_line=body_chunk.end_line,
                        content=body_content,
                        content_hash=sha256(body_content.encode("utf-8")).hexdigest(),
                    )
                    function_body_records.append(body_record)
            except Exception:
                # Skip files that can't be parsed
                continue
        
        self._emit_progress(
            progress_callback,
            stage="chunks",
            completed=max(len(function_metadata_records) + len(function_body_records), 1),
            total=max(len(function_metadata_records) + len(function_body_records), 1),
            unit="function_docs",
            detail="Extracted function metadata and body chunks",
        )

        # Embed function metadata and body chunks
        all_function_docs: list[tuple[str, FunctionMetadataRecord | FunctionBodyChunkRecord]] = []
        for meta in function_metadata_records:
            all_function_docs.append(("metadata", meta))
        for body in function_body_records:
            all_function_docs.append(("body", body))
        
        if all_function_docs:
            self._emit_progress(progress_callback, stage="embed", detail="Embedding function documents")
            provider = self._provider_instance()

            # Reuse cached vectors (keyed on model + embedded text) so re-indexing
            # only embeds new/changed function docs instead of every function in
            # the repo on every run. The embedded text is unchanged (raw
            # metadata/body content) so retrieval behavior is identical.
            pending_indices: list[int] = []
            pending_texts: list[str] = []
            reused_count = 0
            for i, (_doc_type, doc) in enumerate(all_function_docs):
                source_text = doc.metadata_content if isinstance(doc, FunctionMetadataRecord) else doc.content
                if self.settings.enable_contextual_embeddings:
                    if isinstance(doc, FunctionMetadataRecord):
                        header = build_context_header(
                            relative_path=doc.relative_path,
                            kind=doc.kind,
                            enclosing_name=doc.fully_qualified_name.rsplit(".", 1)[0],
                            purpose=first_doc_line(doc.docstring),
                        )
                    else:
                        header = build_context_header(
                            relative_path=doc.relative_path,
                            kind=doc.kind,
                            enclosing_name=doc.display_name,
                        )
                    source_text = contextual_embedding_text(source_text, header)
                cached = self.storage.get_or_create_embedding(provider.model_id, source_text)
                if cached is not None:
                    doc.embedding = cached.astype("float32").tobytes()
                    doc.embedding_dim = int(cached.shape[0])
                    doc.embedding_model_id = provider.model_id
                    reused_count += 1
                    continue
                pending_indices.append(i)
                pending_texts.append(source_text)

            if pending_texts:
                vectors = generate_embeddings(provider, pending_texts)
                for list_index, vector in enumerate(vectors):
                    _doc_type, doc = all_function_docs[pending_indices[list_index]]
                    self.storage.store_embedding(provider.model_id, pending_texts[list_index], vector)
                    doc.embedding = vector.astype("float32").tobytes()
                    doc.embedding_dim = int(vector.shape[0])
                    doc.embedding_model_id = provider.model_id

            self._emit_progress(
                progress_callback,
                stage="embed",
                completed=len(all_function_docs),
                total=len(all_function_docs),
                unit="function_docs",
                detail=f"Embedded function documents (reused {reused_count} cached)",
            )

        # Vocabulary expansion: bridge code<->query terms for the lexical index
        # (e.g. a query for "authentication" matching a symbol named authnHandler).
        # Indexed into FTS only; never displayed.
        if self.settings.enable_query_expansion:
            for chunk in chunks:
                chunk.fts_expansion = expansion_text(chunk.symbol_name, path=chunk.relative_path)
            for meta_record in function_metadata_records:
                meta_record.fts_expansion = expansion_text(
                    meta_record.display_name,
                    signature=meta_record.signature,
                    path=meta_record.relative_path,
                )

        self._embed_chunks(chunks, symbol_lookup, progress_callback=progress_callback)
        self._emit_progress(progress_callback, stage="store", detail="Writing index to SQLite")
        existing_file_hashes = self.storage.get_file_hashes(self.project_id)
        indexed_file_hashes = {record.relative_path: record.content_hash for record in files}
        changed_paths = {
            relative_path
            for relative_path, content_hash in indexed_file_hashes.items()
            if existing_file_hashes.get(relative_path) != content_hash
        }
        removed_paths = set(existing_file_hashes) - set(indexed_file_hashes)
        # A config/format change (embedding backend/model, INDEX_FORMAT_VERSION,
        # chunk params, ignore settings) must trigger a FULL rebuild even when no
        # file content changed; otherwise unchanged files keep stale-format or
        # stale-model rows (which the model-id-filtered dense loaders then drop).
        if (
            existing_file_hashes
            and self.storage.get_project_config_signature(self.project_id) != config_signature
        ):
            force = True
        if not force and existing_file_hashes:
            self._emit_progress(
                progress_callback,
                stage="store",
                completed=0,
                total=1,
                unit="delta",
                detail=f"Incremental update: {len(changed_paths)} changed, {len(removed_paths)} removed",
            )

        if force or not existing_file_hashes:
            self.storage.replace_project_contents(
                self.project_id,
                files,
                symbols,
                edges,
                chunks,
                function_metadata=function_metadata_records,
                function_body_chunks=function_body_records,
            )
            reused_files = 0
        else:
            self.storage.replace_project_contents(
                self.project_id,
                files,
                symbols,
                edges,
                chunks,
                function_metadata=function_metadata_records,
                function_body_chunks=function_body_records,
                changed_paths=changed_paths,
                removed_paths=removed_paths,
            )
            reused_files = max(len(indexed_file_hashes) - len(changed_paths), 0)
        self.storage.upsert_project(self.project_id, self.project_id, signature, config_signature)
        self._emit_progress(
            progress_callback,
            stage="store",
            completed=1,
            total=1,
            unit="projects",
            detail="Stored indexed data",
        )
        # Git-history signals: recency/churn lift importance (folded into global
        # ranks), and co-change couples files for context-personalized ranking.
        # collect_git_signals never raises — it degrades to empty off a git repo.
        self._git_signals = git_signals_module.GitSignals.empty()
        if self.settings.enable_git_signals:
            self._git_signals = git_signals_module.collect_git_signals(
                self.settings.project_root,
                history_limit=self.settings.git_history_limit,
                half_life_days=self.settings.git_half_life_days,
            )
            indexed_paths = {record.relative_path for record in files}
            cochange = {
                pair: weight
                for pair, weight in self._git_signals.cochange.items()
                if pair[0] in indexed_paths and pair[1] in indexed_paths
            }
            self.storage.replace_cochange(self.project_id, cochange)
        else:
            self.storage.replace_cochange(self.project_id, {})

        self._emit_progress(progress_callback, stage="pagerank", detail="Computing global ranks")
        self._global_ranks()
        self._emit_progress(
            progress_callback,
            stage="pagerank",
            completed=1,
            total=1,
            unit="projects",
            detail="Finished indexing",
        )
        return IndexSummary(
            project_id=self.project_id,
            indexed_files=len(files),
            indexed_symbols=len(symbols),
            indexed_edges=len(edges),
            indexed_chunks=len(chunks),
            reused_files=reused_files,
            artifact_files=len(artifact_bundle.files),
            documentation_files=len(doc_files),
            warnings=scip_warnings,
        )

    def query(
        self,
        text: str,
        limit: int = 10,
        *,
        scope: SearchScope = "all",
        focus_paths: tuple[str, ...] = (),
    ):
        if self._query_engine is None:
            self._query_engine = QueryEngine(self.storage, self._provider_instance())
        return self._query_engine.search(
            self.project_id, text, limit=limit, scope=scope, focus_paths=focus_paths
        )

    def top_symbols(self, limit: int = 10):
        return self.storage.get_top_symbols(self.project_id, limit)

    def important_symbols(
        self,
        *,
        limit: int = 20,
        kind_filter: str | None = None,
        include_external: bool = False,
    ):
        """Return the top-ranked symbols for the project.

        External library stubs are excluded by default (``include_external=False``).
        They remain in the graph and influence project symbol ranking, but the
        "important symbols" output is meant to reflect the project's own
        architecture, not third-party library inventory.
        """
        symbol_rows = self.storage.get_symbol_rows(self.project_id)
        normalized_filter = _normalized_kind(kind_filter) if kind_filter else None

        def _is_eligible(row) -> bool:
            rp = str(row["relative_path"])
            if not include_external and is_external_symbol(rp):
                return False
            return True

        def _fallback_positive_rows():
            rows = [
                row
                for row in symbol_rows.values()
                if str(row["kind"]) not in {"File", "Module", "Unknown"}
                and float(row["global_rank"]) > 0
                and _is_eligible(row)
                and (
                    normalized_filter is None
                    or _normalized_kind(str(row["kind"])) == normalized_filter
                )
            ]
            rows.sort(key=lambda row: (-float(row["global_rank"]), str(row["display_name"])))
            return rows[:limit]

        rankable_rows = {
            symbol_id: row
            for symbol_id, row in symbol_rows.items()
            if is_rankable_symbol(str(row["relative_path"]), str(row["kind"]))
            and _is_eligible(row)
        }
        if normalized_filter:
            rankable_rows = {
                symbol_id: row
                for symbol_id, row in rankable_rows.items()
                if _normalized_kind(str(row["kind"])) == normalized_filter
            }
        if not rankable_rows:
            return _fallback_positive_rows()

        rows = [
            row
            for row in rankable_rows.values()
            if float(row["global_rank"]) > 0
        ]
        rows.sort(key=lambda row: (-float(row["global_rank"]), str(row["display_name"])))
        if rows:
            return rows[:limit]
        return _fallback_positive_rows()

    def project_overview(self, *, max_per_kind: int = 15) -> str:
        symbol_rows = self.storage.get_symbol_rows(self.project_id)
        rankable_rows = [
            row
            for row in symbol_rows.values()
            if is_rankable_symbol(str(row["relative_path"]), str(row["kind"]))
            and not is_external_symbol(str(row["relative_path"]))
            and float(row["global_rank"]) > 0
        ]
        rankable_rows.sort(key=lambda row: (-float(row["global_rank"]), str(row["display_name"])))
        buckets = {
            "Interfaces": {"interface"},
            "Structs and Classes": {"struct", "class", "trait", "typealias", "type"},
            "Functions and Methods": {"function", "method"},
            "Constants and Values": {"constant", "const", "property"},
        }
        lines = [f"Project overview for {self.settings.project_root}"]
        used_ids: set[str] = set()
        for title, kinds in buckets.items():
            lines.append("")
            lines.append(f"{title}:")
            count = 0
            for row in rankable_rows:
                symbol_id = str(row["symbol_id"])
                if symbol_id in used_ids:
                    continue
                if _normalized_kind(str(row["kind"])) not in kinds:
                    continue
                used_ids.add(symbol_id)
                count += 1
                lines.append(
                    f"- {row['display_name']} [{row['kind']}] {row['relative_path']} score={float(row['global_rank']):.5f}"
                )
                if count >= max_per_kind:
                    break
            if count == 0:
                lines.append("- none")
        return "\n".join(lines)

    def project_stats(self) -> str:
        files = list(self.storage.get_file_hashes(self.project_id).keys())
        symbols = self.storage.get_symbol_rows(self.project_id)
        edges = self.storage.get_edges(self.project_id)
        chunks = self.storage.load_chunk_vectors(self.project_id)[1]
        kind_counts = Counter(str(row["kind"]) for row in symbols.values())
        folder_counts: dict[str, int] = defaultdict(int)
        for relative_path in files:
            parent = PurePosixPath(relative_path).parent.as_posix()
            folder_counts["." if parent == "." else parent] += 1
        lines = [
            f"Project: {self.settings.project_root}",
            f"Files: {len(files)}",
            f"Symbols: {len(symbols)}",
            f"Edges: {len(edges)}",
            f"Embedded chunks: {len(chunks)}",
            "",
            "Top symbol kinds:",
        ]
        for kind, count in kind_counts.most_common(10):
            lines.append(f"- {kind}: {count}")
        lines.append("")
        lines.append("Top folders by file count:")
        for folder, count in sorted(folder_counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
            lines.append(f"- {folder}: {count}")
        return "\n".join(lines)

    def scored_tree(
        self,
        *,
        max_tokens: int | None = 3_000,
        top_k: int | None = None,
        header_title: str = "Project tree scored by usefulness.",
        header_description: list[str] | None = None,
        include_value_details: bool = False,
    ) -> str:
        rows = self.storage.get_tree_score_rows(self.project_id)
        
        # If top_k is specified, filter to top k files by score
        if top_k is not None and top_k > 0:
            scored_rows = []
            for row in rows:
                score = _file_usefulness_score(
                    float(row["rank_sum"]),
                    float(row["rank_max"]),
                    int(row["useful_symbol_count"]),
                    int(row["chunk_count"]),
                )
                scored_rows.append((row, score))
            scored_rows.sort(key=lambda item: -item[1])
            rows = [row for row, _ in scored_rows[:top_k]]
        
        root = TreeScoreNode(
            name=self.settings.project_root.name or str(self.settings.project_root),
            relative_path="",
            node_type="dir",
            raw_score=0.0,
        )
        directories: dict[str, TreeScoreNode] = {"": root}

        def ensure_dir(relative_path: str) -> TreeScoreNode:
            if relative_path in directories:
                return directories[relative_path]
            parent_path = PurePosixPath(relative_path).parent.as_posix()
            if parent_path == ".":
                parent_path = ""
            parent = ensure_dir(parent_path)
            node = TreeScoreNode(
                name=PurePosixPath(relative_path).name,
                relative_path=relative_path,
                node_type="dir",
                raw_score=0.0,
            )
            parent.children.append(node)
            directories[relative_path] = node
            return node

        for row in rows:
            relative_path = str(row["relative_path"])
            path = PurePosixPath(relative_path)
            parent_path = path.parent.as_posix()
            if parent_path == ".":
                parent_path = ""
            parent = ensure_dir(parent_path)
            file_node = TreeScoreNode(
                name=path.name,
                relative_path=relative_path,
                node_type="file",
                raw_score=_file_usefulness_score(
                    float(row["rank_sum"]),
                    float(row["rank_max"]),
                    int(row["useful_symbol_count"]),
                    int(row["chunk_count"]),
                ),
                useful_symbol_count=int(row["useful_symbol_count"]),
                chunk_count=int(row["chunk_count"]),
                source_kind=str(row["source_kind"]),
            )
            parent.children.append(file_node)
            current_path = parent_path
            while True:
                directory = ensure_dir(current_path)
                directory.raw_score += file_node.raw_score
                if current_path == "":
                    break
                next_parent = PurePosixPath(current_path).parent.as_posix()
                current_path = "" if next_parent == "." else next_parent

        _normalize_tree_scores(root, root.raw_score)
        return format_scored_tree(
            root,
            max_tokens=max_tokens,
            top_k=top_k,
            header_title=header_title,
            header_description=header_description,
            include_value_details=include_value_details,
        )

    def sorted_tree(
        self,
        *,
        max_tokens: int | None = 3_000,
        top_k: int | None = None,
    ) -> str:
        return self.scored_tree(
            max_tokens=max_tokens,
            top_k=top_k,
            header_title="Project tree sorted by folder accumulation and file value.",
            header_description=[
                "Folders sort by accumulated descendant score.",
                "Files sort by their direct value score.",
            ],
            include_value_details=True,
        )

    def visualize(self, groups, title: str = "Ultimate Indexer Visualization") -> Path:
        output_path = self.settings.visuals_dir / "query_graph.html"
        return write_query_visualization(
            storage=self.storage,
            project_id=self.project_id,
            groups=groups,
            output_path=output_path,
            title=title,
        )
