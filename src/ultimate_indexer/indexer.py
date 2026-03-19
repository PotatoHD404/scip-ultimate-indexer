from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

import networkx as nx

from .constants import MAX_FILE_BYTES, SPECIAL_FILES, is_indexable_filename
from .config import Settings
from .embeddings import (
    HashEmbeddingProvider,
    generate_embeddings,
    _provider_prepare_document_text,
    resolve_llama_cpp_provider,
)
from .fallback import build_fallback_bundle
from .formatter import _pretty_signature, format_scored_tree
from .ignore_rules import create_ignore_matcher
from .models import ChunkRecord, EdgeRecord, FileRecord, IndexProgress, IndexSummary, SymbolRecord, TreeScoreNode
from .query import QueryEngine, apply_kind_boost, dependency_ordered_pagerank
from .ranking_rules import is_rankable_symbol
from .scip_parser import ParsedScip, parse_scip_index
from .scip_runner import ScipRunFailure, StructuredIndexingRequiredError, run_scip_indexers
from .socraticode import ingest_socraticode_artifacts
from .storage import Storage
from .visuals import write_query_visualization


INDEX_STAGES = (
    "discover",
    "scip",
    "artifacts",
    "fallback",
    "chunks",
    "embed",
    "store",
    "pagerank",
)
INDEX_FORMAT_VERSION = 5


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
    project_root: Path,
    embedding_backend: str,
    model_path: str | None,
    model_filename: str,
    extra_extensions: set[str],
    max_chunk_lines: int,
    chunk_overlap: int,
) -> str:
    payload: list[str] = []
    for path in code_files:
        relative = path.relative_to(project_root)
        payload.append(f"{relative}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
    payload.append(f"embedding-backend:{embedding_backend}")
    payload.append(f"index-format-version:{INDEX_FORMAT_VERSION}")
    payload.append(f"model-path:{model_path or ''}")
    payload.append(f"model-file:{model_filename}")
    payload.append(f"extra-extensions:{','.join(sorted(extra_extensions))}")
    payload.append(f"respect-gitignore:{os.getenv('RESPECT_GITIGNORE', 'true')}")
    payload.append(f"ignore-dirs:{os.getenv('IGNORE_DIRS', '')}")
    payload.append(f"scip-languages:{os.getenv('SCIP_LANGUAGES', '')}")
    payload.append(f"max-chunk-lines:{max_chunk_lines}")
    payload.append(f"chunk-overlap:{chunk_overlap}")
    for special_name in (
        ".socraticodecontextartifacts.json",
        ".gitignore",
        ".socraticodeignore",
        ".cgcignore",
    ):
        candidate = project_root / special_name
        if candidate.exists():
            payload.append(f"{special_name}:{candidate.stat().st_mtime_ns}:{candidate.stat().st_size}")
    return sha256("\n".join(payload).encode("utf-8")).hexdigest()


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

    def close(self) -> None:
        self.storage.close()

    def _provider_instance(self):
        if self._provider is not None:
            return self._provider
        backend = self.settings.embedding_backend
        if backend == "hash":
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
            if backend in {"llama-cpp", "local"}:
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
        progress_callback: Callable[[IndexProgress], None] | None = None,
    ) -> list[ChunkRecord]:
        provider = self._provider_instance()
        existing_chunk_embeddings = self.storage.get_chunk_embeddings(self.project_id)
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        cached_count = 0
        for index, chunk in enumerate(chunks):
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
            embedding_text = _provider_prepare_document_text(provider, chunk.content, chunk.relative_path)
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
        self._emit_progress(
            progress_callback,
            stage="embed",
            completed=cached_count,
            total=max(len(chunks), 1),
            unit="chunks",
            detail="Embedding chunks",
        )
        if pending_texts:
            vectors = generate_embeddings(
                provider,
                pending_texts,
                on_batch_complete=lambda processed, total: self._emit_progress(
                    progress_callback,
                    stage="embed",
                    completed=cached_count + processed,
                    total=max(len(chunks), 1),
                    unit="chunks",
                    detail="Embedding chunks",
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
            completed=max(len(chunks), 1),
            total=max(len(chunks), 1),
            unit="chunks",
            detail="Embedded chunks",
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
        boosted = {
            symbol_id: apply_kind_boost(symbol_rows[symbol_id], score)
            for symbol_id, score in ranks.items()
            if symbol_id in symbol_rows
        }
        total = sum(boosted.values())
        if total > 0:
            boosted = {symbol_id: score / total for symbol_id, score in boosted.items()}
        else:
            boosted = {}
        self.storage.set_global_ranks(self.project_id, boosted)

    def index(
        self,
        scip_path: Path | None = None,
        force: bool = False,
        progress_callback: Callable[[IndexProgress], None] | None = None,
    ) -> IndexSummary:
        scip_warnings: list[str] = []
        self._emit_progress(progress_callback, stage="discover", detail="Scanning project files")
        code_files = _discover_code_files(self.settings.project_root, self.settings.extra_extensions)
        self._emit_progress(
            progress_callback,
            stage="discover",
            completed=max(len(code_files), 1),
            total=max(len(code_files), 1),
            unit="files",
            detail="Discovered indexable files",
        )
        signature = _project_signature(
            code_files,
            self.settings.project_root,
            self.settings.embedding_backend,
            self.settings.model_path,
            self.settings.model_filename,
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
            scip_report = run_scip_indexers(self.settings.project_root, code_files, self.settings.cache_dir)
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
        covered_paths = {record.relative_path for record in parsed.files}
        covered_paths.update(record.relative_path for record in artifact_bundle.files)
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
        files: list[FileRecord] = [*parsed.files, *fallback_bundle.files, *artifact_bundle.files]
        symbols: list[SymbolRecord] = [*parsed.symbols, *fallback_bundle.symbols, *artifact_bundle.symbols]
        edges = [*parsed.edges, *fallback_bundle.edges, *artifact_bundle.edges]

        symbols_by_file: dict[str, list[SymbolRecord]] = {}
        symbol_lookup: dict[str, SymbolRecord] = {}
        children_by_symbol_id: dict[str, list[SymbolRecord]] = {}
        for symbol in symbols:
            symbols_by_file.setdefault(symbol.relative_path, []).append(symbol)
            symbol_lookup[symbol.symbol_id] = symbol
            if symbol.enclosing_symbol_id:
                children_by_symbol_id.setdefault(symbol.enclosing_symbol_id, []).append(symbol)

        chunks: list[ChunkRecord] = [*artifact_bundle.chunks, *fallback_bundle.chunks]
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
                        content_hash=sha256(chunk_content.encode("utf-8")).hexdigest(),
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

        self._embed_chunks(chunks, progress_callback=progress_callback)
        self._emit_progress(progress_callback, stage="store", detail="Writing index to SQLite")
        self.storage.replace_project_contents(self.project_id, files, symbols, edges, chunks)
        self.storage.upsert_project(self.project_id, self.project_id, signature)
        self._emit_progress(
            progress_callback,
            stage="store",
            completed=1,
            total=1,
            unit="projects",
            detail="Stored indexed data",
        )
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
            reused_files=0,
            artifact_files=len(artifact_bundle.files),
            warnings=scip_warnings,
        )

    def query(self, text: str, limit: int = 10):
        if self._query_engine is None:
            self._query_engine = QueryEngine(self.storage, self._provider_instance())
        return self._query_engine.search(self.project_id, text, limit=limit)

    def top_symbols(self, limit: int = 10):
        return self.storage.get_top_symbols(self.project_id, limit)

    def important_symbols(
        self,
        *,
        limit: int = 20,
        metric: str = "pagerank",
        kind_filter: str | None = None,
    ):
        symbol_rows = self.storage.get_symbol_rows(self.project_id)
        normalized_filter = _normalized_kind(kind_filter) if kind_filter else None

        def _fallback_positive_rows():
            rows = [
                row
                for row in symbol_rows.values()
                if str(row["kind"]) not in {"File", "Module", "Unknown"}
                and float(row["global_rank"]) > 0
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
        }
        if normalized_filter:
            rankable_rows = {
                symbol_id: row
                for symbol_id, row in rankable_rows.items()
                if _normalized_kind(str(row["kind"])) == normalized_filter
            }
        if not rankable_rows:
            return _fallback_positive_rows()

        if metric == "pagerank":
            rows = [
                row
                for row in rankable_rows.values()
                if float(row["global_rank"]) > 0
            ]
            rows.sort(key=lambda row: (-float(row["global_rank"]), str(row["display_name"])))
            if rows:
                return rows[:limit]
            return _fallback_positive_rows()

        graph = nx.DiGraph()
        for symbol_id in rankable_rows:
            graph.add_node(symbol_id)
        for edge in self.storage.get_edges(self.project_id):
            source = str(edge["source_symbol_id"])
            target = str(edge["target_symbol_id"])
            if source in rankable_rows and target in rankable_rows:
                graph.add_edge(source, target, weight=float(edge["weight"]))

        if metric == "betweenness":
            raw_scores = nx.betweenness_centrality(graph) if graph.number_of_nodes() else {}
        elif metric == "in_degree":
            raw_scores = {node: float(score) for node, score in graph.in_degree()}
        elif metric == "out_degree":
            raw_scores = {node: float(score) for node, score in graph.out_degree()}
        else:
            raw_scores = {symbol_id: float(row["global_rank"]) for symbol_id, row in rankable_rows.items()}

        ranked_ids = sorted(raw_scores, key=lambda symbol_id: (-raw_scores[symbol_id], str(rankable_rows[symbol_id]["display_name"])))
        rows = [rankable_rows[symbol_id] for symbol_id in ranked_ids[:limit] if raw_scores[symbol_id] > 0]
        if rows:
            return rows
        return _fallback_positive_rows()

    def project_overview(self, *, max_per_kind: int = 15) -> str:
        symbol_rows = self.storage.get_symbol_rows(self.project_id)
        rankable_rows = [
            row
            for row in symbol_rows.values()
            if is_rankable_symbol(str(row["relative_path"]), str(row["kind"]))
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

    def scored_tree(self, *, max_chars: int | None = 12_000) -> str:
        rows = self.storage.get_tree_score_rows(self.project_id)
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
        return format_scored_tree(root, max_chars=max_chars)

    def visualize(self, groups, title: str = "Ultimate Indexer Visualization") -> Path:
        output_path = self.settings.visuals_dir / "query_graph.html"
        return write_query_visualization(
            storage=self.storage,
            project_id=self.project_id,
            groups=groups,
            output_path=output_path,
            title=title,
        )
