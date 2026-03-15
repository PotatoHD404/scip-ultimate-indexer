from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Iterable

from .constants import MAX_FILE_BYTES, SPECIAL_FILES, get_language_from_extension, is_indexable_filename
from .config import Settings
from .embeddings import (
    HashEmbeddingProvider,
    generate_embeddings,
    prepare_document_text,
    resolve_llama_cpp_provider,
)
from .fallback import build_fallback_bundle
from .ignore_rules import create_ignore_matcher
from .models import ChunkRecord, EdgeRecord, FileRecord, IndexSummary, SymbolRecord
from .pagerank import weighted_pagerank
from .python_scip import emit_python_scip
from .query import QueryEngine
from .scip_parser import ParsedScip, parse_scip_index
from .scip_runner import run_scip_indexers
from .socraticode import ingest_socraticode_artifacts
from .storage import Storage
from .visuals import write_query_visualization


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
    model_repo_id: str,
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
    payload.append(f"model-repo:{model_repo_id}")
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


class UltimateIndexer:
    def __init__(self, project_root: Path, embedding_backend: str | None = None) -> None:
        self.settings = Settings(project_root=project_root)
        if embedding_backend is not None:
            self.settings.embedding_backend = embedding_backend
        self.settings.ensure_directories()
        self.project_id = str(self.settings.project_root)
        self.storage = Storage(self.settings.database_path)
        self._provider = None

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
                model_cache_dir=self.settings.model_cache_dir,
                repo_id=self.settings.model_repo_id,
                filename=self.settings.model_filename,
            )
            return self._provider
        except Exception:
            if backend == "llama-cpp":
                raise
            self._provider = HashEmbeddingProvider()
            return self._provider

    def _embed_chunks(self, chunks: list[ChunkRecord]) -> list[ChunkRecord]:
        provider = self._provider_instance()
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        for index, chunk in enumerate(chunks):
            embedding_text = prepare_document_text(chunk.content, chunk.relative_path)
            cached = self.storage.get_or_create_embedding(provider.model_id, embedding_text)
            if cached is not None:
                chunk.embedding = cached.astype("float32").tobytes()
                chunk.embedding_dim = int(cached.shape[0])
                continue
            pending_indices.append(index)
            pending_texts.append(embedding_text)
        if pending_texts:
            vectors = generate_embeddings(provider, pending_texts)
            for list_index, vector in enumerate(vectors):
                chunk = chunks[pending_indices[list_index]]
                embedding_text = pending_texts[list_index]
                self.storage.store_embedding(provider.model_id, embedding_text, vector)
                chunk.embedding = vector.astype("float32").tobytes()
                chunk.embedding_dim = int(vector.shape[0])
        return chunks

    def _global_ranks(self) -> None:
        symbol_ids = list(self.storage.get_symbol_rows(self.project_id).keys())
        if not symbol_ids:
            return
        edges = [
            (
                str(edge["source_symbol_id"]),
                str(edge["target_symbol_id"]),
                float(edge["weight"]),
            )
            for edge in self.storage.get_edges(self.project_id)
        ]
        ranks = weighted_pagerank(nodes=symbol_ids, edges=edges, alpha=0.85)
        self.storage.set_global_ranks(self.project_id, ranks)

    def index(self, scip_path: Path | None = None, force: bool = False) -> IndexSummary:
        code_files = _discover_code_files(self.settings.project_root, self.settings.extra_extensions)
        signature = _project_signature(
            code_files,
            self.settings.project_root,
            self.settings.embedding_backend,
            self.settings.model_repo_id,
            self.settings.model_filename,
            self.settings.extra_extensions,
            self.settings.max_chunk_lines,
            self.settings.chunk_overlap,
        )
        if not force and self.storage.get_project_signature(self.project_id) == signature:
            top_count = len(self.storage.get_symbol_rows(self.project_id))
            return IndexSummary(
                project_id=self.project_id,
                indexed_files=len(self.storage.get_file_hashes(self.project_id)),
                indexed_symbols=top_count,
                indexed_edges=len(self.storage.get_edges(self.project_id)),
                indexed_chunks=len(self.storage.load_chunk_vectors(self.project_id)[1]),
                reused_files=len(code_files),
                artifact_files=0,
            )

        parsed_files: list[FileRecord] = []
        parsed_symbols: list[SymbolRecord] = []
        parsed_edges: list[EdgeRecord] = []
        python_files = [path for path in code_files if get_language_from_extension(path.suffix.lower()) == "python"]

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
        else:
            scip_results = run_scip_indexers(self.settings.project_root, code_files, self.settings.cache_dir)
            for result in scip_results:
                parsed = parse_scip_index(
                    project_id=self.project_id,
                    project_root=self.settings.project_root,
                    index_path=result.index_path,
                    edge_weights=self.settings.edge_weights,
                )
                parsed_files.extend(parsed.files)
                parsed_symbols.extend(parsed.symbols)
                parsed_edges.extend(parsed.edges)
            if python_files:
                parsed_paths = {record.relative_path for record in parsed_files}
                python_only = [path for path in python_files if path.relative_to(self.settings.project_root).as_posix() not in parsed_paths]
                if python_only:
                    python_scip_path = emit_python_scip(self.settings.project_root, python_only, self.settings.scip_cache_path)
                    parsed = parse_scip_index(
                        project_id=self.project_id,
                        project_root=self.settings.project_root,
                        index_path=python_scip_path,
                        edge_weights=self.settings.edge_weights,
                    )
                    parsed_files.extend(parsed.files)
                    parsed_symbols.extend(parsed.symbols)
                    parsed_edges.extend(parsed.edges)

        parsed = ParsedScip(files=parsed_files, symbols=parsed_symbols, edges=parsed_edges)

        artifact_bundle = ingest_socraticode_artifacts(self.project_id, self.settings.project_root)
        covered_paths = {record.relative_path for record in parsed.files}
        covered_paths.update(record.relative_path for record in artifact_bundle.files)
        fallback_bundle = build_fallback_bundle(
            project_id=self.project_id,
            project_root=self.settings.project_root,
            files=code_files,
            covered_paths=covered_paths,
            contains_weight=self.settings.edge_weights["contains"],
            import_weight=self.settings.edge_weights["imports"],
            max_chunk_lines=self.settings.max_chunk_lines,
            chunk_overlap=self.settings.chunk_overlap,
        )
        files: list[FileRecord] = [*parsed.files, *fallback_bundle.files, *artifact_bundle.files]
        symbols: list[SymbolRecord] = [*parsed.symbols, *fallback_bundle.symbols, *artifact_bundle.symbols]
        edges = [*parsed.edges, *fallback_bundle.edges, *artifact_bundle.edges]

        symbols_by_file: dict[str, list[SymbolRecord]] = {}
        for symbol in symbols:
            symbols_by_file.setdefault(symbol.relative_path, []).append(symbol)

        chunks: list[ChunkRecord] = [*artifact_bundle.chunks, *fallback_bundle.chunks]
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
                chunk_content = "\n".join(
                    part
                    for part in [
                        symbol.relative_path,
                        symbol.signature,
                        symbol.docstring,
                        symbol.snippet,
                    ]
                    if part
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

        self._embed_chunks(chunks)
        self.storage.replace_project_contents(self.project_id, files, symbols, edges, chunks)
        self.storage.upsert_project(self.project_id, self.project_id, signature)
        self._global_ranks()
        return IndexSummary(
            project_id=self.project_id,
            indexed_files=len(files),
            indexed_symbols=len(symbols),
            indexed_edges=len(edges),
            indexed_chunks=len(chunks),
            reused_files=0,
            artifact_files=len(artifact_bundle.files),
        )

    def query(self, text: str, limit: int = 10):
        engine = QueryEngine(self.storage, self._provider_instance())
        return engine.search(self.project_id, text, limit=limit)

    def top_symbols(self, limit: int = 10):
        return self.storage.get_top_symbols(self.project_id, limit)

    def visualize(self, groups, title: str = "Ultimate Indexer Visualization") -> Path:
        output_path = self.settings.visuals_dir / "query_graph.html"
        return write_query_visualization(
            storage=self.storage,
            project_id=self.project_id,
            groups=groups,
            output_path=output_path,
            title=title,
        )
