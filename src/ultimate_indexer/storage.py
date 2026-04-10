from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .embeddings import hash_text
from .models import (
    ChunkRecord,
    EdgeRecord,
    FileRecord,
    FunctionBodyChunkRecord,
    FunctionMetadataRecord,
    QueryChunkHit,
    SymbolRecord,
)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Storage:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                root_path TEXT NOT NULL,
                signature TEXT DEFAULT '',
                indexed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS files (
                project_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                abs_path TEXT NOT NULL,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                artifact_name TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_id, relative_path)
            );

            CREATE TABLE IF NOT EXISTS symbols (
                project_id TEXT NOT NULL,
                symbol_id TEXT NOT NULL,
                scip_symbol TEXT NOT NULL,
                display_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                signature TEXT NOT NULL,
                docstring TEXT NOT NULL,
                snippet TEXT NOT NULL,
                enclosing_symbol_id TEXT,
                source_kind TEXT NOT NULL,
                global_rank REAL DEFAULT 0.0,
                PRIMARY KEY (project_id, symbol_id)
            );

            CREATE TABLE IF NOT EXISTS edges (
                project_id TEXT NOT NULL,
                source_symbol_id TEXT NOT NULL,
                target_symbol_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                project_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                symbol_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                artifact_name TEXT,
                chunk_kind TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model_id TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (project_id, chunk_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                project_id UNINDEXED,
                chunk_id UNINDEXED,
                relative_path,
                symbol_name,
                artifact_name,
                content
            );

            CREATE TABLE IF NOT EXISTS function_metadata (
                project_id TEXT NOT NULL,
                symbol_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                fully_qualified_name TEXT NOT NULL,
                signature TEXT NOT NULL,
                normalized_signature TEXT NOT NULL,
                docstring TEXT NOT NULL,
                params TEXT NOT NULL,
                param_types TEXT NOT NULL,
                return_type TEXT NOT NULL,
                decorators TEXT NOT NULL,
                referenced_types TEXT NOT NULL,
                called_functions TEXT NOT NULL,
                raised_exceptions TEXT NOT NULL,
                literals TEXT NOT NULL,
                behavioral_tags TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                metadata_content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model_id TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (project_id, symbol_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS function_metadata_fts USING fts5(
                project_id UNINDEXED,
                symbol_id UNINDEXED,
                fully_qualified_name,
                signature,
                params,
                return_type,
                called_functions,
                referenced_types,
                behavioral_tags,
                literals,
                docstring,
                metadata_content
            );

            CREATE TABLE IF NOT EXISTS function_body_chunks (
                project_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                symbol_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                display_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                signature TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                body TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model_id TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (project_id, chunk_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS function_body_fts USING fts5(
                project_id UNINDEXED,
                chunk_id UNINDEXED,
                symbol_id UNINDEXED,
                display_name,
                signature,
                chunk_type,
                body,
                content
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                model_id TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (model_id, text_hash)
            );

            CREATE TABLE IF NOT EXISTS query_cache (
                project_id TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (project_id, query_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(project_id, relative_path);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(project_id, source_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(project_id, target_symbol_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(project_id, symbol_id);
            CREATE INDEX IF NOT EXISTS idx_function_metadata_symbol ON function_metadata(project_id, symbol_id);
            CREATE INDEX IF NOT EXISTS idx_function_body_symbol ON function_body_chunks(project_id, symbol_id);
            """
        )
        chunk_columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "embedding_model_id" not in chunk_columns:
            self.connection.execute(
                "ALTER TABLE chunks ADD COLUMN embedding_model_id TEXT NOT NULL DEFAULT ''"
            )
        self.connection.commit()

    def upsert_project(self, project_id: str, root_path: str, signature: str) -> None:
        self.connection.execute(
            """
            INSERT INTO projects(project_id, root_path, signature, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                root_path=excluded.root_path,
                signature=excluded.signature,
                indexed_at=excluded.indexed_at
            """,
            (project_id, root_path, signature, _utcnow()),
        )
        self.connection.commit()

    def get_project_signature(self, project_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT signature FROM projects WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        return None if row is None else str(row["signature"])

    def get_file_hashes(self, project_id: str) -> dict[str, str]:
        rows = self.connection.execute(
            "SELECT relative_path, content_hash FROM files WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {str(row["relative_path"]): str(row["content_hash"]) for row in rows}

    def replace_project_contents(
        self,
        project_id: str,
        files: Iterable[FileRecord],
        symbols: Iterable[SymbolRecord],
        edges: Iterable[EdgeRecord],
        chunks: Iterable[ChunkRecord],
        function_metadata: Iterable[FunctionMetadataRecord] | None = None,
        function_body_chunks: Iterable[FunctionBodyChunkRecord] | None = None,
        changed_paths: set[str] | None = None,
        removed_paths: set[str] | None = None,
    ) -> None:
        files = list(files)
        symbols = list(symbols)
        edges = list(edges)
        chunks = list(chunks)
        function_metadata = list(function_metadata) if function_metadata else []
        function_body_chunks = (
            list(function_body_chunks) if function_body_chunks else []
        )
        incremental_mode = changed_paths is not None or removed_paths is not None
        changed_paths = set(changed_paths or set())
        removed_paths = set(removed_paths or set())
        target_paths = changed_paths | removed_paths
        affected_symbol_ids: set[str] = set()

        if incremental_mode:
            files = [item for item in files if item.relative_path in changed_paths]
            symbols = [item for item in symbols if item.relative_path in changed_paths]
            chunks = [item for item in chunks if item.relative_path in changed_paths]
            function_metadata = [
                item for item in function_metadata if item.relative_path in changed_paths
            ]
            function_body_chunks = [
                item for item in function_body_chunks if item.relative_path in changed_paths
            ]
            if target_paths:
                path_placeholders = ",".join("?" for _ in target_paths)
                old_rows = self.connection.execute(
                    f"""
                    SELECT symbol_id
                    FROM symbols
                    WHERE project_id = ? AND relative_path IN ({path_placeholders})
                    """,
                    (project_id, *sorted(target_paths)),
                ).fetchall()
                affected_symbol_ids.update(str(row["symbol_id"]) for row in old_rows)
            affected_symbol_ids.update(item.symbol_id for item in symbols)

        keep_paths = {item.relative_path for item in files}
        with self.connection:
            self.connection.execute(
                "DELETE FROM query_cache WHERE project_id = ?", (project_id,)
            )
            if incremental_mode:
                if affected_symbol_ids:
                    symbol_placeholders = ",".join("?" for _ in affected_symbol_ids)
                    self.connection.execute(
                        f"""
                        DELETE FROM edges
                        WHERE project_id = ?
                          AND (
                            source_symbol_id IN ({symbol_placeholders})
                            OR target_symbol_id IN ({symbol_placeholders})
                          )
                        """,
                        (project_id, *sorted(affected_symbol_ids), *sorted(affected_symbol_ids)),
                    )
                if target_paths:
                    path_placeholders = ",".join("?" for _ in target_paths)
                    self.connection.execute(
                        f"""
                        DELETE FROM chunk_fts
                        WHERE project_id = ?
                          AND chunk_id IN (
                            SELECT chunk_id
                            FROM chunks
                            WHERE project_id = ? AND relative_path IN ({path_placeholders})
                          )
                        """,
                        (project_id, project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM chunks WHERE project_id = ? AND relative_path IN ({path_placeholders})",
                        (project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM symbols WHERE project_id = ? AND relative_path IN ({path_placeholders})",
                        (project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM function_metadata_fts WHERE project_id = ? AND symbol_id IN (SELECT symbol_id FROM function_metadata WHERE project_id = ? AND relative_path IN ({path_placeholders}))",
                        (project_id, project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM function_metadata WHERE project_id = ? AND relative_path IN ({path_placeholders})",
                        (project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM function_body_fts WHERE project_id = ? AND chunk_id IN (SELECT chunk_id FROM function_body_chunks WHERE project_id = ? AND relative_path IN ({path_placeholders}))",
                        (project_id, project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM function_body_chunks WHERE project_id = ? AND relative_path IN ({path_placeholders})",
                        (project_id, *sorted(target_paths)),
                    )
                    self.connection.execute(
                        f"DELETE FROM files WHERE project_id = ? AND relative_path IN ({path_placeholders})",
                        (project_id, *sorted(target_paths)),
                    )
            else:
                self.connection.execute(
                    "DELETE FROM edges WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM chunk_fts WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM chunks WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM symbols WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM function_metadata_fts WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM function_metadata WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM function_body_fts WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM function_body_chunks WHERE project_id = ?", (project_id,)
                )
                self.connection.execute(
                    "DELETE FROM files WHERE project_id = ? AND relative_path NOT IN ({})".format(
                        ",".join("?" for _ in keep_paths) if keep_paths else "''"
                    ),
                    (project_id, *sorted(keep_paths)),
                )

            for file_record in files:
                self.connection.execute(
                    """
                    INSERT INTO files(
                        project_id, relative_path, abs_path, language, content_hash, content,
                        source_kind, artifact_name, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(project_id, relative_path) DO UPDATE SET
                        abs_path=excluded.abs_path,
                        language=excluded.language,
                        content_hash=excluded.content_hash,
                        content=excluded.content,
                        source_kind=excluded.source_kind,
                        artifact_name=excluded.artifact_name,
                        updated_at=excluded.updated_at
                    """,
                    (
                        file_record.project_id,
                        file_record.relative_path,
                        file_record.abs_path,
                        file_record.language,
                        file_record.content_hash,
                        file_record.content,
                        file_record.source_kind,
                        file_record.artifact_name,
                        _utcnow(),
                    ),
                )

            for symbol in symbols:
                self.connection.execute(
                    """
                    INSERT INTO symbols(
                        project_id, symbol_id, scip_symbol, display_name, kind, relative_path,
                        start_line, end_line, signature, docstring, snippet, enclosing_symbol_id,
                        source_kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol.project_id,
                        symbol.symbol_id,
                        symbol.scip_symbol,
                        symbol.display_name,
                        symbol.kind,
                        symbol.relative_path,
                        symbol.start_line,
                        symbol.end_line,
                        symbol.signature,
                        symbol.docstring,
                        symbol.snippet,
                        symbol.enclosing_symbol_id,
                        symbol.source_kind,
                    ),
                )

            for edge in edges:
                if incremental_mode and affected_symbol_ids:
                    if (
                        edge.source_symbol_id not in affected_symbol_ids
                        and edge.target_symbol_id not in affected_symbol_ids
                    ):
                        continue
                self.connection.execute(
                    """
                    INSERT INTO edges(project_id, source_symbol_id, target_symbol_id, edge_type, weight)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        edge.project_id,
                        edge.source_symbol_id,
                        edge.target_symbol_id,
                        edge.edge_type,
                        edge.weight,
                    ),
                )

            for chunk in chunks:
                self.connection.execute(
                    """
                    INSERT INTO chunks(
                        project_id, chunk_id, relative_path, symbol_id, symbol_name, artifact_name,
                        chunk_kind, start_line, end_line, content, content_hash, embedding, embedding_dim,
                        embedding_model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.project_id,
                        chunk.chunk_id,
                        chunk.relative_path,
                        chunk.symbol_id,
                        chunk.symbol_name,
                        chunk.artifact_name,
                        chunk.chunk_kind,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.content,
                        chunk.content_hash,
                        chunk.embedding,
                        chunk.embedding_dim,
                        chunk.embedding_model_id,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO chunk_fts(project_id, chunk_id, relative_path, symbol_name, artifact_name, content)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.project_id,
                        chunk.chunk_id,
                        chunk.relative_path,
                        chunk.symbol_name,
                        chunk.artifact_name or "",
                        chunk.content,
                    ),
                )

            for meta in function_metadata:
                self.connection.execute(
                    """
                    INSERT INTO function_metadata(
                        project_id, symbol_id, relative_path, display_name, kind, fully_qualified_name,
                        signature, normalized_signature, docstring, params, param_types, return_type,
                        decorators, referenced_types, called_functions, raised_exceptions, literals,
                        behavioral_tags, start_line, end_line, metadata_content, content_hash,
                        embedding, embedding_dim, embedding_model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.project_id,
                        meta.symbol_id,
                        meta.relative_path,
                        meta.display_name,
                        meta.kind,
                        meta.fully_qualified_name,
                        meta.signature,
                        meta.normalized_signature,
                        meta.docstring,
                        json.dumps(meta.params),
                        json.dumps(meta.param_types),
                        meta.return_type,
                        json.dumps(meta.decorators),
                        json.dumps(meta.referenced_types),
                        json.dumps(meta.called_functions),
                        json.dumps(meta.raised_exceptions),
                        json.dumps(meta.literals),
                        json.dumps(meta.behavioral_tags),
                        meta.start_line,
                        meta.end_line,
                        meta.metadata_content,
                        meta.content_hash,
                        meta.embedding,
                        meta.embedding_dim,
                        meta.embedding_model_id,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO function_metadata_fts(
                        project_id, symbol_id, fully_qualified_name, signature, params, return_type,
                        called_functions, referenced_types, behavioral_tags, literals, docstring, metadata_content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        meta.project_id,
                        meta.symbol_id,
                        meta.fully_qualified_name,
                        meta.signature,
                        json.dumps(meta.params),
                        meta.return_type,
                        json.dumps(meta.called_functions),
                        json.dumps(meta.referenced_types),
                        json.dumps(meta.behavioral_tags),
                        json.dumps(meta.literals),
                        meta.docstring,
                        meta.metadata_content,
                    ),
                )

            for body_chunk in function_body_chunks:
                self.connection.execute(
                    """
                    INSERT INTO function_body_chunks(
                        project_id, chunk_id, symbol_id, relative_path, display_name, kind, signature,
                        chunk_index, total_chunks, body, chunk_type, start_line, end_line, content,
                        content_hash, embedding, embedding_dim, embedding_model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        body_chunk.project_id,
                        body_chunk.chunk_id,
                        body_chunk.symbol_id,
                        body_chunk.relative_path,
                        body_chunk.display_name,
                        body_chunk.kind,
                        body_chunk.signature,
                        body_chunk.chunk_index,
                        body_chunk.total_chunks,
                        body_chunk.body,
                        body_chunk.chunk_type,
                        body_chunk.start_line,
                        body_chunk.end_line,
                        body_chunk.content,
                        body_chunk.content_hash,
                        body_chunk.embedding,
                        body_chunk.embedding_dim,
                        body_chunk.embedding_model_id,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO function_body_fts(
                        project_id, chunk_id, symbol_id, display_name, signature, chunk_type, body, content
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        body_chunk.project_id,
                        body_chunk.chunk_id,
                        body_chunk.symbol_id,
                        body_chunk.display_name,
                        body_chunk.signature,
                        body_chunk.chunk_type,
                        body_chunk.body,
                        body_chunk.content,
                    ),
                )

    def get_or_create_embedding(
        self,
        model_id: str,
        text: str,
    ) -> np.ndarray | None:
        text_hash = hash_text(text)
        row = self.connection.execute(
            """
            SELECT embedding, embedding_dim
            FROM embedding_cache
            WHERE model_id = ? AND text_hash = ?
            """,
            (model_id, text_hash),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(
            row["embedding"], dtype=np.float32, count=row["embedding_dim"]
        ).copy()

    def store_embedding(self, model_id: str, text: str, vector: np.ndarray) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO embedding_cache(model_id, text_hash, embedding, embedding_dim, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                model_id,
                hash_text(text),
                vector.astype(np.float32).tobytes(),
                int(vector.shape[0]),
                _utcnow(),
            ),
        )
        self.connection.commit()

    def load_chunk_vectors(
        self,
        project_id: str,
    ) -> tuple[np.ndarray, list[sqlite3.Row]]:
        rows = self.connection.execute(
            """
            SELECT chunk_id, relative_path, symbol_id, symbol_name, content, start_line, end_line, embedding, embedding_dim
            FROM chunks
            WHERE project_id = ? AND embedding IS NOT NULL
            ORDER BY chunk_id
            """,
            (project_id,),
        ).fetchall()
        vectors = [
            np.frombuffer(
                row["embedding"], dtype=np.float32, count=row["embedding_dim"]
            ).copy()
            for row in rows
        ]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), rows
        return np.vstack(vectors), rows

    def load_function_metadata_vectors(
        self,
        project_id: str,
    ) -> tuple[np.ndarray, list[sqlite3.Row]]:
        """Load function metadata vectors for dense search."""
        rows = self.connection.execute(
            """
            SELECT symbol_id, relative_path, display_name, kind, fully_qualified_name,
                   signature, docstring, metadata_content, start_line, end_line, embedding, embedding_dim
            FROM function_metadata
            WHERE project_id = ? AND embedding IS NOT NULL
            ORDER BY symbol_id
            """,
            (project_id,),
        ).fetchall()
        vectors = [
            np.frombuffer(
                row["embedding"], dtype=np.float32, count=row["embedding_dim"]
            ).copy()
            for row in rows
        ]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), rows
        return np.vstack(vectors), rows

    def load_function_body_vectors(
        self,
        project_id: str,
    ) -> tuple[np.ndarray, list[sqlite3.Row]]:
        """Load function body chunk vectors for dense search."""
        rows = self.connection.execute(
            """
            SELECT chunk_id, symbol_id, relative_path, display_name, kind, signature,
                   chunk_index, total_chunks, body, chunk_type, start_line, end_line,
                   content, embedding, embedding_dim
            FROM function_body_chunks
            WHERE project_id = ? AND embedding IS NOT NULL
            ORDER BY symbol_id, chunk_index
            """,
            (project_id,),
        ).fetchall()
        vectors = [
            np.frombuffer(
                row["embedding"], dtype=np.float32, count=row["embedding_dim"]
            ).copy()
            for row in rows
        ]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), rows
        return np.vstack(vectors), rows

    def get_chunk_embeddings(self, project_id: str) -> dict[str, sqlite3.Row]:
        rows = self.connection.execute(
            """
            SELECT chunk_id, content_hash, embedding, embedding_dim, embedding_model_id
            FROM chunks
            WHERE project_id = ? AND embedding IS NOT NULL
            """,
            (project_id,),
        ).fetchall()
        return {str(row["chunk_id"]): row for row in rows}

    def search_function_metadata_bm25(
        self,
        project_id: str,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Search function metadata using BM25."""
        tokens = [
            token
            for token in query.replace("/", " ").replace(".", " ").split()
            if token
        ]
        if not tokens:
            return []
        fts_query = " ".join(f'"{token}"' for token in tokens)
        try:
            rows = self.connection.execute(
                """
                SELECT m.symbol_id, m.relative_path, m.display_name, m.kind, m.fully_qualified_name,
                       m.signature, m.docstring, m.called_functions, m.referenced_types, m.behavioral_tags,
                       m.literals, m.metadata_content, m.start_line, m.end_line,
                       bm25(function_metadata_fts) AS rank
                FROM function_metadata_fts f
                JOIN function_metadata m ON f.symbol_id = m.symbol_id AND f.project_id = m.project_id
                WHERE f.project_id = ? AND f.function_metadata_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (project_id, fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self.connection.execute(
                """
                SELECT symbol_id, relative_path, display_name, kind, fully_qualified_name,
                       signature, docstring, called_functions, referenced_types, behavioral_tags,
                       literals, metadata_content, start_line, end_line, 0.0 AS rank
                FROM function_metadata
                WHERE project_id = ? AND metadata_content LIKE ?
                LIMIT ?
                """,
                (project_id, f"%{query}%", limit),
            ).fetchall()

        results = []
        for idx, row in enumerate(rows):
            raw_rank = float(row["rank"])
            score = 1.0 / (1.0 + max(raw_rank, 0.0) + idx)
            results.append(
                {
                    "symbol_id": str(row["symbol_id"]),
                    "relative_path": str(row["relative_path"]),
                    "display_name": str(row["display_name"]),
                    "kind": str(row["kind"]),
                    "fully_qualified_name": str(row["fully_qualified_name"]),
                    "signature": str(row["signature"]),
                    "docstring": str(row["docstring"]),
                    "called_functions": json.loads(row["called_functions"])
                    if row["called_functions"]
                    else [],
                    "referenced_types": json.loads(row["referenced_types"])
                    if row["referenced_types"]
                    else [],
                    "behavioral_tags": json.loads(row["behavioral_tags"])
                    if row["behavioral_tags"]
                    else [],
                    "literals": json.loads(row["literals"]) if row["literals"] else [],
                    "score": score,
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                }
            )
        return results

    def search_function_body_bm25(
        self,
        project_id: str,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Search function body chunks using BM25."""
        tokens = [
            token
            for token in query.replace("/", " ").replace(".", " ").split()
            if token
        ]
        if not tokens:
            return []
        fts_query = " ".join(f'"{token}"' for token in tokens)
        try:
            rows = self.connection.execute(
                """
                SELECT chunk_id, symbol_id, relative_path, display_name, kind, signature,
                       chunk_index, total_chunks, body, chunk_type, start_line, end_line,
                       content, bm25(function_body_fts) AS rank
                FROM function_body_fts
                JOIN function_body_chunks USING (project_id, chunk_id)
                WHERE function_body_fts.project_id = ? AND function_body_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (project_id, fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self.connection.execute(
                """
                SELECT chunk_id, symbol_id, relative_path, display_name, kind, signature,
                       chunk_index, total_chunks, body, chunk_type, start_line, end_line,
                       content, 0.0 AS rank
                FROM function_body_chunks
                WHERE project_id = ? AND content LIKE ?
                LIMIT ?
                """,
                (project_id, f"%{query}%", limit),
            ).fetchall()

        results = []
        for idx, row in enumerate(rows):
            raw_rank = float(row["rank"])
            score = 1.0 / (1.0 + max(raw_rank, 0.0) + idx)
            results.append(
                {
                    "chunk_id": str(row["chunk_id"]),
                    "symbol_id": str(row["symbol_id"]),
                    "relative_path": str(row["relative_path"]),
                    "display_name": str(row["display_name"]),
                    "kind": str(row["kind"]),
                    "signature": str(row["signature"]),
                    "chunk_index": int(row["chunk_index"]),
                    "total_chunks": int(row["total_chunks"]),
                    "body": str(row["body"]),
                    "chunk_type": str(row["chunk_type"]),
                    "score": score,
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                }
            )
        return results

    def search_bm25(
        self, project_id: str, query: str, limit: int
    ) -> list[QueryChunkHit]:
        tokens = [
            token
            for token in query.replace("/", " ").replace(".", " ").split()
            if token
        ]
        if not tokens:
            return []
        fts_query = " ".join(f'"{token}"' for token in tokens)
        try:
            rows = self.connection.execute(
                """
                SELECT c.chunk_id, c.relative_path, c.symbol_id, c.symbol_name, c.content,
                       c.start_line, c.end_line, bm25(chunk_fts) AS rank
                FROM chunk_fts
                JOIN chunks c
                  ON c.project_id = chunk_fts.project_id
                 AND c.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts.project_id = ? AND chunk_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (project_id, fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self.connection.execute(
                """
                SELECT chunk_id, relative_path, symbol_id, symbol_name, content, start_line, end_line, 0.0 AS rank
                FROM chunks
                WHERE project_id = ? AND content LIKE ?
                LIMIT ?
                """,
                (project_id, f"%{query}%", limit),
            ).fetchall()
        hits = []
        for idx, row in enumerate(rows):
            raw_rank = float(row["rank"])
            score = 1.0 / (1.0 + max(raw_rank, 0.0) + idx)
            hits.append(
                QueryChunkHit(
                    chunk_id=str(row["chunk_id"]),
                    relative_path=str(row["relative_path"]),
                    symbol_id=str(row["symbol_id"]),
                    symbol_name=str(row["symbol_name"]),
                    score=score,
                    content=str(row["content"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                )
            )
        return hits

    def get_symbol_rows(self, project_id: str) -> dict[str, sqlite3.Row]:
        rows = self.connection.execute(
            "SELECT * FROM symbols WHERE project_id = ?",
            (project_id,),
        ).fetchall()
        return {str(row["symbol_id"]): row for row in rows}

    def get_symbol_children(self, project_id: str, symbol_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM symbols
            WHERE project_id = ? AND enclosing_symbol_id = ?
            ORDER BY start_line, display_name
            """,
            (project_id, symbol_id),
        ).fetchall()

    def get_edges(self, project_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM edges WHERE project_id = ?",
            (project_id,),
        ).fetchall()

    def set_global_ranks(self, project_id: str, ranks: dict[str, float]) -> None:
        # Two statements: bulk-reset then batch-update — avoids N individual round-trips.
        with self.connection:
            self.connection.execute(
                "UPDATE symbols SET global_rank = 0.0 WHERE project_id = ?",
                (project_id,),
            )
            self.connection.executemany(
                "UPDATE symbols SET global_rank = ? WHERE project_id = ? AND symbol_id = ?",
                [(score, project_id, sid) for sid, score in ranks.items()],
            )

    def get_top_symbols(
        self,
        project_id: str,
        limit: int,
        *,
        include_external: bool = False,
    ) -> list[sqlite3.Row]:
        """Return the top-ranked symbols for a project.

        By default external library stubs (source_kind='external') are excluded.
        They participate in PageRank to correctly boost project symbols that use
        widely-adopted APIs, but the "important project symbols" output should
        reflect project architecture, not library inventory.  Pass
        ``include_external=True`` to include them.
        """
        extra = "" if include_external else "AND source_kind != 'external'"
        return self.connection.execute(
            f"""
            SELECT *
            FROM symbols
            WHERE project_id = ?
              AND kind NOT IN ('File', 'Artifact', 'ArtifactConfig', 'ArtifactSection', 'Section', 'Module', 'Unknown')
              AND global_rank > 0
              {extra}
            ORDER BY global_rank DESC, display_name ASC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()

    def get_query_cache(self, project_id: str, query_hash: str) -> str | None:
        row = self.connection.execute(
            """
            SELECT response_json
            FROM query_cache
            WHERE project_id = ? AND query_hash = ?
            """,
            (project_id, query_hash),
        ).fetchone()
        return None if row is None else str(row["response_json"])

    def store_query_cache(
        self, project_id: str, query_hash: str, payload: dict
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO query_cache(project_id, query_hash, response_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (project_id, query_hash, json.dumps(payload), _utcnow()),
        )
        self.connection.commit()

    def get_file(self, project_id: str, relative_path: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT *
            FROM files
            WHERE project_id = ? AND relative_path = ?
            """,
            (project_id, relative_path),
        ).fetchone()

    def get_tree_score_rows(self, project_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT
                f.relative_path,
                f.source_kind,
                f.artifact_name,
                COALESCE(symbol_stats.rank_sum, 0.0) AS rank_sum,
                COALESCE(symbol_stats.rank_max, 0.0) AS rank_max,
                COALESCE(symbol_stats.useful_symbol_count, 0) AS useful_symbol_count,
                COALESCE(chunk_stats.chunk_count, 0) AS chunk_count
            FROM files f
            LEFT JOIN (
                SELECT
                    relative_path,
                    SUM(
                        CASE
                            WHEN kind NOT IN ('File', 'Artifact', 'Section', 'Module', 'Unknown')
                            THEN global_rank
                            ELSE 0.0
                        END
                    ) AS rank_sum,
                    MAX(
                        CASE
                            WHEN kind NOT IN ('File', 'Artifact', 'Section', 'Module', 'Unknown')
                            THEN global_rank
                            ELSE 0.0
                        END
                    ) AS rank_max,
                    COUNT(
                        CASE
                            WHEN kind NOT IN ('File', 'Artifact', 'Section', 'Module', 'Unknown')
                            THEN symbol_id
                            ELSE NULL
                        END
                    ) AS useful_symbol_count
                FROM symbols
                WHERE project_id = ?
                GROUP BY relative_path
            ) AS symbol_stats
                ON symbol_stats.relative_path = f.relative_path
            LEFT JOIN (
                SELECT
                    relative_path,
                    COUNT(*) AS chunk_count
                FROM chunks
                WHERE project_id = ?
                GROUP BY relative_path
            ) AS chunk_stats
                ON chunk_stats.relative_path = f.relative_path
            WHERE f.project_id = ?
            ORDER BY f.relative_path ASC
            """,
            (project_id, project_id, project_id),
        ).fetchall()
