from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .embeddings import hash_text
from .models import ChunkRecord, EdgeRecord, FileRecord, QueryChunkHit, SymbolRecord


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
    ) -> None:
        files = list(files)
        symbols = list(symbols)
        edges = list(edges)
        chunks = list(chunks)
        keep_paths = {item.relative_path for item in files}
        with self.connection:
            self.connection.execute("DELETE FROM edges WHERE project_id = ?", (project_id,))
            self.connection.execute("DELETE FROM chunk_fts WHERE project_id = ?", (project_id,))
            self.connection.execute("DELETE FROM chunks WHERE project_id = ?", (project_id,))
            self.connection.execute("DELETE FROM symbols WHERE project_id = ?", (project_id,))
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
        return np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"]).copy()

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
            np.frombuffer(row["embedding"], dtype=np.float32, count=row["embedding_dim"]).copy()
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

    def search_bm25(self, project_id: str, query: str, limit: int) -> list[QueryChunkHit]:
        tokens = [token for token in query.replace("/", " ").replace(".", " ").split() if token]
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
        with self.connection:
            self.connection.execute(
                "UPDATE symbols SET global_rank = 0.0 WHERE project_id = ?",
                (project_id,),
            )
            for symbol_id, score in ranks.items():
                self.connection.execute(
                    """
                    UPDATE symbols
                    SET global_rank = ?
                    WHERE project_id = ? AND symbol_id = ?
                    """,
                    (score, project_id, symbol_id),
                )

    def get_top_symbols(self, project_id: str, limit: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM symbols
            WHERE project_id = ?
              AND kind NOT IN ('File', 'Artifact', 'Section', 'Module', 'Unknown')
              AND global_rank > 0
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

    def store_query_cache(self, project_id: str, query_hash: str, payload: dict) -> None:
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
