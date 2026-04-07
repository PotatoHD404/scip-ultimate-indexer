"""Database connection module.

Provides database connection pooling and query execution utilities.
"""

import sqlite3
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class DatabasePool:
    """Connection pool for database operations."""

    def __init__(self, db_path: str, pool_size: int = 5):
        """Initialize connection pool."""
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections: List[sqlite3.Connection] = []
        self._initialized = False

    def initialize(self) -> None:
        """Initialize connection pool."""
        if self._initialized:
            return
        for _ in range(self.pool_size):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connections.append(conn)
        self._initialized = True

    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        if not self._connections:
            raise RuntimeError("Pool not initialized")
        conn = self._connections.pop(0)
        try:
            yield conn
        finally:
            self._connections.append(conn)

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close_all(self) -> None:
        """Close all connections in pool."""
        for conn in self._connections:
            conn.close()
        self._connections.clear()


def create_database_pool(db_path: str, size: int = 5) -> DatabasePool:
    """Factory function to create database pool."""
    pool = DatabasePool(db_path, size)
    pool.initialize()
    return pool
