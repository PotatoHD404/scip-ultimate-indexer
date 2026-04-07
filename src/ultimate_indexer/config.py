from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .constants import parse_extra_extensions


DEFAULT_EDGE_WEIGHTS = {
    "contains": 0.55,
    "calls": 1.0,
    "uses": 0.75,
    "type": 1.15,
    "imports": 0.45,
    "references": 0.70,
    "implements": 0.95,
    "inherits": 0.90,
    # Documentation-specific edge weights
    "cross_file": 1.0,
    "cross_anchor": 1.0,
    "intra_anchor": 0.5,
    "hierarchy": 0.3,
    "openapi_ref": 0.8,
    "openapi_tag": 0.8,
    "sequence": 0.15,
}


@dataclass(slots=True)
class Settings:
    project_root: Path
    state_dir_override: Path | None = None
    state_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    visuals_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    scip_cache_path: Path = field(init=False)
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "auto")
    )
    model_path: str | None = field(
        default_factory=lambda: os.getenv(
            "ULTIMATE_INDEXER_MODEL_PATH",
        )
    )
    model_filename: str = field(
        default_factory=lambda: os.getenv(
            "ULTIMATE_INDEXER_MODEL_FILENAME",
            "coderankembed-q8_0.gguf",
        )
    )
    model_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ULTIMATE_INDEXER_MODEL_CACHE_DIR",
                str(Path.home() / ".cache" / "ultimate_indexer" / "models"),
            )
        )
    )
    # API embedding provider settings
    embedding_api_key: str | None = field(
        default_factory=lambda: os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_KEY")
    )
    embedding_api_endpoint: str | None = field(
        default_factory=lambda: os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_ENDPOINT")
    )
    embedding_api_model: str | None = field(
        default_factory=lambda: os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_MODEL")
    )
    embedding_api_max_tokens: int | None = field(
        default_factory=lambda: int(os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_MAX_TOKENS", "0")) or None
    )
    embedding_api_batch_size: int = field(
        default_factory=lambda: int(os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_BATCH_SIZE", "32"))
    )
    embedding_api_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_TIMEOUT_SECONDS", "30.0"))
    )
    embedding_api_max_retries: int = field(
        default_factory=lambda: int(os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_MAX_RETRIES", "3"))
    )
    embedding_api_retry_base_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("ULTIMATE_INDEXER_EMBEDDING_API_RETRY_BASE_DELAY_MS", "500"))
    )
    edge_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_EDGE_WEIGHTS.copy())
    max_chunk_lines: int = 120
    chunk_overlap: int = 20
    query_cache_ttl_seconds: int = 60 * 60
    extra_extensions: set[str] = field(
        default_factory=lambda: parse_extra_extensions(os.getenv("EXTRA_EXTENSIONS"))
    )

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
        if self.state_dir_override is not None:
            self.state_dir = self.state_dir_override.expanduser().resolve()
        else:
            self.state_dir = self.project_root / ".ultimate_indexer"
        self.database_path = self.state_dir / "index.sqlite3"
        self.visuals_dir = self.state_dir / "visuals"
        self.cache_dir = self.state_dir / "cache"
        self.scip_cache_path = self.cache_dir / "project.scip"

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.visuals_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
