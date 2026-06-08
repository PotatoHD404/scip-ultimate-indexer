from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .constants import parse_extra_extensions


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to *default* on missing/garbage values.

    A malformed numeric env var must not crash every CLI invocation.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
        default_factory=lambda: _env_int("ULTIMATE_INDEXER_EMBEDDING_API_MAX_TOKENS", 0) or None
    )
    embedding_api_batch_size: int = field(
        default_factory=lambda: _env_int("ULTIMATE_INDEXER_EMBEDDING_API_BATCH_SIZE", 32)
    )
    embedding_api_timeout_seconds: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_EMBEDDING_API_TIMEOUT_SECONDS", 30.0)
    )
    embedding_api_max_retries: int = field(
        default_factory=lambda: _env_int("ULTIMATE_INDEXER_EMBEDDING_API_MAX_RETRIES", 3)
    )
    embedding_api_retry_base_delay_ms: int = field(
        default_factory=lambda: _env_int("ULTIMATE_INDEXER_EMBEDDING_API_RETRY_BASE_DELAY_MS", 500)
    )
    edge_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_EDGE_WEIGHTS.copy())
    max_chunk_lines: int = 120
    chunk_overlap: int = 20
    query_cache_ttl_seconds: int = 60 * 60
    extra_extensions: set[str] = field(
        default_factory=lambda: parse_extra_extensions(os.getenv("EXTRA_EXTENSIONS"))
    )
    # --- Retrieval / ranking upgrades (index-time) ---
    # Code<->query vocabulary expansion folded into the lexical (BM25) index.
    enable_query_expansion: bool = field(
        default_factory=lambda: _env_bool("ULTIMATE_INDEXER_ENABLE_EXPANSION", True)
    )
    # Structural context prepended to dense embedding text (kept out of BM25).
    enable_contextual_embeddings: bool = field(
        default_factory=lambda: _env_bool("ULTIMATE_INDEXER_ENABLE_CONTEXTUAL", True)
    )
    # Git-history importance signals (recency/churn) and co-change coupling.
    enable_git_signals: bool = field(
        default_factory=lambda: _env_bool("ULTIMATE_INDEXER_ENABLE_GIT_SIGNALS", True)
    )
    git_history_limit: int = field(
        default_factory=lambda: _env_int("ULTIMATE_INDEXER_GIT_HISTORY_LIMIT", 2000)
    )
    git_half_life_days: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_GIT_HALF_LIFE_DAYS", 90.0)
    )
    # How strongly recency/churn lift a file's symbols' global rank (0 = off).
    git_signal_strength: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_GIT_SIGNAL_STRENGTH", 0.5)
    )
    git_recency_weight: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_GIT_RECENCY_WEIGHT", 0.6)
    )
    git_churn_weight: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_GIT_CHURN_WEIGHT", 0.4)
    )
    # Weight of git co-change neighbours when a query supplies focus files.
    cochange_personalization_weight: float = field(
        default_factory=lambda: _env_float("ULTIMATE_INDEXER_COCHANGE_WEIGHT", 0.5)
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
