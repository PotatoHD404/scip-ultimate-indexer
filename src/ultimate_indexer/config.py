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
}


@dataclass(slots=True)
class Settings:
    project_root: Path
    state_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    visuals_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    scip_cache_path: Path = field(init=False)
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "auto")
    )
    model_repo_id: str = field(
        default_factory=lambda: os.getenv(
            "ULTIMATE_INDEXER_MODEL_REPO_ID",
            "nomic-ai/CodeRankEmbed",
        )
    )
    llama_model_repo_id: str = field(
        default_factory=lambda: os.getenv(
            "ULTIMATE_INDEXER_LLAMA_MODEL_REPO_ID",
            "lmstudio-community/nomic-embed-code-GGUF",
        )
    )
    model_filename: str = field(
        default_factory=lambda: os.getenv(
            "ULTIMATE_INDEXER_MODEL_FILENAME",
            "nomic-embed-code-Q4_K_M.gguf",
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
    edge_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_EDGE_WEIGHTS.copy())
    max_chunk_lines: int = 120
    chunk_overlap: int = 20
    query_cache_ttl_seconds: int = 60 * 60
    extra_extensions: set[str] = field(
        default_factory=lambda: parse_extra_extensions(os.getenv("EXTRA_EXTENSIONS"))
    )

    def __post_init__(self) -> None:
        self.project_root = self.project_root.resolve()
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
