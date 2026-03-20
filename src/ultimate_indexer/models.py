from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FileRecord:
    project_id: str
    relative_path: str
    abs_path: str
    language: str
    content_hash: str
    content: str
    source_kind: str = "code"
    artifact_name: str | None = None


@dataclass(slots=True)
class SymbolRecord:
    project_id: str
    symbol_id: str
    scip_symbol: str
    display_name: str
    kind: str
    relative_path: str
    start_line: int
    end_line: int
    signature: str
    docstring: str
    snippet: str
    enclosing_symbol_id: str | None = None
    source_kind: str = "code"


@dataclass(slots=True)
class EdgeRecord:
    project_id: str
    source_symbol_id: str
    target_symbol_id: str
    edge_type: str
    weight: float


@dataclass(slots=True)
class ChunkRecord:
    project_id: str
    chunk_id: str
    relative_path: str
    symbol_id: str
    symbol_name: str
    artifact_name: str | None
    chunk_kind: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    embedding: bytes | None = None
    embedding_dim: int = 0
    embedding_model_id: str = ""


@dataclass(slots=True)
class ArtifactSpec:
    name: str
    path: str
    description: str


@dataclass(slots=True)
class QueryChunkHit:
    chunk_id: str
    relative_path: str
    symbol_id: str
    symbol_name: str
    score: float
    content: str
    start_line: int
    end_line: int


@dataclass(slots=True)
class RankedSymbol:
    symbol_id: str
    relative_path: str
    display_name: str
    kind: str
    score: float
    signature: str
    docstring: str
    snippet: str


@dataclass(slots=True)
class FileGroup:
    relative_path: str
    score: float
    symbols: list[RankedSymbol] = field(default_factory=list)


@dataclass(slots=True)
class IndexSummary:
    project_id: str
    indexed_files: int
    indexed_symbols: int
    indexed_edges: int
    indexed_chunks: int
    reused_files: int
    artifact_files: int
    documentation_files: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IndexProgress:
    stage: str
    stage_index: int
    stage_total: int
    completed: int = 0
    total: int = 0
    unit: str = "items"
    detail: str = ""


@dataclass(slots=True)
class TreeScoreNode:
    name: str
    relative_path: str
    node_type: str
    raw_score: float
    score: float = 0.0
    useful_symbol_count: int = 0
    chunk_count: int = 0
    source_kind: str = "code"
    children: list["TreeScoreNode"] = field(default_factory=list)
