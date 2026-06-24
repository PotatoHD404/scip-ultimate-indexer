from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

import numpy as np

from . import hyde, reranker
from .config import _env_bool, _env_float, _env_int
from .model_providers import resolve_hyde_generator, resolve_rerank_provider
from .embeddings import EmbeddingProvider, cosine_similarity, generate_query_embedding
from .models import FileGroup, QueryChunkHit, RankedSymbol
from .pagerank import weighted_pagerank
from .ranking_rules import is_queryable_symbol
from .storage import Storage


@dataclass(slots=True)
class QueryConfig:
    """Query-time retrieval feature flags (Tier-1 search upgrades)."""

    # HyDE: expand NL queries with a hypothetical code snippet before embedding.
    enable_hyde: bool = field(default_factory=lambda: _env_bool("ULTIMATE_INDEXER_ENABLE_HYDE", True))
    hyde_blend: float = field(default_factory=lambda: _env_float("ULTIMATE_INDEXER_HYDE_BLEND", 0.5))
    # Two-stage rerank of fused candidates.
    enable_reranker: bool = field(default_factory=lambda: _env_bool("ULTIMATE_INDEXER_ENABLE_RERANKER", True))
    rerank_top_k: int = field(default_factory=lambda: _env_int("ULTIMATE_INDEXER_RERANK_TOP_K", 30))
    rerank_blend: float = field(default_factory=lambda: _env_float("ULTIMATE_INDEXER_RERANK_BLEND", 0.5))
    # Weight of git co-change neighbours when the caller supplies focus files.
    cochange_weight: float = field(default_factory=lambda: _env_float("ULTIMATE_INDEXER_COCHANGE_WEIGHT", 0.5))
    # Optional model-backed variants. When an endpoint+model is configured the
    # cross-encoder rerank / LLM HyDE is used; otherwise the deterministic
    # feature reranker / template HyDE is used (and any runtime error falls back).
    rerank_endpoint: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_RERANK_API_ENDPOINT"))
    rerank_model: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_RERANK_API_MODEL"))
    rerank_api_key: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_RERANK_API_KEY"))
    hyde_endpoint: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_HYDE_API_ENDPOINT"))
    hyde_model: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_HYDE_API_MODEL"))
    hyde_api_key: str | None = field(default_factory=lambda: os.getenv("ULTIMATE_INDEXER_HYDE_API_KEY"))
    hyde_max_tokens: int = field(default_factory=lambda: _env_int("ULTIMATE_INDEXER_HYDE_MAX_TOKENS", 256))

DEPRECATED_KINDS = {
    "Variable",
    "Field",
    "Parameter",
}

DEPENDENCY_EDGE_TYPES = {
    "calls",
    "imports",
    "implements",
    "inherits",
    "references",
    "type",
    "uses",
}
COMPOSITIONAL_EDGE_TYPES = {
    "contains",
}
TYPE_CONTAINER_KINDS = {
    "class",
    "struct",
    "interface",
    "trait",
    "enum",
    "typealias",
    "type",
}
TYPE_MEMBER_KINDS = {
    "method",
    "field",
    "property",
    "constant",
    "const",
}
KIND_BOOSTS = {
    "struct": 1.4,
    "interface": 1.5,
    "class": 1.4,
    "trait": 1.4,
    "typealias": 1.3,
    "enum": 1.3,
    "type": 1.3,
    "constant": 1.25,
    "const": 1.25,
    "property": 0.85,
    "field": 0.8,
    "variable": 0.75,
    "function": 1.0,
    "method": 1.0,
}
QUERY_CACHE_VERSION = 3
DOCUMENTATION_SOURCE_KIND = "documentation"
SearchScope = Literal["all", "code", "docs"]


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    positive = {key: max(0.0, value) for key, value in values.items()}
    max_value = max(positive.values())
    if max_value <= 0:
        return {key: 0.0 for key in values}
    return {key: value / max_value for key, value in positive.items()}


def _symbol_kind(row: object) -> str:
    return str(row["kind"]).replace("_", "").replace("-", "").lower()


def _is_exported_symbol(row: object) -> bool:
    display_name = str(row["display_name"]).strip()
    if not display_name:
        return False
    if display_name.startswith("_"):
        return False
    first = display_name[0]
    return first.isupper()


def apply_kind_boost(row: object, score: float) -> float:
    multiplier = KIND_BOOSTS.get(_symbol_kind(row), 1.0)
    if _is_exported_symbol(row):
        multiplier *= 1.1
    return score * multiplier


def _rankable_query_rows(symbol_rows: dict[str, object]) -> dict[str, object]:
    return {
        symbol_id: row
        for symbol_id, row in symbol_rows.items()
        if is_queryable_symbol(str(row["relative_path"]), str(row["kind"]))
    }


def _result_source_bucket(row: object | None, fallback_kind: str) -> str:
    if row is not None:
        source_kind = str(row["source_kind"] or "").strip()
        if source_kind == DOCUMENTATION_SOURCE_KIND:
            return DOCUMENTATION_SOURCE_KIND
        if str(row["kind"]) == "Document":
            return DOCUMENTATION_SOURCE_KIND
    if fallback_kind == "Document":
        return DOCUMENTATION_SOURCE_KIND
    return "code"


def _scope_allows_bucket(scope: SearchScope, bucket: str) -> bool:
    if scope == "all":
        return True
    if scope == "docs":
        return bucket == DOCUMENTATION_SOURCE_KIND
    return bucket != DOCUMENTATION_SOURCE_KIND


def dependency_ordered_pagerank(
    symbol_rows: dict[str, object],
    edge_rows: list[object],
    *,
    personalization: dict[str, float] | None = None,
    alpha: float = 0.85,
) -> dict[str, float]:
    rankable_rows = _rankable_query_rows(symbol_rows)
    rankable_ids = set(rankable_rows)
    if not rankable_ids:
        return {}

    # ------------------------------------------------------------------
    # Edge promotion: when a non-rankable node (Variable, Field, Parameter,
    # Unknown …) appears in an edge, walk its enclosing_symbol_id chain
    # until we reach a rankable ancestor.  This means type references
    # through parameters and struct fields contribute to the rank of
    # their containing function/struct rather than being silently dropped.
    # ------------------------------------------------------------------
    _ancestor_cache: dict[str, str | None] = {}

    def _resolve_ancestor(sid: str) -> str | None:
        """Return the nearest rankable ancestor of *sid* (or *sid* itself)."""
        if sid in _ancestor_cache:
            return _ancestor_cache[sid]
        if sid in rankable_ids:
            _ancestor_cache[sid] = sid
            return sid
        # Walk the enclosing chain — guard against cycles.
        current = sid
        chain: list[str] = []
        while current not in rankable_ids:
            if current in _ancestor_cache:
                resolved = _ancestor_cache[current]
                # Back-fill the entire chain so future lookups are O(1).
                for chained in chain:
                    _ancestor_cache[chained] = resolved
                _ancestor_cache[sid] = resolved
                return resolved
            chain.append(current)
            row = symbol_rows.get(current)
            if row is None:
                for chained in chain:
                    _ancestor_cache[chained] = None
                _ancestor_cache[sid] = None
                return None
            parent = str(row["enclosing_symbol_id"] or "")
            if not parent or parent == current:
                for chained in chain:
                    _ancestor_cache[chained] = None
                _ancestor_cache[sid] = None
                return None
            current = parent
        for chained in chain:
            _ancestor_cache[chained] = current
        _ancestor_cache[sid] = current
        return current

    reoriented_edges: list[tuple[str, str, float]] = []

    def _add_weight(source: str, target: str, weight: float) -> None:
        if weight <= 0:
            return
        reoriented_edges.append((source, target, weight))

    for edge in edge_rows:
        raw_source = str(edge["source_symbol_id"])
        raw_target = str(edge["target_symbol_id"])

        source = _resolve_ancestor(raw_source)
        target = _resolve_ancestor(raw_target)

        # Drop if either end has no rankable ancestor, or promotion
        # collapsed both ends to the same node (self-loop adds no information).
        if source is None or target is None or source == target:
            continue

        edge_type = str(edge["edge_type"])
        base_weight = float(edge["weight"])
        if base_weight <= 0:
            continue

        # Apply a mild discount when either endpoint was promoted so that
        # indirect relationships count slightly less than direct ones.
        was_promoted = raw_source != source or raw_target != target
        promotion_factor = 0.75 if was_promoted else 1.0

        if edge_type in DEPENDENCY_EDGE_TYPES:
            w = base_weight * promotion_factor
            # Rank flows caller → callee (callee is "depended on").
            # Tiny backflow (2 %) keeps graph connected but does not let callers
            # accumulate meaningful rank from what they use — only being called
            # by many things matters for rank.
            _add_weight(source, target, w)
            _add_weight(target, source, w * 0.02)
            continue
        if edge_type in COMPOSITIONAL_EDGE_TYPES:
            source_kind = _symbol_kind(rankable_rows[source])
            target_kind = _symbol_kind(rankable_rows[target])
            # For `Container --contains--> Member` edges, asymmetric weights:
            # members' usage reflects on their owning type (strong member→container),
            # while the container distributes only a tiny signal downward so that
            # large containers (e.g. test suites with 60+ methods) don't inflate
            # their own rank purely by containing many low-rank items.
            if source_kind in TYPE_CONTAINER_KINDS and target_kind in TYPE_MEMBER_KINDS:
                container_w = base_weight * 0.85 * promotion_factor
                _add_weight(target, source, container_w)        # member → container (strong)
                _add_weight(source, target, container_w * 0.05) # container → member (very weak)
            elif target_kind in TYPE_CONTAINER_KINDS and source_kind in TYPE_MEMBER_KINDS:
                container_w = base_weight * 0.85 * promotion_factor
                _add_weight(source, target, container_w)        # member → container (strong)
                _add_weight(target, source, container_w * 0.05) # container → member (very weak)
            else:
                w = base_weight * 0.20 * promotion_factor
                _add_weight(source, target, w)
                _add_weight(target, source, w)
            continue
        w = base_weight * promotion_factor
        _add_weight(source, target, w * 0.5)
        _add_weight(target, source, w * 0.02)

    filtered_personalization = None
    if personalization:
        filtered_personalization = {
            symbol_id: value
            for symbol_id, value in personalization.items()
            if symbol_id in rankable_ids and value > 0
        }

    return weighted_pagerank(
        nodes=sorted(rankable_ids),
        edges=reoriented_edges,
        alpha=alpha,
        personalization=filtered_personalization,
    )


class QueryEngine:
    def __init__(
        self,
        storage: Storage,
        provider: EmbeddingProvider,
        config: QueryConfig | None = None,
    ) -> None:
        self.storage = storage
        self.provider = provider
        self.config = config or QueryConfig()
        self._rerank_provider = resolve_rerank_provider(
            endpoint=self.config.rerank_endpoint,
            model=self.config.rerank_model,
            api_key=self.config.rerank_api_key,
        )
        self._hyde_generator = resolve_hyde_generator(
            endpoint=self.config.hyde_endpoint,
            model=self.config.hyde_model,
            api_key=self.config.hyde_api_key,
            max_tokens=self.config.hyde_max_tokens,
        )
        self._vector_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}
        self._function_metadata_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}
        self._function_body_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}

    def _cached_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        signature = self.storage.get_project_signature(project_id)
        cached = self._vector_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_chunk_vectors(project_id, self.provider.model_id)
        self._vector_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _cached_function_metadata_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        """Load cached function metadata vectors."""
        signature = self.storage.get_project_signature(project_id)
        cached = self._function_metadata_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_function_metadata_vectors(project_id, self.provider.model_id)
        self._function_metadata_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _cached_function_body_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        """Load cached function body chunk vectors."""
        signature = self.storage.get_project_signature(project_id)
        cached = self._function_body_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_function_body_vectors(project_id, self.provider.model_id)
        self._function_body_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _embed_cached(self, cache_text: str, gen_text: str) -> np.ndarray | None:
        vec = self.storage.get_or_create_embedding(self.provider.model_id, cache_text)
        if vec is not None:
            return vec
        try:
            vec = generate_query_embedding(self.provider, gen_text)
        except Exception:
            # Keep retrieval usable when the preferred dense backend is missing at
            # query time. BM25 and cached hash/local results can still answer.
            return None
        self.storage.store_embedding(self.provider.model_id, cache_text, vec)
        return vec

    def _query_vector(self, query: str) -> np.ndarray | None:
        base = self._embed_cached(f"query::{query}", query)
        if base is None:
            return None
        # HyDE: for natural-language queries, blend in the embedding of a
        # hypothetical code snippet so the query lands closer to real code.
        if not (
            self.config.enable_hyde
            and self.config.hyde_blend > 0.0
            and hyde.looks_like_natural_language(query)
        ):
            return base
        hypo_text = self._hypothetical_text(query)
        hypo = self._embed_cached(
            f"hyde::{sha256(hypo_text.encode('utf-8')).hexdigest()}", hypo_text
        )
        if hypo is None or hypo.shape != base.shape:
            return base
        blend = min(max(self.config.hyde_blend, 0.0), 1.0)

        def _unit(vec: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else vec

        combined = (1.0 - blend) * _unit(base) + blend * _unit(hypo)
        return combined.astype(np.float32)

    def _hypothetical_text(self, query: str) -> str:
        """The HyDE hypothetical document: LLM-generated when configured, else a
        deterministic template. Any generation error degrades to the template."""
        if self._hyde_generator is not None:
            try:
                generated = self._hyde_generator.generate(query)
            except Exception:
                generated = ""
            if generated.strip():
                return generated
        return hyde.hypothetical_code(query)

    def _function_metadata_dense_hits(
        self,
        project_id: str,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Search function metadata using dense embeddings."""
        matrix, rows = self._cached_function_metadata_vectors(project_id)
        if matrix.size == 0:
            return []

        query_vector = self._query_vector(query)
        if query_vector is None:
            return []
        if matrix.ndim != 2 or matrix.shape[1] != query_vector.shape[0]:
            return []
        
        scores = cosine_similarity(query_vector, matrix)
        order = np.argsort(scores)[::-1][:limit]
        
        hits: list[dict] = []
        for index in order:
            row = rows[int(index)]
            score = float(scores[int(index)])
            if score <= 0:
                continue
            hits.append({
                "symbol_id": str(row["symbol_id"]),
                "relative_path": str(row["relative_path"]),
                "display_name": str(row["display_name"]),
                "kind": str(row["kind"]),
                "fully_qualified_name": str(row["fully_qualified_name"]),
                "signature": str(row["signature"]),
                "docstring": str(row["docstring"]),
                "metadata_content": str(row["metadata_content"]),
                "score": score,
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
            })
        return hits

    def _function_body_dense_hits(
        self,
        project_id: str,
        query: str,
        limit: int,
    ) -> list[dict]:
        """Search function body chunks using dense embeddings."""
        matrix, rows = self._cached_function_body_vectors(project_id)
        if matrix.size == 0:
            return []

        query_vector = self._query_vector(query)
        if query_vector is None:
            return []
        if matrix.ndim != 2 or matrix.shape[1] != query_vector.shape[0]:
            return []
        
        scores = cosine_similarity(query_vector, matrix)
        order = np.argsort(scores)[::-1][:limit]
        
        hits: list[dict] = []
        for index in order:
            row = rows[int(index)]
            score = float(scores[int(index)])
            if score <= 0:
                continue
            hits.append({
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
                "content": str(row["content"]),
                "score": score,
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
            })
        return hits

    def _dense_hits(self, project_id: str, query: str, limit: int) -> list[QueryChunkHit]:
        matrix, rows = self._cached_vectors(project_id)
        if matrix.size == 0:
            return []
        query_vector = self._query_vector(query)
        if query_vector is None:
            return []
        if matrix.ndim != 2 or matrix.shape[1] != query_vector.shape[0]:
            return []
        scores = cosine_similarity(query_vector, matrix)
        order = np.argsort(scores)[::-1][:limit]
        hits: list[QueryChunkHit] = []
        for index in order:
            row = rows[int(index)]
            hits.append(
                QueryChunkHit(
                    chunk_id=str(row["chunk_id"]),
                    relative_path=str(row["relative_path"]),
                    symbol_id=str(row["symbol_id"]),
                    symbol_name=str(row["symbol_name"]),
                    score=float(scores[int(index)]),
                    content=str(row["content"]),
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                )
            )
        return hits

    def _deserialize_groups(self, payload: str) -> list[FileGroup]:
        raw = json.loads(payload)
        groups: list[FileGroup] = []
        for item in raw:
            groups.append(
                FileGroup(
                    relative_path=item["relative_path"],
                    score=float(item["score"]),
                    symbols=[
                        RankedSymbol(
                            symbol_id=symbol["symbol_id"],
                            relative_path=symbol["relative_path"],
                            display_name=symbol["display_name"],
                            kind=symbol["kind"],
                            score=float(symbol["score"]),
                            signature=symbol["signature"],
                            docstring=symbol["docstring"],
                            snippet=symbol["snippet"],
                        )
                        for symbol in item["symbols"]
                    ],
                )
            )
        return groups

    def _serialize_groups(self, groups: list[FileGroup]) -> list[dict]:
        payload: list[dict] = []
        for group in groups:
            payload.append(
                {
                    "relative_path": group.relative_path,
                    "score": group.score,
                    "symbols": [
                        {
                            "symbol_id": symbol.symbol_id,
                            "relative_path": symbol.relative_path,
                            "display_name": symbol.display_name,
                            "kind": symbol.kind,
                            "score": symbol.score,
                            "signature": symbol.signature,
                            "docstring": symbol.docstring,
                            "snippet": symbol.snippet,
                        }
                        for symbol in group.symbols
                    ],
                }
            )
        return payload

    def _canonical_symbol_id(self, symbol_rows: dict[str, object], symbol_id: str) -> str:
        current_id = symbol_id
        seen: set[str] = set()
        while current_id not in seen:
            seen.add(current_id)
            row = symbol_rows.get(current_id)
            if row is None:
                return current_id
            kind = str(row["kind"])
            enclosing_symbol_id = str(row["enclosing_symbol_id"] or "")
            if not enclosing_symbol_id:
                return current_id
            parent = symbol_rows.get(enclosing_symbol_id)
            if parent is None:
                return current_id
            parent_kind = str(parent["kind"])
            if kind in {"ArtifactConfig", "ArtifactSection"} and parent_kind == "Artifact":
                current_id = enclosing_symbol_id
                continue
            if kind == "Method" and parent_kind in {"Interface", "Struct", "Class", "TypeAlias", "Trait", "Enum"}:
                current_id = enclosing_symbol_id
                continue
            if parent_kind in {"File", "Module", "Section", "Unknown"}:
                return current_id
            if kind not in {"Field", "Parameter", "Variable"}:
                return current_id
            current_id = enclosing_symbol_id
        return symbol_id

    def _symbol_scores(
        self,
        hits: list[tuple[QueryChunkHit, str]],
    ) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for hit, symbol_id in hits:
            scores[symbol_id] = max(scores.get(symbol_id, 0.0), float(hit.score))
        return scores

    def _group_source_bucket(
        self,
        symbol_rows: dict[str, object],
        group: FileGroup,
    ) -> str:
        for symbol in group.symbols:
            row = symbol_rows.get(symbol.symbol_id)
            bucket = _result_source_bucket(row, symbol.kind)
            if bucket == DOCUMENTATION_SOURCE_KIND:
                return bucket
        return "code"

    def _select_groups(
        self,
        ordered_groups: list[FileGroup],
        symbol_rows: dict[str, object],
        *,
        limit: int,
    ) -> list[FileGroup]:
        if limit <= 0:
            return []
        if len(ordered_groups) <= limit or limit == 1:
            return ordered_groups[:limit]

        selected = list(ordered_groups[:limit])
        selected_buckets = {
            self._group_source_bucket(symbol_rows, group)
            for group in selected
        }
        all_buckets = {
            self._group_source_bucket(symbol_rows, group)
            for group in ordered_groups
        }

        # Keep the highest-ranked mix by default, but force one documentation file
        # into the limited result set when the query matches both code and docs.
        for missing_bucket in sorted(all_buckets - selected_buckets):
            replacement = next(
                (
                    group
                    for group in ordered_groups[limit:]
                    if self._group_source_bucket(symbol_rows, group) == missing_bucket
                ),
                None,
            )
            if replacement is None:
                continue
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if self._group_source_bucket(symbol_rows, selected[index]) != missing_bucket
                ),
                None,
            )
            if replace_index is None:
                continue
            selected[replace_index] = replacement
            selected_buckets.add(missing_bucket)

        return sorted(selected, key=lambda item: (-item.score, item.relative_path))

    def _merge_scope_groups(
        self,
        code_groups: list[FileGroup],
        doc_groups: list[FileGroup],
        *,
        limit: int,
    ) -> list[FileGroup]:
        if limit <= 0:
            return []
        by_path: dict[str, FileGroup] = {}
        fused_scores: dict[str, float] = defaultdict(float)
        rrf_k = 60.0

        for groups in (code_groups, doc_groups):
            for rank, group in enumerate(groups, start=1):
                fused_scores[group.relative_path] += 1.0 / (rrf_k + rank)
                existing = by_path.get(group.relative_path)
                if existing is None:
                    by_path[group.relative_path] = FileGroup(
                        relative_path=group.relative_path,
                        score=group.score,
                        symbols=list(group.symbols),
                    )
                    continue
                existing.score = max(existing.score, group.score)
                known_ids = {symbol.symbol_id for symbol in existing.symbols}
                for symbol in group.symbols:
                    if symbol.symbol_id not in known_ids:
                        existing.symbols.append(symbol)
                        known_ids.add(symbol.symbol_id)
                existing.symbols.sort(key=lambda item: (-item.score, item.display_name))

        ordered_paths = sorted(
            fused_scores,
            key=lambda path: (-fused_scores[path], path),
        )
        merged: list[FileGroup] = []
        for path in ordered_paths[:limit]:
            group = by_path[path]
            group.score = fused_scores[path]
            merged.append(group)
        return merged

    def _apply_focus_personalization(
        self,
        project_id: str,
        rankable_rows: dict[str, object],
        seed_personalization: dict[str, float],
        focus_paths: tuple[str, ...],
    ) -> None:
        """Add focus files (and their git co-change neighbours) to the PPR seeds."""
        focus_set = set(focus_paths)
        for symbol_id, row in rankable_rows.items():
            if str(row["relative_path"]) in focus_set:
                seed_personalization[symbol_id] = max(
                    seed_personalization.get(symbol_id, 0.0), 1.0
                )
        neighbors = self.storage.get_cochange_neighbors(project_id, focus_paths)
        if not neighbors:
            return
        weight_scale = max(self.config.cochange_weight, 0.0)
        for symbol_id, row in rankable_rows.items():
            coupling = neighbors.get(str(row["relative_path"]))
            if coupling is None:
                continue
            value = coupling * weight_scale
            if value > 0:
                seed_personalization[symbol_id] = max(
                    seed_personalization.get(symbol_id, 0.0), value
                )

    def _rerank(self, query: str, ranked: list[RankedSymbol]) -> list[RankedSymbol]:
        # The cheap feature reranker scores every candidate; the model
        # (cross-encoder) reranker is rate/cost-limited to the top-K and the rest
        # is kept strictly below it.
        use_model = self._rerank_provider is not None
        if use_model:
            cap = max(self.config.rerank_top_k, 1)
            head, tail = ranked[:cap], ranked[cap:]
        else:
            head, tail = ranked, []
        items = [
            reranker.RerankItem(
                id=symbol.symbol_id,
                name=symbol.display_name,
                signature=symbol.signature,
                docstring=symbol.docstring,
                snippet=symbol.snippet,
                path=symbol.relative_path,
                kind=symbol.kind,
                stage1=symbol.score,
            )
            for symbol in head
        ]
        scores: dict[str, float] | None = None
        if use_model:
            scores = self._model_rerank_scores(query, head)
        if scores is None:
            scores = reranker.feature_scores(query, items)
        order = reranker.combine(items, scores, blend=self.config.rerank_blend)
        by_id = {symbol.symbol_id: symbol for symbol in head}
        reranked: list[RankedSymbol] = []
        for symbol_id, final in order:
            symbol = by_id[symbol_id]
            symbol.score = final
            reranked.append(symbol)
        if tail:
            floor = min((symbol.score for symbol in reranked), default=0.0)
            tail_max = max((symbol.score for symbol in tail), default=0.0)
            if tail_max > 0.0 and floor > 0.0:
                scale = (floor * 0.99) / tail_max
                for symbol in tail:
                    symbol.score *= scale
        return reranked + tail

    def _model_rerank_scores(
        self, query: str, symbols: list[RankedSymbol]
    ) -> dict[str, float] | None:
        """Per-symbol rerank scores from the cross-encoder, min-max normalised to
        [0,1]; ``None`` on any error so the feature reranker takes over."""
        if self._rerank_provider is None or not symbols:
            return None
        documents = [self._rerank_document_text(symbol) for symbol in symbols]
        try:
            raw = self._rerank_provider.score(query, documents)
        except Exception:
            return None
        if len(raw) != len(symbols):
            return None
        lo, hi = min(raw), max(raw)
        span = hi - lo
        return {
            symbol.symbol_id: ((value - lo) / span if span > 0 else 1.0)
            for symbol, value in zip(symbols, raw)
        }

    @staticmethod
    def _rerank_document_text(symbol: RankedSymbol) -> str:
        parts = [symbol.display_name, symbol.signature, symbol.docstring, symbol.snippet]
        return "\n".join(part for part in parts if part)[:2000]

    def _search_uncached(
        self,
        project_id: str,
        query: str,
        limit: int,
        *,
        scope: SearchScope,
        focus_paths: tuple[str, ...] = (),
    ) -> list[FileGroup]:
        if scope == "all":
            scope_limit = max(limit * 2, 10)
            code_results = self._search_uncached(
                project_id,
                query,
                scope_limit,
                scope="code",
                focus_paths=focus_paths,
            )
            docs_results = self._search_uncached(
                project_id,
                query,
                scope_limit,
                scope="docs",
                focus_paths=focus_paths,
            )
            return self._merge_scope_groups(code_results, docs_results, limit=limit)

        symbol_rows = self.storage.get_symbol_rows(project_id)
        rankable_rows = _rankable_query_rows(symbol_rows)
        all_symbol_scores: dict[str, dict[str, float]] = defaultdict(dict)

        if scope == "code":
            metadata_bm25_hits = self.storage.search_function_metadata_bm25(project_id, query, max(limit * 4, 16))
            for hit in metadata_bm25_hits:
                symbol_id = hit["symbol_id"]
                all_symbol_scores[symbol_id]["metadata_bm25"] = hit["score"]
                all_symbol_scores[symbol_id]["metadata"] = hit

            metadata_dense_hits = self._function_metadata_dense_hits(project_id, query, max(limit * 4, 16))
            for hit in metadata_dense_hits:
                symbol_id = hit["symbol_id"]
                all_symbol_scores[symbol_id]["metadata_dense"] = hit["score"]
                if "metadata" not in all_symbol_scores[symbol_id]:
                    all_symbol_scores[symbol_id]["metadata"] = hit

            body_bm25_hits = self.storage.search_function_body_bm25(project_id, query, max(limit * 6, 24))
            for hit in body_bm25_hits:
                symbol_id = hit["symbol_id"]
                all_symbol_scores[symbol_id]["body_bm25"] = hit["score"]
                if "body_hits" not in all_symbol_scores[symbol_id]:
                    all_symbol_scores[symbol_id]["body_hits"] = []
                all_symbol_scores[symbol_id]["body_hits"].append(hit)

            body_dense_hits = self._function_body_dense_hits(project_id, query, max(limit * 6, 24))
            for hit in body_dense_hits:
                symbol_id = hit["symbol_id"]
                all_symbol_scores[symbol_id]["body_dense"] = hit["score"]
                if "body_hits" not in all_symbol_scores[symbol_id]:
                    all_symbol_scores[symbol_id]["body_hits"] = []
                if not any(h["chunk_id"] == hit["chunk_id"] for h in all_symbol_scores[symbol_id]["body_hits"]):
                    all_symbol_scores[symbol_id]["body_hits"].append(hit)

        for hit in self.storage.search_bm25(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            row = rankable_rows.get(canonical_symbol_id)
            if row is None:
                continue
            bucket = _result_source_bucket(row, str(row["kind"]))
            if not _scope_allows_bucket(scope, bucket):
                continue
            all_symbol_scores[canonical_symbol_id]["chunk_bm25"] = hit.score

        for hit in self._dense_hits(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            row = rankable_rows.get(canonical_symbol_id)
            if row is None:
                continue
            bucket = _result_source_bucket(row, str(row["kind"]))
            if not _scope_allows_bucket(scope, bucket):
                continue
            all_symbol_scores[canonical_symbol_id]["chunk_dense"] = hit.score

        def _blend(a: float, b: float) -> float:
            # Keep the stronger signal dominant (65 %) while still capturing
            # the complementary weaker signal (35 %).  Using plain max() would
            # discard the second source entirely.
            hi, lo = (a, b) if a >= b else (b, a)
            return hi * 0.65 + lo * 0.35

        def _compute_symbol_score(symbol_id: str) -> float:
            scores = all_symbol_scores.get(symbol_id, {})
            metadata_bm25 = scores.get("metadata_bm25", 0.0)
            metadata_dense = scores.get("metadata_dense", 0.0)
            body_bm25 = scores.get("body_bm25", 0.0)
            body_dense = scores.get("body_dense", 0.0)
            chunk_bm25 = scores.get("chunk_bm25", 0.0)
            chunk_dense = scores.get("chunk_dense", 0.0)

            meta_raw = _blend(metadata_bm25, metadata_dense)
            body_raw = _blend(body_bm25, body_dense)
            chunk_raw = _blend(chunk_bm25, chunk_dense)

            metadata_score = meta_raw * 0.40
            body_score = body_raw * 0.30
            chunk_score = chunk_raw * 0.30

            # Bonus when both name/signature and body signals agree — based on
            # pre-weight raw scores so the bonus is meaningful in magnitude.
            dual_bonus = 0.12 * min(meta_raw, body_raw) if meta_raw > 0 and body_raw > 0 else 0.0

            return metadata_score + body_score + chunk_score + dual_bonus

        raw_scores = {sid: _compute_symbol_score(sid) for sid in all_symbol_scores}
        if not raw_scores:
            return []
        max_score = max(raw_scores.values())
        normalized_scores = {sid: score / max_score for sid, score in raw_scores.items()} if max_score > 0 else raw_scores

        candidates: dict[str, float] = {}
        for symbol_id, base_score in normalized_scores.items():
            row = rankable_rows.get(symbol_id)
            if row is None:
                if scope == "code":
                    meta = all_symbol_scores[symbol_id].get("metadata", {})
                    if meta:
                        candidates[symbol_id] = base_score * 1.1
                continue
            bucket = _result_source_bucket(row, str(row["kind"]))
            if not _scope_allows_bucket(scope, bucket):
                continue
            score = apply_kind_boost(row, base_score)
            if score > 0:
                candidates[symbol_id] = score

        max_seed = max(normalized_scores.values()) if normalized_scores else 0
        threshold = max(0.35, max_seed * 0.6)
        seed_personalization = {
            symbol_id: score
            for symbol_id, score in normalized_scores.items()
            if score >= threshold
        }
        # Context-personalized ranking: bias toward the caller's focus files and
        # the files that historically co-change with them (git coupling).
        if focus_paths:
            self._apply_focus_personalization(
                project_id, rankable_rows, seed_personalization, focus_paths
            )
        if seed_personalization:
            # Standard personalized-PageRank damping: 85 % of mass follows the
            # dependency graph, 15 % restarts at the query-seed personalization.
            # Matches the global-rank convention in indexer._global_ranks; query
            # bias comes from the restart vector, not a tiny damping factor.
            ppr_scores = dependency_ordered_pagerank(
                symbol_rows,
                self.storage.get_edges(project_id),
                personalization=seed_personalization,
                alpha=0.85,
            )
            normalized_ppr_scores = _normalize_scores(ppr_scores)
            for symbol_id in candidates:
                ppr = normalized_ppr_scores.get(symbol_id, 0.0)
                candidates[symbol_id] = candidates[symbol_id] * 0.70 + ppr * 0.30

        ranked: list[RankedSymbol] = []
        for symbol_id, score in sorted(candidates.items(), key=lambda item: (-item[1], item[0])):
            row = rankable_rows.get(symbol_id)
            if row:
                ranked.append(
                    RankedSymbol(
                        symbol_id=symbol_id,
                        relative_path=str(row["relative_path"]),
                        display_name=str(row["display_name"]),
                        kind=str(row["kind"]),
                        score=score,
                        signature=str(row["signature"]),
                        docstring=str(row["docstring"]),
                        snippet=str(row["snippet"]),
                    )
                )
                continue
            if scope != "code":
                continue
            meta = all_symbol_scores[symbol_id].get("metadata", {})
            if meta:
                ranked.append(
                    RankedSymbol(
                        symbol_id=symbol_id,
                        relative_path=str(meta.get("relative_path", "")),
                        display_name=str(meta.get("display_name", "")),
                        kind=str(meta.get("kind", "Function")),
                        score=score,
                        signature=str(meta.get("signature", "")),
                        docstring=str(meta.get("docstring", "")),
                        snippet=str(meta.get("metadata_content", ""))[:500],
                    )
                )

        # Two-stage rerank: re-score the fused candidates against the query with a
        # feature reranker and blend with the first-stage score. Promotes exact
        # name/signature matches the noisy first stage may have buried.
        if self.config.enable_reranker and len(ranked) > 1 and query.strip():
            ranked = self._rerank(query, ranked)

        grouped: dict[str, FileGroup] = {}
        for symbol in ranked:
            group = grouped.setdefault(
                symbol.relative_path,
                FileGroup(relative_path=symbol.relative_path, score=0.0),
            )
            group.score = max(group.score, symbol.score)
            if symbol.symbol_id not in {item.symbol_id for item in group.symbols}:
                group.symbols.append(symbol)

        ordered_groups = sorted(grouped.values(), key=lambda item: (-item.score, item.relative_path))
        if scope != "all":
            ordered_groups = [
                group
                for group in ordered_groups
                if _scope_allows_bucket(scope, self._group_source_bucket(symbol_rows, group))
            ]
        results = ordered_groups[:limit]
        for group in results:
            group.symbols.sort(key=lambda item: (-item.score, item.display_name))
        return results

    def search(
        self,
        project_id: str,
        query: str,
        limit: int = 10,
        *,
        scope: SearchScope = "all",
        focus_paths: tuple[str, ...] = (),
    ) -> list[FileGroup]:
        signature = self.storage.get_project_signature(project_id) or ""
        focus_key = ",".join(sorted(focus_paths))
        query_hash = sha256(
            f"{QUERY_CACHE_VERSION}:{signature}:{self.provider.model_id}:{scope}:{query}:{limit}:{focus_key}".encode("utf-8")
        ).hexdigest()
        cached = self.storage.get_query_cache(project_id, query_hash)
        if cached is not None:
            return self._deserialize_groups(cached)
        results = self._search_uncached(project_id, query, limit, scope=scope, focus_paths=focus_paths)
        self.storage.store_query_cache(project_id, query_hash, self._serialize_groups(results))
        return results
