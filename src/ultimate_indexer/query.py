from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256

import numpy as np

from .embeddings import EmbeddingProvider, cosine_similarity, generate_query_embedding
from .models import FileGroup, QueryChunkHit, RankedSymbol
from .pagerank import weighted_pagerank
from .ranking_rules import is_queryable_symbol
from .storage import Storage

DEPENDENCY_EDGE_TYPES = {
    "calls",
    "imports",
    "references",
    "type",
    "uses",
}
COMPOSITIONAL_EDGE_TYPES = {
    "contains",
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


def dependency_ordered_pagerank(
    symbol_rows: dict[str, object],
    edge_rows: list[object],
    *,
    personalization: dict[str, float] | None = None,
    alpha: float = 0.15,
) -> dict[str, float]:
    rankable_rows = _rankable_query_rows(symbol_rows)
    rankable_ids = set(rankable_rows)
    if not rankable_ids:
        return {}

    reoriented_edges: list[tuple[str, str, float]] = []

    def _add_weight(source: str, target: str, weight: float) -> None:
        if weight <= 0:
            return
        reoriented_edges.append((source, target, weight))

    for edge in edge_rows:
        source = str(edge["source_symbol_id"])
        target = str(edge["target_symbol_id"])
        if source not in rankable_ids or target not in rankable_ids:
            continue
        edge_type = str(edge["edge_type"])
        base_weight = float(edge["weight"])
        if base_weight <= 0:
            continue
        if edge_type in DEPENDENCY_EDGE_TYPES:
            _add_weight(target, source, base_weight)
            _add_weight(source, target, base_weight * 0.15)
            continue
        if edge_type in COMPOSITIONAL_EDGE_TYPES:
            _add_weight(source, target, base_weight * 0.85)
            _add_weight(target, source, base_weight * 0.85)
            continue
        _add_weight(target, source, base_weight * 0.5)
        _add_weight(source, target, base_weight * 0.15)

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
    def __init__(self, storage: Storage, provider: EmbeddingProvider) -> None:
        self.storage = storage
        self.provider = provider
        self._vector_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}

    def _cached_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        signature = self.storage.get_project_signature(project_id)
        cached = self._vector_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_chunk_vectors(project_id)
        self._vector_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _dense_hits(self, project_id: str, query: str, limit: int) -> list[QueryChunkHit]:
        matrix, rows = self._cached_vectors(project_id)
        if matrix.size == 0:
            return []
        query_cache_key = f"query::{query}"
        query_vector = self.storage.get_or_create_embedding(self.provider.model_id, query_cache_key)
        if query_vector is None:
            query_vector = generate_query_embedding(self.provider, query)
            self.storage.store_embedding(self.provider.model_id, query_cache_key, query_vector)
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
            if parent_kind in {"File", "Module", "Section", "Unknown"}:
                return current_id
            if kind == "Method" and parent_kind == "Interface":
                current_id = enclosing_symbol_id
                continue
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

    def search(self, project_id: str, query: str, limit: int = 10) -> list[FileGroup]:
        signature = self.storage.get_project_signature(project_id) or ""
        query_hash = sha256(f"{signature}:{self.provider.model_id}:{query}:{limit}".encode("utf-8")).hexdigest()
        cached = self.storage.get_query_cache(project_id, query_hash)
        if cached is not None:
            return self._deserialize_groups(cached)

        symbol_rows = self.storage.get_symbol_rows(project_id)
        rankable_rows = _rankable_query_rows(symbol_rows)
        if not rankable_rows:
            return []

        canonical_bm25_hits: list[tuple[QueryChunkHit, str]] = []
        for hit in self.storage.search_bm25(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            if canonical_symbol_id in rankable_rows:
                canonical_bm25_hits.append((hit, canonical_symbol_id))

        canonical_dense_hits: list[tuple[QueryChunkHit, str]] = []
        for hit in self._dense_hits(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            if canonical_symbol_id in rankable_rows:
                canonical_dense_hits.append((hit, canonical_symbol_id))

        lexical_scores = _normalize_scores(self._symbol_scores(canonical_bm25_hits))
        semantic_scores = _normalize_scores(self._symbol_scores(canonical_dense_hits))

        combined_seed_scores: dict[str, float] = {}
        for symbol_id in set(lexical_scores) | set(semantic_scores):
            combined_seed_scores[symbol_id] = (
                semantic_scores.get(symbol_id, 0.0) * 0.65
                + lexical_scores.get(symbol_id, 0.0) * 0.35
            )

        if not combined_seed_scores:
            return []

        max_seed = max(combined_seed_scores.values())
        threshold = max(0.35, max_seed * 0.6)
        seed_personalization = {
            symbol_id: score
            for symbol_id, score in combined_seed_scores.items()
            if score >= threshold
        }
        if not seed_personalization:
            top_seed_ids = sorted(combined_seed_scores, key=combined_seed_scores.get, reverse=True)[:5]
            seed_personalization = {symbol_id: combined_seed_scores[symbol_id] for symbol_id in top_seed_ids}

        ppr_scores = dependency_ordered_pagerank(
            symbol_rows,
            self.storage.get_edges(project_id),
            personalization=seed_personalization,
            alpha=0.15,
        )
        normalized_ppr_scores = _normalize_scores(ppr_scores)

        candidates: dict[str, float] = {}
        for symbol_id in set(combined_seed_scores) | set(sorted(normalized_ppr_scores, key=normalized_ppr_scores.get, reverse=True)[: limit * 4]):
            row = rankable_rows.get(symbol_id)
            if row is None:
                continue
            semantic = semantic_scores.get(symbol_id, 0.0)
            lexical = lexical_scores.get(symbol_id, 0.0)
            ppr = normalized_ppr_scores.get(symbol_id, 0.0)
            score = semantic * 0.50 + lexical * 0.15 + ppr * 0.35
            score = apply_kind_boost(row, score)
            if score > 0:
                candidates[symbol_id] = score

        ranked: list[RankedSymbol] = []
        for symbol_id, score in sorted(candidates.items(), key=lambda item: (-item[1], item[0])):
            row = rankable_rows[symbol_id]
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

        grouped: dict[str, FileGroup] = {}
        for symbol in ranked:
            group = grouped.setdefault(
                symbol.relative_path,
                FileGroup(relative_path=symbol.relative_path, score=0.0),
            )
            group.score = max(group.score, symbol.score)
            if symbol.symbol_id not in {item.symbol_id for item in group.symbols}:
                group.symbols.append(symbol)

        results = sorted(grouped.values(), key=lambda item: (-item.score, item.relative_path))[:limit]
        for group in results:
            group.symbols.sort(key=lambda item: (-item.score, item.display_name))
        self.storage.store_query_cache(project_id, query_hash, self._serialize_groups(results))
        return results
