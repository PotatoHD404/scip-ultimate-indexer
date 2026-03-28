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
QUERY_CACHE_VERSION = 2
DOCUMENTATION_SOURCE_KIND = "documentation"


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
        if str(row["kind"]) in {"Document", "Section"}:
            return DOCUMENTATION_SOURCE_KIND
    if fallback_kind in {"Document", "Section"}:
        return DOCUMENTATION_SOURCE_KIND
    return "code"


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
            _add_weight(source, target, base_weight * 0.10)
            continue
        if edge_type in COMPOSITIONAL_EDGE_TYPES:
            source_kind = _symbol_kind(rankable_rows[source])
            target_kind = _symbol_kind(rankable_rows[target])
            if source_kind in TYPE_CONTAINER_KINDS and target_kind in TYPE_MEMBER_KINDS:
                factor = 0.85
            elif target_kind in TYPE_CONTAINER_KINDS and source_kind in TYPE_MEMBER_KINDS:
                factor = 0.85
            else:
                factor = 0.20
            _add_weight(source, target, base_weight * factor)
            _add_weight(target, source, base_weight * factor)
            continue
        _add_weight(target, source, base_weight * 0.5)
        _add_weight(source, target, base_weight * 0.10)

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
        self._function_metadata_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}
        self._function_body_cache: dict[str, tuple[str | None, np.ndarray, list]] = {}

    def _cached_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        signature = self.storage.get_project_signature(project_id)
        cached = self._vector_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_chunk_vectors(project_id)
        self._vector_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _cached_function_metadata_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        """Load cached function metadata vectors."""
        signature = self.storage.get_project_signature(project_id)
        cached = self._function_metadata_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_function_metadata_vectors(project_id)
        self._function_metadata_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _cached_function_body_vectors(self, project_id: str) -> tuple[np.ndarray, list]:
        """Load cached function body chunk vectors."""
        signature = self.storage.get_project_signature(project_id)
        cached = self._function_body_cache.get(project_id)
        if cached is not None and cached[0] == signature:
            return cached[1], cached[2]
        matrix, rows = self.storage.load_function_body_vectors(project_id)
        self._function_body_cache[project_id] = (signature, matrix, rows)
        return matrix, rows

    def _query_vector(self, query: str) -> np.ndarray | None:
        query_cache_key = f"query::{query}"
        query_vector = self.storage.get_or_create_embedding(self.provider.model_id, query_cache_key)
        if query_vector is not None:
            return query_vector
        try:
            query_vector = generate_query_embedding(self.provider, query)
        except Exception:
            # Keep retrieval usable when the preferred dense backend is missing at
            # query time. BM25 and cached hash/local results can still answer.
            return None
        self.storage.store_embedding(self.provider.model_id, query_cache_key, query_vector)
        return query_vector

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

    def search(self, project_id: str, query: str, limit: int = 10) -> list[FileGroup]:
        """
        Search with dual-representation function indexing.
        
        Retrieval strategy:
        1. Search the metadata index for high-precision matches.
        2. Search the body index for behavioral or implementation matches.
        3. Search traditional chunks for general code matches.
        4. Merge hits by symbol.
        5. Rank symbols higher when they match in both indexes.
        """
        signature = self.storage.get_project_signature(project_id) or ""
        query_hash = sha256(
            f"{QUERY_CACHE_VERSION}:{signature}:{self.provider.model_id}:{query}:{limit}".encode("utf-8")
        ).hexdigest()
        cached = self.storage.get_query_cache(project_id, query_hash)
        if cached is not None:
            return self._deserialize_groups(cached)

        symbol_rows = self.storage.get_symbol_rows(project_id)
        rankable_rows = _rankable_query_rows(symbol_rows)
        
        # Collect all symbol scores from different sources
        all_symbol_scores: dict[str, dict[str, float]] = defaultdict(dict)
        
        # 1. Function metadata search (BM25 + dense)
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
        
        # 2. Function body search (BM25 + dense)
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
            # Avoid duplicates
            if not any(h["chunk_id"] == hit["chunk_id"] for h in all_symbol_scores[symbol_id]["body_hits"]):
                all_symbol_scores[symbol_id]["body_hits"].append(hit)
        
        # 3. Traditional chunk search (BM25 + dense)
        canonical_bm25_hits: list[tuple[QueryChunkHit, str]] = []
        for hit in self.storage.search_bm25(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            if canonical_symbol_id in rankable_rows:
                canonical_bm25_hits.append((hit, canonical_symbol_id))
                all_symbol_scores[canonical_symbol_id]["chunk_bm25"] = hit.score

        canonical_dense_hits: list[tuple[QueryChunkHit, str]] = []
        for hit in self._dense_hits(project_id, query, max(limit * 6, 24)):
            canonical_symbol_id = self._canonical_symbol_id(symbol_rows, hit.symbol_id)
            if canonical_symbol_id in rankable_rows:
                canonical_dense_hits.append((hit, canonical_symbol_id))
                all_symbol_scores[canonical_symbol_id]["chunk_dense"] = hit.score

        # Compute combined scores for each symbol
        def _compute_symbol_score(symbol_id: str) -> float:
            scores = all_symbol_scores.get(symbol_id, {})
            
            # Metadata scores (high precision for API shape queries)
            metadata_bm25 = scores.get("metadata_bm25", 0.0)
            metadata_dense = scores.get("metadata_dense", 0.0)
            metadata_score = max(metadata_bm25, metadata_dense) * 0.40
            
            # Body scores (behavioral/implementation recall)
            body_bm25 = scores.get("body_bm25", 0.0)
            body_dense = scores.get("body_dense", 0.0)
            body_score = max(body_bm25, body_dense) * 0.30
            
            # Chunk scores (traditional)
            chunk_bm25 = scores.get("chunk_bm25", 0.0)
            chunk_dense = scores.get("chunk_dense", 0.0)
            chunk_score = max(chunk_bm25, chunk_dense) * 0.30
            
            # Bonus for matching in both metadata and body (dual-representation boost)
            dual_bonus = 0.0
            if metadata_score > 0 and body_score > 0:
                dual_bonus = 0.15 * min(metadata_score, body_score)
            
            return metadata_score + body_score + chunk_score + dual_bonus
        
        # Normalize and compute final scores
        raw_scores = {sid: _compute_symbol_score(sid) for sid in all_symbol_scores}
        if not raw_scores:
            return []
        
        max_score = max(raw_scores.values())
        if max_score > 0:
            normalized_scores = {sid: score / max_score for sid, score in raw_scores.items()}
        else:
            normalized_scores = raw_scores
        
        # Apply kind boost and pagerank
        candidates: dict[str, float] = {}
        for symbol_id, base_score in normalized_scores.items():
            row = rankable_rows.get(symbol_id)
            if row is None:
                # For function metadata symbols not in symbol_rows, create a synthetic entry
                meta = all_symbol_scores[symbol_id].get("metadata", {})
                if meta:
                    candidates[symbol_id] = base_score * 1.1  # Boost for function symbols
                continue
            
            score = base_score
            score = apply_kind_boost(row, score)
            if score > 0:
                candidates[symbol_id] = score
        
        # Apply pagerank personalization
        max_seed = max(normalized_scores.values()) if normalized_scores else 0
        threshold = max(0.35, max_seed * 0.6)
        seed_personalization = {
            symbol_id: score
            for symbol_id, score in normalized_scores.items()
            if score >= threshold
        }
        
        if seed_personalization:
            ppr_scores = dependency_ordered_pagerank(
                symbol_rows,
                self.storage.get_edges(project_id),
                personalization=seed_personalization,
                alpha=0.15,
            )
            normalized_ppr_scores = _normalize_scores(ppr_scores)
            
            # Combine with pagerank
            for symbol_id in candidates:
                ppr = normalized_ppr_scores.get(symbol_id, 0.0)
                candidates[symbol_id] = candidates[symbol_id] * 0.70 + ppr * 0.30

        # Build ranked results
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
            else:
                # Use function metadata for symbols not in symbol_rows
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

        # Group by file
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
        results = self._select_groups(ordered_groups, symbol_rows, limit=limit)
        for group in results:
            group.symbols.sort(key=lambda item: (-item.score, item.display_name))
        self.storage.store_query_cache(project_id, query_hash, self._serialize_groups(results))
        return results
