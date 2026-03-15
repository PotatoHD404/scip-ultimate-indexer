from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256

import numpy as np

from .embeddings import EmbeddingProvider, cosine_similarity, generate_query_embedding
from .models import FileGroup, QueryChunkHit, RankedSymbol
from .pagerank import weighted_pagerank
from .storage import Storage


def _rrf_scores(items: list[QueryChunkHit], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, item in enumerate(items, start=1):
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


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

    def search(self, project_id: str, query: str, limit: int = 10) -> list[FileGroup]:
        signature = self.storage.get_project_signature(project_id) or ""
        query_hash = sha256(f"{signature}:{self.provider.model_id}:{query}:{limit}".encode("utf-8")).hexdigest()
        cached = self.storage.get_query_cache(project_id, query_hash)
        if cached is not None:
            return self._deserialize_groups(cached)

        bm25_hits = self.storage.search_bm25(project_id, query, max(limit * 5, 20))
        dense_hits = self._dense_hits(project_id, query, max(limit * 5, 20))
        fused = _rrf_scores(bm25_hits) | {}
        for chunk_id, value in _rrf_scores(dense_hits).items():
            fused[chunk_id] = fused.get(chunk_id, 0.0) + value

        chunk_by_id = {item.chunk_id: item for item in [*bm25_hits, *dense_hits]}
        symbol_rows = self.storage.get_symbol_rows(project_id)
        seed_scores: dict[str, float] = defaultdict(float)
        for chunk_id, score in fused.items():
            chunk = chunk_by_id[chunk_id]
            seed_scores[chunk.symbol_id] += score

        personalization_total = sum(seed_scores.values())
        personalization = None
        if personalization_total > 0:
            personalization = {
                key: value / personalization_total
                for key, value in seed_scores.items()
                if key in symbol_rows
            }

        reverse_edges = [
            (
                str(edge["target_symbol_id"]),
                str(edge["source_symbol_id"]),
                float(edge["weight"]),
            )
            for edge in self.storage.get_edges(project_id)
        ]
        ppr_scores = (
            weighted_pagerank(
                nodes=list(symbol_rows.keys()),
                edges=reverse_edges,
                alpha=0.85,
                personalization=personalization,
            )
            if personalization
            else {}
        )

        ranked: list[RankedSymbol] = []
        for symbol_id, row in symbol_rows.items():
            if row["kind"] == "File":
                continue
            final_score = (
                0.45 * seed_scores.get(symbol_id, 0.0)
                + 0.45 * ppr_scores.get(symbol_id, 0.0)
                + 0.10 * float(row["global_rank"])
            )
            if final_score <= 0:
                continue
            ranked.append(
                RankedSymbol(
                    symbol_id=symbol_id,
                    relative_path=str(row["relative_path"]),
                    display_name=str(row["display_name"]),
                    kind=str(row["kind"]),
                    score=final_score,
                    signature=str(row["signature"]),
                    docstring=str(row["docstring"]),
                    snippet=str(row["snippet"]),
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.relative_path, item.display_name))
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
