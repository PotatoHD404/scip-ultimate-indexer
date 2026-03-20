"""Document graph construction and Personalized PageRank scoring.

This module builds a directed graph of documentation chunks with weighted edges
and supports Personalized PageRank for query-time relevance boosting. It extends
the existing pagerank.py functionality with:

- Document-specific edge types (cross_file, intra_anchor, hierarchy, openapi_ref)
- Personalized PageRank with custom seed nodes
- Graph neighbor expansion for query-time retrieval
- Integration with the existing weighted_pagerank function

The graph is used to boost retrieval results by propagating relevance through
the document structure, similar to how the main indexer uses PageRank for code.
"""
from __future__ import annotations

import logging
from typing import Any

from ..pagerank import weighted_pagerank

logger = logging.getLogger(__name__)


# Default edge weights for different link types
DEFAULT_LINK_WEIGHTS = {
    'cross_file': 1.0,
    'cross_anchor': 1.0,
    'intra_anchor': 0.5,
    'hierarchy': 0.3,
    'openapi_ref': 0.8,
    'openapi_tag': 0.8,
    'sequence': 0.15,  # Half of hierarchy weight
}


class DocumentGraph:
    """
    Directed graph of document chunks with weighted edges.
    Supports Personalized PageRank for query-time relevance boosting.
    
    This is designed to work alongside the existing pagerank.py module,
    reusing the weighted_pagerank function while adding document-specific
    features like Personalized PageRank with seed nodes.
    """

    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        # Graph structure: node_id -> {target_id: {'weight': float, 'edge_type': str}}
        self._adjacency: dict[str, dict[str, dict[str, Any]]] = {}
        self._nodes: set[str] = set()
        self._node_data: dict[str, dict[str, Any]] = {}
        self._pagerank_cache: dict[str, dict[str, float]] = {}

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        count = 0
        for targets in self._adjacency.values():
            count += len(targets)
        return count

    def add_node(self, chunk_id: str, **kwargs):
        """Add a chunk as a node in the graph."""
        self._nodes.add(chunk_id)
        if chunk_id not in self._adjacency:
            self._adjacency[chunk_id] = {}
        self._node_data[chunk_id] = kwargs
        # Clear cache when graph changes
        self._pagerank_cache.clear()

    def add_edge(self, source_id: str, target_id: str, edge_type: str, weight: float):
        """Add a weighted directed edge."""
        if source_id not in self._nodes:
            logger.warning(f"Source node not found: {source_id}")
            return
        if target_id not in self._nodes:
            logger.warning(f"Target node not found: {target_id}")
            return

        # If edge already exists, use max weight
        if target_id in self._adjacency.get(source_id, {}):
            existing = self._adjacency[source_id][target_id]
            existing['weight'] = max(existing['weight'], weight)
            if edge_type not in existing.get('edge_types', []):
                existing.setdefault('edge_types', []).append(edge_type)
        else:
            self._adjacency.setdefault(source_id, {})
            self._adjacency[source_id][target_id] = {
                'weight': weight,
                'edge_type': edge_type,
                'edge_types': [edge_type],
            }
        # Clear cache when graph changes
        self._pagerank_cache.clear()

    def add_edges(self, edges: list[dict[str, Any]]):
        """Add multiple edges at once."""
        for edge in edges:
            self.add_edge(
                edge['source_chunk_id'],
                edge['target_chunk_id'],
                edge['edge_type'],
                edge['weight'],
            )

    def personalized_pagerank(
        self,
        seed_chunk_ids: list[str],
        seed_weights: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Compute Personalized PageRank with seeds on the given chunks.

        This biases the random walk to restart at the seed nodes,
        propagating relevance through the graph structure.

        Args:
            seed_chunk_ids: List of chunk IDs to use as personalization seeds
            seed_weights: Optional weights for each seed (default: uniform)

        Returns:
            Dictionary mapping chunk IDs to PageRank scores
        """
        if not seed_chunk_ids or not self._nodes:
            return {}

        # Build cache key
        cache_key = str(sorted(zip(seed_chunk_ids, seed_weights or [])))
        if cache_key in self._pagerank_cache:
            return self._pagerank_cache[cache_key]

        # Build personalization vector
        personalization: dict[str, float] = {}
        valid_seeds = [cid for cid in seed_chunk_ids if cid in self._nodes]

        if not valid_seeds:
            return {}

        if seed_weights:
            for cid, w in zip(seed_chunk_ids, seed_weights):
                if cid in self._nodes:
                    personalization[cid] = max(w, 0.0)
        else:
            for cid in valid_seeds:
                personalization[cid] = 1.0 / len(valid_seeds)

        # Normalize
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v / total for k, v in personalization.items()}

        # Build edge list for weighted_pagerank
        edges: list[tuple[str, str, float]] = []
        for source, targets in self._adjacency.items():
            for target, data in targets.items():
                edges.append((source, target, data['weight']))

        try:
            scores = weighted_pagerank(
                nodes=sorted(self._nodes),
                edges=edges,
                alpha=self.damping,
                personalization=personalization,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        except Exception as e:
            logger.warning(f"PageRank did not converge: {e}, using uniform scores")
            n = len(self._nodes)
            scores = {node: 1.0 / n for node in self._nodes}

        self._pagerank_cache[cache_key] = scores
        return scores

    def get_neighbors(
        self,
        chunk_id: str,
        max_hops: int = 2,
    ) -> set[str]:
        """Get all chunk IDs reachable within max_hops."""
        if chunk_id not in self._nodes:
            return set()

        visited: set[str] = set()
        frontier = {chunk_id}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                # Successors
                for neighbor in self._adjacency.get(node, {}).keys():
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                # Predecessors
                for source, targets in self._adjacency.items():
                    if node in targets and source not in visited:
                        next_frontier.add(source)
            visited.update(frontier)
            frontier = next_frontier - visited

        visited.update(frontier)
        visited.discard(chunk_id)
        return visited

    def get_node_data(self, chunk_id: str) -> dict[str, Any]:
        """Get metadata for a node."""
        return self._node_data.get(chunk_id, {})

    def get_all_nodes(self) -> list[str]:
        """Get all chunk IDs in the graph."""
        return list(self._nodes)

    def get_stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        if not self._nodes:
            return {'nodes': 0, 'edges': 0}

        in_degrees: list[int] = []
        out_degrees: list[int] = []
        for node in self._nodes:
            out_deg = len(self._adjacency.get(node, {}))
            out_degrees.append(out_deg)
            in_deg = sum(
                1 for targets in self._adjacency.values()
                if node in targets
            )
            in_degrees.append(in_deg)

        # Count edges by type
        edge_types: dict[str, int] = {}
        for targets in self._adjacency.values():
            for data in targets.values():
                lt = data.get('edge_type', 'unknown')
                edge_types[lt] = edge_types.get(lt, 0) + 1

        return {
            'nodes': self.num_nodes,
            'edges': self.num_edges,
            'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            'max_in_degree': max(in_degrees) if in_degrees else 0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'edge_types': edge_types,
        }
