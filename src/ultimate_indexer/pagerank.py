from __future__ import annotations

from collections import defaultdict


def weighted_pagerank(
    nodes: list[str],
    edges: list[tuple[str, str, float]],
    alpha: float = 0.85,
    personalization: dict[str, float] | None = None,
    max_iter: int = 200,
    tol: float = 1.0e-6,
) -> dict[str, float]:
    if not nodes:
        return {}

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    outgoing: dict[str, float] = defaultdict(float)
    for source, target, weight in edges:
        adjacency[source].append((target, weight))
        outgoing[source] += weight

    if personalization:
        total = sum(value for node, value in personalization.items() if node in nodes)
        if total <= 0:
            teleport = {node: 1.0 / len(nodes) for node in nodes}
        else:
            teleport = {node: personalization.get(node, 0.0) / total for node in nodes}
    else:
        teleport = {node: 1.0 / len(nodes) for node in nodes}

    rank = teleport.copy()
    for _ in range(max_iter):
        updated = {node: (1.0 - alpha) * teleport[node] for node in nodes}
        dangling_mass = alpha * sum(rank[node] for node in nodes if outgoing.get(node, 0.0) == 0.0)
        for node in nodes:
            updated[node] += dangling_mass * teleport[node]
        for source, targets in adjacency.items():
            total_weight = outgoing[source]
            if total_weight == 0:
                continue
            source_rank = alpha * rank.get(source, 0.0)
            for target, weight in targets:
                updated[target] += source_rank * (weight / total_weight)
        error = sum(abs(updated[node] - rank[node]) for node in nodes)
        rank = updated
        if error < len(nodes) * tol:
            break
    total = sum(rank.values())
    if total > 0:
        rank = {node: value / total for node, value in rank.items()}
    return rank
