"""Dependency-free directed-graph helpers for the bundle link graph.

Adjacency maps use concept IDs and hold deduplicated, ID-sorted neighbor
lists so every traversal is deterministic. No graph package is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from langgraph_okf.models import LinkEdge

__all__ = [
    "breadth_first_reachable",
    "build_adjacency",
    "merge_adjacency",
]

Adjacency = dict[str, list[str]]


def build_adjacency(edges: Iterable[LinkEdge]) -> tuple[Adjacency, Adjacency]:
    """Build ``(outbound, inbound)`` adjacency maps from resolved edges.

    Unresolved edges are excluded because their targets are not loaded
    concepts. Neighbor lists are deduplicated and sorted by concept ID.
    """
    outbound: dict[str, set[str]] = {}
    inbound: dict[str, set[str]] = {}
    for edge in edges:
        if not edge.resolved:
            continue
        outbound.setdefault(edge.source_id, set()).add(edge.target_id)
        inbound.setdefault(edge.target_id, set()).add(edge.source_id)
    return (
        {node: sorted(targets) for node, targets in outbound.items()},
        {node: sorted(sources) for node, sources in inbound.items()},
    )


def merge_adjacency(
    first: Mapping[str, Sequence[str]],
    second: Mapping[str, Sequence[str]],
) -> Adjacency:
    """Merge two adjacency maps into deduplicated, ID-sorted neighbor lists."""
    merged: dict[str, set[str]] = {}
    for adjacency in (first, second):
        for node, neighbors in adjacency.items():
            merged.setdefault(node, set()).update(neighbors)
    return {node: sorted(neighbors) for node, neighbors in merged.items()}


def breadth_first_reachable(
    adjacency: Mapping[str, Sequence[str]],
    start: str,
    hops: int,
) -> list[str]:
    """Return node IDs reachable from ``start`` within ``hops`` edges.

    Nodes are marked visited when enqueued, so cycles terminate and each
    node appears at most once. ``start`` itself is excluded. Results are
    ordered by distance from ``start``, then by ID within each distance.

    Raises:
        ValueError: If ``hops`` is negative.
    """
    if hops < 0:
        raise ValueError(f"hops must be non-negative, got {hops}")
    visited = {start}
    reachable: list[str] = []
    frontier = [start]
    for _ in range(hops):
        next_frontier = sorted(
            {
                neighbor
                for node in frontier
                for neighbor in adjacency.get(node, ())
                if neighbor not in visited
            }
        )
        if not next_frontier:
            break
        visited.update(next_frontier)
        reachable.extend(next_frontier)
        frontier = next_frontier
    return reachable
