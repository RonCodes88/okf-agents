"""Unit tests for the internal directed-graph helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest

from okf_agents._internal.graph_utils import (
    breadth_first_reachable,
    build_adjacency,
    merge_adjacency,
)
from okf_agents.models import LinkEdge

pytestmark = pytest.mark.unit


def edge(source: str, target: str, *, resolved: bool = True) -> LinkEdge:
    return LinkEdge(source_id=source, target_id=target, anchor_text=target, resolved=resolved)


class TestBuildAdjacency:
    def test_splits_outbound_and_inbound(self) -> None:
        outbound, inbound = build_adjacency([edge("a", "b"), edge("b", "c")])
        assert outbound == {"a": ["b"], "b": ["c"]}
        assert inbound == {"b": ["a"], "c": ["b"]}

    def test_excludes_unresolved_edges(self) -> None:
        outbound, inbound = build_adjacency([edge("a", "b"), edge("a", "ghost", resolved=False)])
        assert outbound == {"a": ["b"]}
        assert inbound == {"b": ["a"]}

    def test_deduplicates_and_sorts_neighbors(self) -> None:
        outbound, _ = build_adjacency([edge("a", "c"), edge("a", "b"), edge("a", "c")])
        assert outbound == {"a": ["b", "c"]}

    def test_empty_edges(self) -> None:
        assert build_adjacency([]) == ({}, {})


class TestMergeAdjacency:
    def test_unions_deduplicates_and_sorts(self) -> None:
        merged = merge_adjacency({"a": ["b"], "b": ["a"]}, {"a": ["c", "b"], "d": ["a"]})
        assert merged == {"a": ["b", "c"], "b": ["a"], "d": ["a"]}

    def test_empty_inputs(self) -> None:
        assert merge_adjacency({}, {}) == {}


class TestBreadthFirstReachable:
    ADJACENCY: ClassVar[dict[str, list[str]]] = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["a"],  # cycle back to the root
        "d": [],
    }

    def test_zero_hops_returns_nothing(self) -> None:
        assert breadth_first_reachable(self.ADJACENCY, "a", 0) == []

    def test_one_hop_returns_direct_neighbors_sorted(self) -> None:
        assert breadth_first_reachable(self.ADJACENCY, "a", 1) == ["b", "c"]

    def test_multiple_hops_order_by_distance_then_id(self) -> None:
        assert breadth_first_reachable(self.ADJACENCY, "a", 2) == ["b", "c", "d"]

    def test_cycle_excludes_root_and_terminates(self) -> None:
        assert breadth_first_reachable(self.ADJACENCY, "a", 10) == ["b", "c", "d"]

    def test_node_visited_at_shortest_distance_only(self) -> None:
        adjacency = {"a": ["b", "c"], "b": ["c", "d"], "c": [], "d": []}
        assert breadth_first_reachable(adjacency, "a", 3) == ["b", "c", "d"]

    def test_unknown_start_has_no_neighbors(self) -> None:
        assert breadth_first_reachable(self.ADJACENCY, "ghost", 3) == []

    def test_negative_hops_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="hops"):
            breadth_first_reachable(self.ADJACENCY, "a", -1)
