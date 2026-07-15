"""Unit tests for eager bundle loading, lexical search, and traversal."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.exceptions import (
    BundleNotFoundError,
    BundleValidationError,
    ConceptNotFoundError,
)

pytestmark = pytest.mark.unit

ALL_IDS = [
    "concepts/customers",
    "concepts/orders",
    "concepts/payments",
    "guides/getting-started",
]

VALID_CONCEPT = "---\ntype: note\ntitle: Valid\n---\n\nBody.\n"


def write_bundle(root: Path, files: dict[str, str]) -> Path:
    for relative, content in files.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    return root


class TestLoad:
    def test_loads_all_concepts_excluding_reserved_files(self, bundle: OKFBundle) -> None:
        assert bundle.concept_count == len(ALL_IDS)
        assert [concept.id for concept in bundle.all_concepts()] == ALL_IDS

    def test_nested_reserved_files_are_not_concepts(self, bundle: OKFBundle) -> None:
        for reserved_id in ("index", "log", "concepts/index"):
            with pytest.raises(ConceptNotFoundError):
                bundle.get(reserved_id)

    def test_accepts_string_path(self, sample_bundle_path: Path) -> None:
        assert OKFBundle.load(str(sample_bundle_path)).concept_count == len(ALL_IDS)

    def test_root_is_resolved_bundle_path(
        self, bundle: OKFBundle, sample_bundle_path: Path
    ) -> None:
        assert bundle.root == sample_bundle_path.resolve()
        assert bundle.root.is_absolute()

    def test_missing_path_raises_bundle_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(BundleNotFoundError):
            OKFBundle.load(tmp_path / "nowhere")

    def test_non_directory_path_raises_bundle_not_found(self, sample_bundle_path: Path) -> None:
        with pytest.raises(BundleNotFoundError):
            OKFBundle.load(sample_bundle_path / "index.md")

    def test_aggregates_all_invalid_files_into_one_error(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "good.md": VALID_CONCEPT,
                "missing-type.md": "---\ntitle: No type\n---\n",
                "nested/bad-yaml.md": "---\ntype: [unclosed\n---\n",
            },
        )
        with pytest.raises(BundleValidationError) as excinfo:
            OKFBundle.load(tmp_path)
        assert sorted(excinfo.value.failed_files) == ["missing-type.md", "nested/bad-yaml.md"]

    def test_load_is_deterministic_across_runs(self, sample_bundle_path: Path) -> None:
        first = OKFBundle.load(sample_bundle_path)
        second = OKFBundle.load(sample_bundle_path)
        assert [concept.id for concept in first.all_concepts()] == [
            concept.id for concept in second.all_concepts()
        ]
        assert first.index() == second.index()
        assert first.links_from("concepts/orders") == second.links_from("concepts/orders")


class TestIndex:
    def test_parses_root_index(self, bundle: OKFBundle) -> None:
        index = bundle.index()
        assert index.title == "Sample Store Knowledge"
        assert index.description == "A small commerce knowledge bundle used by the unit tests."
        assert index.concept_ids == ["concepts/orders", "concepts/customers", "concepts/payments"]

    def test_synthesizes_index_without_writing_to_disk(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, {"b.md": VALID_CONCEPT, "a.md": VALID_CONCEPT})
        loaded = OKFBundle.load(tmp_path)
        index = loaded.index()
        assert index.title == "Index"
        assert index.concept_ids == ["a", "b"]
        assert not (tmp_path / "index.md").exists()

    def test_invalid_root_index_is_reported(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {"a.md": VALID_CONCEPT, "index.md": "# Escapes\n\n[out](../outside.md)\n"},
        )
        with pytest.raises(BundleValidationError) as excinfo:
            OKFBundle.load(tmp_path)
        assert list(excinfo.value.failed_files) == ["index.md"]


class TestGet:
    def test_returns_parsed_concept(self, bundle: OKFBundle) -> None:
        concept = bundle.get("concepts/orders")
        assert concept.id == "concepts/orders"
        assert Path(concept.path).is_absolute()
        assert concept.frontmatter.type == "table"
        assert concept.frontmatter.title == "Orders"
        assert concept.frontmatter.tags == ["sales", "commerce"]
        assert concept.frontmatter.timestamp == datetime(2026, 7, 1, 9, 30)
        assert concept.frontmatter.extra == {"owner": "data-team"}
        assert concept.outbound_links == [
            "concepts/customers",
            "concepts/payments",
            "concepts/missing",
        ]

    def test_unknown_id_raises_concept_not_found(self, bundle: OKFBundle) -> None:
        with pytest.raises(ConceptNotFoundError) as excinfo:
            bundle.get("concepts/nonexistent")
        assert excinfo.value.concept_id == "concepts/nonexistent"


class TestSearch:
    def test_matches_and_ranks_by_relevance(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("customer")]
        assert results == ["concepts/customers", "concepts/orders"]

    def test_case_insensitive_partial_match(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("CUSTOM")]
        assert results == ["concepts/customers", "concepts/orders"]

    def test_title_outweighs_body(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("payment")]
        assert results[0] == "concepts/payments"
        assert "concepts/orders" in results

    def test_tags_are_searched(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("billing")]
        assert results == ["concepts/payments"]

    def test_ties_break_by_concept_id(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("table")]
        assert results == ["concepts/customers", "concepts/payments", "guides/getting-started"]

    def test_top_k_caps_results(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.search("table", top_k=2)]
        assert results == ["concepts/customers", "concepts/payments"]

    @pytest.mark.parametrize("query", ["", "   ", "!!!"], ids=["empty", "blank", "punctuation"])
    def test_queries_without_tokens_return_nothing(self, bundle: OKFBundle, query: str) -> None:
        assert bundle.search(query) == []

    def test_no_match_returns_nothing(self, bundle: OKFBundle) -> None:
        assert bundle.search("zzzqx") == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_invalid_top_k_raises_value_error(self, bundle: OKFBundle, top_k: int) -> None:
        with pytest.raises(ValueError, match="top_k"):
            bundle.search("customer", top_k=top_k)


class TestLinks:
    def test_links_from_in_document_order(self, bundle: OKFBundle) -> None:
        edges = bundle.links_from("concepts/orders")
        assert [(edge.target_id, edge.resolved) for edge in edges] == [
            ("concepts/customers", True),
            ("concepts/payments", True),
            ("concepts/missing", False),
        ]
        assert [edge.anchor_text for edge in edges] == ["customer", "payment", "missing archive"]
        assert all(edge.source_id == "concepts/orders" for edge in edges)

    def test_leaf_has_no_outbound_links(self, bundle: OKFBundle) -> None:
        assert bundle.links_from("concepts/payments") == []

    def test_backlinks_ordered_by_source_id(self, bundle: OKFBundle) -> None:
        edges = bundle.backlinks("concepts/orders")
        assert [edge.source_id for edge in edges] == [
            "concepts/customers",
            "guides/getting-started",
        ]
        assert all(edge.target_id == "concepts/orders" and edge.resolved for edge in edges)

    def test_backlinks_exclude_unresolved_targets(self, bundle: OKFBundle) -> None:
        edges = bundle.backlinks("concepts/payments")
        assert [edge.source_id for edge in edges] == ["concepts/orders"]

    @pytest.mark.parametrize("method", ["links_from", "backlinks"])
    def test_unknown_id_raises_concept_not_found(self, bundle: OKFBundle, method: str) -> None:
        with pytest.raises(ConceptNotFoundError):
            getattr(bundle, method)("concepts/nonexistent")


class TestNeighbors:
    def test_outbound_one_hop(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.neighbors("concepts/orders")]
        assert results == ["concepts/customers", "concepts/payments"]

    def test_cycle_terminates_and_excludes_root(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.neighbors("concepts/orders", hops=5)]
        assert results == ["concepts/customers", "concepts/payments"]

    def test_inbound_direction(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.neighbors("concepts/orders", direction="in")]
        assert results == ["concepts/customers", "guides/getting-started"]

    def test_both_directions(self, bundle: OKFBundle) -> None:
        results = [concept.id for concept in bundle.neighbors("concepts/orders", direction="both")]
        assert results == ["concepts/customers", "concepts/payments", "guides/getting-started"]

    def test_multiple_hops_order_by_distance_then_id(self, bundle: OKFBundle) -> None:
        results = [
            concept.id
            for concept in bundle.neighbors("concepts/payments", hops=2, direction="in")
        ]
        assert results == ["concepts/orders", "concepts/customers", "guides/getting-started"]

    def test_zero_hops_returns_nothing(self, bundle: OKFBundle) -> None:
        assert bundle.neighbors("concepts/orders", hops=0) == []

    def test_leaf_has_no_outbound_neighbors(self, bundle: OKFBundle) -> None:
        assert bundle.neighbors("concepts/payments") == []

    def test_unknown_id_raises_concept_not_found(self, bundle: OKFBundle) -> None:
        with pytest.raises(ConceptNotFoundError):
            bundle.neighbors("concepts/nonexistent")

    def test_negative_hops_raise_value_error(self, bundle: OKFBundle) -> None:
        with pytest.raises(ValueError, match="hops"):
            bundle.neighbors("concepts/orders", hops=-1)

    def test_invalid_direction_raises_value_error(self, bundle: OKFBundle) -> None:
        with pytest.raises(ValueError, match="direction"):
            bundle.neighbors("concepts/orders", direction="sideways")  # type: ignore[arg-type]


class TestImmutability:
    def test_all_concepts_returns_new_list(self, bundle: OKFBundle) -> None:
        first = bundle.all_concepts()
        first.clear()
        assert [concept.id for concept in bundle.all_concepts()] == ALL_IDS

    def test_links_from_returns_edge_copies(self, bundle: OKFBundle) -> None:
        edges = bundle.links_from("concepts/orders")
        edges[2].resolved = True
        assert bundle.links_from("concepts/orders")[2].resolved is False

    def test_backlinks_returns_edge_copies(self, bundle: OKFBundle) -> None:
        edges = bundle.backlinks("concepts/orders")
        edges[0].anchor_text = "mutated"
        assert bundle.backlinks("concepts/orders")[0].anchor_text == "orders"

    def test_index_returns_deep_copy(self, bundle: OKFBundle) -> None:
        index = bundle.index()
        index.concept_ids.append("concepts/injected")
        assert "concepts/injected" not in bundle.index().concept_ids

    def test_search_returns_new_list(self, bundle: OKFBundle) -> None:
        results = bundle.search("customer")
        results.clear()
        assert bundle.search("customer") != []
