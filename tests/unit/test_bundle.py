"""Unit tests for eager bundle loading, lexical search, and traversal."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from okf_agents.bundle import OKFBundle
from okf_agents.exceptions import (
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

    def test_missing_path_reason_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(BundleNotFoundError) as excinfo:
            OKFBundle.load(tmp_path / "nowhere")
        assert excinfo.value.reason == "missing"
        assert str(excinfo.value) == f"OKF bundle not found: {tmp_path / 'nowhere'}"

    def test_non_directory_path_reason_is_not_a_directory(
        self, sample_bundle_path: Path
    ) -> None:
        file_path = sample_bundle_path / "index.md"
        with pytest.raises(BundleNotFoundError) as excinfo:
            OKFBundle.load(file_path)
        assert excinfo.value.reason == "not_a_directory"
        assert str(excinfo.value) == (
            f"OKF bundle path exists but is not a directory: {file_path}"
        )

    def test_invalid_on_error_raises_value_error(self, sample_bundle_path: Path) -> None:
        with pytest.raises(ValueError, match="on_error"):
            OKFBundle.load(sample_bundle_path, on_error="ignore")  # type: ignore[arg-type]

    def test_strict_mode_still_raises_on_bad_files(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "good.md": VALID_CONCEPT,
                "missing-type.md": "---\ntitle: No type\n---\n",
            },
        )
        with pytest.raises(BundleValidationError) as excinfo:
            OKFBundle.load(tmp_path, on_error="raise")
        assert list(excinfo.value.failed_files) == ["missing-type.md"]

    def test_default_on_error_matches_explicit_raise(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {"good.md": VALID_CONCEPT, "missing-type.md": "---\ntitle: No type\n---\n"},
        )
        with pytest.raises(BundleValidationError):
            OKFBundle.load(tmp_path)

    def test_skip_mode_loads_good_files_and_reports_bad_ones(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "good.md": VALID_CONCEPT,
                "missing-type.md": "---\ntitle: No type\n---\n",
                "nested/bad-yaml.md": "---\ntype: [unclosed\n---\n",
            },
        )
        loaded = OKFBundle.load(tmp_path, on_error="skip")
        assert loaded.concept_count == 1
        assert [concept.id for concept in loaded.all_concepts()] == ["good"]
        assert sorted(loaded.skipped_files) == ["missing-type.md", "nested/bad-yaml.md"]

    def test_skip_mode_skipped_files_empty_when_nothing_skipped(
        self, sample_bundle_path: Path
    ) -> None:
        loaded = OKFBundle.load(sample_bundle_path, on_error="skip")
        assert loaded.skipped_files == {}

    def test_raise_mode_skipped_files_always_empty(self, bundle: OKFBundle) -> None:
        assert bundle.skipped_files == {}

    def test_skip_mode_returns_copy_of_skipped_files(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, {"good.md": VALID_CONCEPT, "bad.md": "---\ntitle: x\n---\n"})
        loaded = OKFBundle.load(tmp_path, on_error="skip")
        skipped = loaded.skipped_files
        skipped["injected.md"] = "not real"
        assert "injected.md" not in loaded.skipped_files

    def test_skip_mode_invalid_index_is_skipped_and_synthesized(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {"a.md": VALID_CONCEPT, "index.md": "# Escapes\n\n[out](../outside.md)\n"},
        )
        loaded = OKFBundle.load(tmp_path, on_error="skip")
        assert "index.md" in loaded.skipped_files
        index = loaded.index()
        assert index.title == "Index"
        assert index.concept_ids == ["a"]

    def test_skip_mode_unresolved_links_to_skipped_concepts(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "a.md": "---\ntype: note\n---\n\nSee [b](b.md).\n",
                "b.md": "---\ntitle: No type\n---\n",
            },
        )
        loaded = OKFBundle.load(tmp_path, on_error="skip")
        edges = loaded.links_from("a")
        assert [(edge.target_id, edge.resolved) for edge in edges] == [("b", False)]

    def test_empty_bundle_warns(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="no concept files"):
            loaded = OKFBundle.load(tmp_path)
        assert loaded.concept_count == 0

    def test_empty_bundle_after_skipping_everything_warns(self, tmp_path: Path) -> None:
        write_bundle(tmp_path, {"bad.md": "---\ntitle: No type\n---\n"})
        with pytest.warns(UserWarning, match="no concept files"):
            loaded = OKFBundle.load(tmp_path, on_error="skip")
        assert loaded.concept_count == 0
        assert "bad.md" in loaded.skipped_files

    def test_non_empty_bundle_does_not_warn(self, sample_bundle_path: Path) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            OKFBundle.load(sample_bundle_path)

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


class TestWikilinks:
    """Obsidian-style [[wikilink]] resolution by filename/title/alias."""

    def test_resolves_by_filename(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Orders]].\n",
                "orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edges = bundle.links_from("customers")
        assert [(e.target_id, e.resolved, e.link_kind, e.ambiguous) for e in edges] == [
            ("orders", True, "wiki", False)
        ]

    def test_resolves_by_title_when_it_differs_from_filename(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Order Concept]].\n",
                "orders.md": "---\ntype: note\ntitle: Order Concept\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.target_id == "orders"
        assert edge.resolved is True

    def test_resolves_by_alias(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Clients]].\n",
                "orders.md": (
                    "---\ntype: note\ntitle: Orders\naliases: [Clients]\n---\nBody.\n"
                ),
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.target_id == "orders"
        assert edge.resolved is True

    def test_case_insensitive_resolution(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[ORDERS]].\n",
                "orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.target_id == "orders"
        assert edge.resolved is True

    def test_ambiguous_title_is_reported_not_guessed(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Orders]].\n",
                "concepts/orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
                "archive/orders.md": "---\ntype: note\ntitle: Old Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.resolved is False
        assert edge.ambiguous is True
        assert edge.link_kind == "wiki"

    def test_path_qualified_wikilink_disambiguates(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[concepts/orders]].\n",
                "concepts/orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
                "archive/orders.md": "---\ntype: note\ntitle: Old Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.target_id == "concepts/orders"
        assert edge.resolved is True
        assert edge.ambiguous is False

    def test_unresolved_wikilink_is_tolerated_like_broken_markdown_link(
        self, tmp_path: Path
    ) -> None:
        write_bundle(
            tmp_path,
            {"customers.md": "---\ntype: note\n---\nSee [[Nowhere]].\n"},
        )
        bundle = OKFBundle.load(tmp_path)
        edge = bundle.links_from("customers")[0]
        assert edge.resolved is False
        assert edge.ambiguous is False
        assert edge.target_id == "nowhere"

    def test_wiki_backlinks_populate_like_markdown_backlinks(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Orders]].\n",
                "orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        backlinks = bundle.backlinks("orders")
        assert [(e.source_id, e.link_kind) for e in backlinks] == [("customers", "wiki")]

    def test_resolved_wikilink_neighbor_is_reachable(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Orders]].\n",
                "orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        assert [c.id for c in bundle.neighbors("customers")] == ["orders"]

    def test_ambiguous_wikilink_does_not_create_a_traversable_edge(self, tmp_path: Path) -> None:
        write_bundle(
            tmp_path,
            {
                "customers.md": "---\ntype: note\n---\nSee [[Orders]].\n",
                "concepts/orders.md": "---\ntype: note\ntitle: Orders\n---\nBody.\n",
                "archive/orders.md": "---\ntype: note\ntitle: Old Orders\n---\nBody.\n",
            },
        )
        bundle = OKFBundle.load(tmp_path)
        assert bundle.neighbors("customers") == []
