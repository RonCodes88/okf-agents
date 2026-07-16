"""Unit tests for the keyword and graph retrievers and document conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic import ValidationError

from okf_agents.bundle import OKFBundle
from okf_agents.retriever import OKFGraphRetriever, OKFRetriever, concept_to_document

pytestmark = pytest.mark.unit

MINIMAL_CONCEPT = "---\ntype: note\n---\n\nMinimal body about payments.\n"


@pytest.fixture(scope="module")
def retriever(bundle: OKFBundle) -> OKFRetriever:
    return OKFRetriever(bundle=bundle)


@pytest.fixture()
def minimal_bundle(tmp_path: Path) -> OKFBundle:
    (tmp_path / "minimal.md").write_text(MINIMAL_CONCEPT, encoding="utf-8")
    return OKFBundle.load(tmp_path)


class TestInvocation:
    def test_invoke_returns_documents(self, retriever: OKFRetriever) -> None:
        results = retriever.invoke("payments")
        assert results
        assert all(isinstance(document, Document) for document in results)

    def test_no_matches_returns_empty_list(self, retriever: OKFRetriever) -> None:
        assert retriever.invoke("zebra") == []

    def test_top_k_limits_results(self, bundle: OKFBundle) -> None:
        retriever = OKFRetriever(bundle=bundle, top_k=2)
        assert len(retriever.invoke("orders")) == 2

    def test_results_follow_bundle_search_ordering(self, retriever: OKFRetriever) -> None:
        results = retriever.invoke("orders")
        assert [document.metadata["concept_id"] for document in results] == [
            "concepts/orders",
            "concepts/customers",
            "guides/getting-started",
        ]

    def test_top_result_is_most_relevant(self, retriever: OKFRetriever) -> None:
        results = retriever.invoke("payments")
        assert results[0].metadata["concept_id"] == "concepts/payments"

    def test_invalid_top_k_raises_value_error(self, bundle: OKFBundle) -> None:
        with pytest.raises(ValueError):
            OKFRetriever(bundle=bundle, top_k=0)

    @pytest.mark.parametrize("bad_bundle", [None, "not a bundle", 42])
    def test_rejects_non_bundle_immediately(self, bad_bundle: object) -> None:
        with pytest.raises(ValidationError, match="OKFBundle"):
            OKFRetriever(bundle=bad_bundle)  # type: ignore[arg-type]


class TestDocumentContract:
    def test_page_content_is_markdown_body(self, retriever: OKFRetriever) -> None:
        document = retriever.invoke("payments")[0]
        assert document.page_content.lstrip().startswith("# Payments")
        assert "---" not in document.page_content

    def test_complete_metadata(self, bundle: OKFBundle, retriever: OKFRetriever) -> None:
        document = retriever.invoke("orders")[0]
        assert document.metadata == {
            "concept_id": "concepts/orders",
            "title": "Orders",
            "type": "table",
            "tags": ["sales", "commerce"],
            "path": str(bundle.root / "concepts" / "orders.md"),
            "source": "okf_bundle",
            "bundle_root": str(bundle.root),
            "description": "Order lifecycle and fulfilment records.",
            "timestamp": "2026-07-01T09:30:00",
        }

    def test_minimal_metadata_omits_optional_fields(self, minimal_bundle: OKFBundle) -> None:
        document = OKFRetriever(bundle=minimal_bundle).invoke("payments")[0]
        assert document.metadata == {
            "concept_id": "minimal",
            "title": "minimal",
            "type": "note",
            "tags": [],
            "path": str(minimal_bundle.root / "minimal.md"),
            "source": "okf_bundle",
            "bundle_root": str(minimal_bundle.root),
        }

    def test_paths_are_absolute(self, retriever: OKFRetriever) -> None:
        document = retriever.invoke("payments")[0]
        assert Path(document.metadata["path"]).is_absolute()
        assert Path(document.metadata["bundle_root"]).is_absolute()

    def test_metadata_is_json_serializable(self, retriever: OKFRetriever) -> None:
        for document in retriever.invoke("orders"):
            assert json.loads(json.dumps(document.metadata)) == document.metadata

    def test_resource_included_when_present(self, tmp_path: Path) -> None:
        (tmp_path / "linked.md").write_text(
            "---\ntype: note\nresource: https://example.com/doc\n---\n\nResource body.\n",
            encoding="utf-8",
        )
        bundle = OKFBundle.load(tmp_path)
        document = concept_to_document(bundle.get("linked"), bundle_root=bundle.root)
        assert document.metadata["resource"] == "https://example.com/doc"


class TestConversionHelper:
    def test_matches_retriever_output(self, bundle: OKFBundle, retriever: OKFRetriever) -> None:
        document = retriever.invoke("payments")[0]
        expected = concept_to_document(bundle.get("concepts/payments"), bundle_root=bundle.root)
        assert document.page_content == expected.page_content
        assert document.metadata == expected.metadata

    def test_tags_list_is_a_copy(self, bundle: OKFBundle) -> None:
        concept = bundle.get("concepts/orders")
        document = concept_to_document(concept, bundle_root=bundle.root)
        document.metadata["tags"].append("mutated")
        assert "mutated" not in concept.frontmatter.tags


class ScriptedVectorStore(VectorStore):
    """Fake store returning a fixed hit list from ``similarity_search``."""

    def __init__(self, hits: list[Document]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int]] = []

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        self.calls.append((query, k))
        return self.hits[:k]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> ScriptedVectorStore:
        raise NotImplementedError


def hit(bundle: OKFBundle, concept_id: str, **overrides: Any) -> Document:
    """Build a vector hit for a concept, optionally with tampered metadata."""
    metadata: dict[str, Any] = {"concept_id": concept_id, "bundle_root": str(bundle.root)}
    metadata.update(overrides)
    return Document(page_content="stale store copy", metadata=metadata)


def graph_retriever(
    bundle: OKFBundle, hits: list[Document], **fields: Any
) -> OKFGraphRetriever:
    return OKFGraphRetriever(bundle=bundle, vector_store=ScriptedVectorStore(hits), **fields)


def result_ids(documents: list[Document]) -> list[str]:
    return [document.metadata["concept_id"] for document in documents]


class TestGraphRetrieverEntryHits:
    def test_entry_hits_keep_vector_store_order(self, bundle: OKFBundle) -> None:
        hits = [hit(bundle, "concepts/payments"), hit(bundle, "guides/getting-started")]
        retriever = graph_retriever(bundle, hits, expand_hops=0)
        assert result_ids(retriever.invoke("q")) == [
            "concepts/payments",
            "guides/getting-started",
        ]

    def test_top_k_is_passed_to_the_vector_store(self, bundle: OKFBundle) -> None:
        store = ScriptedVectorStore([hit(bundle, "concepts/orders")])
        OKFGraphRetriever(bundle=bundle, vector_store=store, top_k=3).invoke("payments query")
        assert store.calls == [("payments query", 3)]

    def test_no_hits_returns_empty_list(self, bundle: OKFBundle) -> None:
        assert graph_retriever(bundle, []).invoke("q") == []

    def test_malformed_metadata_is_ignored(self, bundle: OKFBundle) -> None:
        hits = [
            Document(page_content="no concept id", metadata={"bundle_root": str(bundle.root)}),
            Document(
                page_content="non-string concept id",
                metadata={"concept_id": 42, "bundle_root": str(bundle.root)},
            ),
            hit(bundle, "concepts/does-not-exist"),
            hit(bundle, "concepts/payments"),
        ]
        retriever = graph_retriever(bundle, hits, expand_hops=0)
        assert result_ids(retriever.invoke("q")) == ["concepts/payments"]

    def test_foreign_bundle_hits_are_ignored(self, bundle: OKFBundle) -> None:
        hits = [
            hit(bundle, "concepts/payments", bundle_root="/some/other/bundle"),
            hit(bundle, "concepts/orders"),
        ]
        retriever = graph_retriever(bundle, hits, expand_hops=0)
        assert result_ids(retriever.invoke("q")) == ["concepts/orders"]

    def test_duplicate_entry_hits_are_deduplicated(self, bundle: OKFBundle) -> None:
        hits = [hit(bundle, "concepts/orders"), hit(bundle, "concepts/orders")]
        retriever = graph_retriever(bundle, hits, expand_hops=0)
        assert result_ids(retriever.invoke("q")) == ["concepts/orders"]


class TestGraphRetrieverExpansion:
    def test_defaults(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(bundle, [])
        assert (retriever.top_k, retriever.expand_hops, retriever.expand_direction) == (
            5,
            1,
            "out",
        )

    def test_zero_hops_returns_only_entry_hits(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(bundle, [hit(bundle, "concepts/orders")], expand_hops=0)
        assert result_ids(retriever.invoke("q")) == ["concepts/orders"]

    def test_one_hop_out_appends_link_targets(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(bundle, [hit(bundle, "concepts/customers")])
        assert result_ids(retriever.invoke("q")) == ["concepts/customers", "concepts/orders"]

    def test_multiple_hops_follow_distance_then_id_order(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(
            bundle, [hit(bundle, "guides/getting-started")], expand_hops=2
        )
        assert result_ids(retriever.invoke("q")) == [
            "guides/getting-started",
            "concepts/orders",
            "concepts/customers",
            "concepts/payments",
        ]

    def test_direction_in_follows_backlinks(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(
            bundle, [hit(bundle, "concepts/payments")], expand_direction="in"
        )
        assert result_ids(retriever.invoke("q")) == ["concepts/payments", "concepts/orders"]

    def test_direction_both_merges_links_and_backlinks(self, bundle: OKFBundle) -> None:
        retriever = graph_retriever(
            bundle, [hit(bundle, "concepts/payments")], expand_hops=2, expand_direction="both"
        )
        assert result_ids(retriever.invoke("q")) == [
            "concepts/payments",
            "concepts/orders",
            "concepts/customers",
            "guides/getting-started",
        ]

    def test_cycles_terminate(self, bundle: OKFBundle) -> None:
        # customers and orders link to each other; large hop counts stay finite.
        retriever = graph_retriever(bundle, [hit(bundle, "concepts/customers")], expand_hops=10)
        results = result_ids(retriever.invoke("q"))
        assert results == ["concepts/customers", "concepts/orders", "concepts/payments"]
        assert len(results) == len(set(results))

    def test_expansion_never_repeats_entry_hits(self, bundle: OKFBundle) -> None:
        hits = [hit(bundle, "concepts/orders"), hit(bundle, "concepts/customers")]
        retriever = graph_retriever(bundle, hits)
        assert result_ids(retriever.invoke("q")) == [
            "concepts/orders",
            "concepts/customers",
            "concepts/payments",
        ]

    @pytest.mark.parametrize(
        "fields",
        [{"top_k": 0}, {"expand_hops": -1}, {"expand_direction": "sideways"}],
    )
    def test_invalid_parameters_raise_value_error(
        self, bundle: OKFBundle, fields: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError):
            graph_retriever(bundle, [], **fields)

    @pytest.mark.parametrize("bad_bundle", [None, "not a bundle", 42])
    def test_rejects_non_bundle_immediately(self, bundle: OKFBundle, bad_bundle: object) -> None:
        with pytest.raises(ValidationError, match="OKFBundle"):
            OKFGraphRetriever(bundle=bad_bundle, vector_store=ScriptedVectorStore([]))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_store", [None, "not a store", 42])
    def test_rejects_non_vector_store_immediately(
        self, bundle: OKFBundle, bad_store: object
    ) -> None:
        with pytest.raises(ValidationError, match="VectorStore"):
            OKFGraphRetriever(bundle=bundle, vector_store=bad_store)  # type: ignore[arg-type]


class TestGraphRetrieverRehydration:
    def test_documents_are_rehydrated_from_the_bundle(self, bundle: OKFBundle) -> None:
        stale = hit(bundle, "concepts/orders", content_hash="deadbeef", title="Tampered")
        document = graph_retriever(bundle, [stale], expand_hops=0).invoke("q")[0]
        canonical = concept_to_document(bundle.get("concepts/orders"), bundle_root=bundle.root)
        assert document.page_content == canonical.page_content
        assert document.metadata == canonical.metadata

    def test_expanded_documents_carry_canonical_metadata(self, bundle: OKFBundle) -> None:
        documents = graph_retriever(bundle, [hit(bundle, "concepts/customers")]).invoke("q")
        expanded = documents[1]
        canonical = concept_to_document(bundle.get("concepts/orders"), bundle_root=bundle.root)
        assert expanded.metadata == canonical.metadata
        assert expanded.page_content == canonical.page_content
