"""Unit tests for the keyword retriever and document conversion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.retriever import OKFRetriever, concept_to_document

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
