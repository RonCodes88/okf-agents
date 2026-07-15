"""Unit tests for idempotent vector-store synchronization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.indexing import (
    CONTENT_HASH_KEY,
    stable_document_id,
    sync_bundle_to_vector_store,
)
from langgraph_okf.retriever import concept_to_document

pytestmark = pytest.mark.unit

SAMPLE_CONCEPT_IDS = [
    "concepts/customers",
    "concepts/orders",
    "concepts/payments",
    "guides/getting-started",
]


class FakeVectorStore(VectorStore):
    """In-memory store implementing only the capabilities sync requires."""

    def __init__(self) -> None:
        self.storage: dict[str, Document] = {}
        self.add_batches: list[list[str]] = []
        self.get_batches: list[list[str]] = []

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids: list[str] = kwargs["ids"]
        self.add_batches.append(list(ids))
        for doc_id, document in zip(ids, documents, strict=True):
            self.storage[doc_id] = Document(
                page_content=document.page_content,
                metadata=dict(document.metadata),
                id=doc_id,
            )
        return list(ids)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        self.get_batches.append(list(ids))
        return [self.storage[doc_id] for doc_id in ids if doc_id in self.storage]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> FakeVectorStore:
        raise NotImplementedError


class NoLookupStore(VectorStore):
    """Accepts stable-ID writes but cannot look documents up by ID."""

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        raise AssertionError("must not be called")

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        raise NotImplementedError

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> NoLookupStore:
        raise NotImplementedError


class NoStableIdStore(NoLookupStore):
    """Looks up by ID but its only write override cannot take IDs."""

    def add_documents(self, documents: list[Document]) -> list[str]:  # type: ignore[override]
        raise AssertionError("must not be called")

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return []


class FailingAddStore(FakeVectorStore):
    """Raises when asked to write a batch containing a poisoned ID."""

    def __init__(self, poisoned_ids: set[str]) -> None:
        super().__init__()
        self.poisoned_ids = poisoned_ids

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        if self.poisoned_ids.intersection(kwargs["ids"]):
            raise RuntimeError("backend write rejected\nwith a second line")
        return super().add_documents(documents, **kwargs)


class FailingLookupStore(FakeVectorStore):
    """Raises when asked to look up a batch containing a poisoned ID."""

    def __init__(self, poisoned_ids: set[str]) -> None:
        super().__init__()
        self.poisoned_ids = poisoned_ids

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if self.poisoned_ids.intersection(ids):
            raise ConnectionError("lookup unavailable")
        return super().get_by_ids(ids)


@pytest.fixture()
def store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture()
def mutable_bundle_path(tmp_path: Path) -> Path:
    (tmp_path / "alpha.md").write_text("---\ntype: note\n---\n\nAlpha body.\n", encoding="utf-8")
    (tmp_path / "beta.md").write_text("---\ntype: note\n---\n\nBeta body.\n", encoding="utf-8")
    return tmp_path


class TestClassification:
    def test_initial_sync_adds_every_concept(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        result = sync_bundle_to_vector_store(bundle, store)
        assert (result.added, result.updated, result.skipped, result.failed) == (4, 0, 0, 0)
        assert result.errors == []
        assert len(store.storage) == 4

    def test_resync_skips_unchanged_concepts(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store)
        result = sync_bundle_to_vector_store(bundle, store)
        assert (result.added, result.updated, result.skipped, result.failed) == (0, 0, 4, 0)

    def test_resync_is_idempotent_in_the_store(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store)
        first_snapshot = {doc_id: doc.model_copy() for doc_id, doc in store.storage.items()}
        first_add_batches = len(store.add_batches)
        sync_bundle_to_vector_store(bundle, store)
        assert len(store.add_batches) == first_add_batches
        assert store.storage.keys() == first_snapshot.keys()
        for doc_id, snapshot in first_snapshot.items():
            assert store.storage[doc_id].page_content == snapshot.page_content
            assert store.storage[doc_id].metadata == snapshot.metadata

    def test_changed_concept_is_updated(
        self, mutable_bundle_path: Path, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(OKFBundle.load(mutable_bundle_path), store)
        (mutable_bundle_path / "alpha.md").write_text(
            "---\ntype: note\n---\n\nAlpha body, revised.\n", encoding="utf-8"
        )
        result = sync_bundle_to_vector_store(OKFBundle.load(mutable_bundle_path), store)
        assert (result.added, result.updated, result.skipped, result.failed) == (0, 1, 1, 0)
        updated_id = stable_document_id(mutable_bundle_path, "alpha")
        assert "revised" in store.storage[updated_id].page_content

    def test_overwrite_rewrites_unchanged_concepts(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store)
        result = sync_bundle_to_vector_store(bundle, store, overwrite=True)
        assert (result.added, result.updated, result.skipped, result.failed) == (0, 4, 0, 0)

    def test_empty_bundle_returns_zero_counts(
        self, tmp_path: Path, store: FakeVectorStore
    ) -> None:
        result = sync_bundle_to_vector_store(OKFBundle.load(tmp_path), store)
        assert (result.added, result.updated, result.skipped, result.failed) == (0, 0, 0, 0)
        assert result.errors == []
        assert store.storage == {}
        assert store.get_batches == []


class TestStableIdsAndMetadata:
    def test_store_keys_are_stable_ids(self, bundle: OKFBundle, store: FakeVectorStore) -> None:
        sync_bundle_to_vector_store(bundle, store)
        expected = {
            stable_document_id(bundle.root, concept_id) for concept_id in SAMPLE_CONCEPT_IDS
        }
        assert store.storage.keys() == expected

    def test_stable_id_is_deterministic_and_namespaced(self, bundle: OKFBundle) -> None:
        doc_id = stable_document_id(bundle.root, "concepts/orders")
        assert doc_id == stable_document_id(bundle.root, "concepts/orders")
        assert doc_id != stable_document_id(bundle.root, "concepts/customers")
        assert doc_id != stable_document_id("/somewhere/else", "concepts/orders")

    def test_stored_metadata_is_canonical_plus_content_hash(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store)
        stored = store.storage[stable_document_id(bundle.root, "concepts/orders")]
        canonical = concept_to_document(bundle.get("concepts/orders"), bundle_root=bundle.root)
        content_hash = stored.metadata[CONTENT_HASH_KEY]
        assert isinstance(content_hash, str) and len(content_hash) == 64
        assert stored.page_content == canonical.page_content
        assert stored.metadata == {**canonical.metadata, CONTENT_HASH_KEY: content_hash}

    def test_content_hash_is_deterministic_across_stores(self, bundle: OKFBundle) -> None:
        first, second = FakeVectorStore(), FakeVectorStore()
        sync_bundle_to_vector_store(bundle, first)
        sync_bundle_to_vector_store(bundle, second)
        for doc_id, document in first.storage.items():
            assert (
                second.storage[doc_id].metadata[CONTENT_HASH_KEY]
                == document.metadata[CONTENT_HASH_KEY]
            )


class TestBatching:
    def test_batch_size_splits_writes_and_lookups(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store, batch_size=3)
        assert [len(batch) for batch in store.get_batches] == [3, 1]
        assert [len(batch) for batch in store.add_batches] == [3, 1]

    def test_batch_larger_than_bundle_uses_one_batch(
        self, bundle: OKFBundle, store: FakeVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, store, batch_size=100)
        assert [len(batch) for batch in store.add_batches] == [4]

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_invalid_batch_size_raises_value_error(
        self, bundle: OKFBundle, store: FakeVectorStore, batch_size: int
    ) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            sync_bundle_to_vector_store(bundle, store, batch_size=batch_size)


class TestUnsupportedStores:
    def test_store_without_get_by_ids_raises_type_error(self, bundle: OKFBundle) -> None:
        with pytest.raises(TypeError, match="get_by_ids"):
            sync_bundle_to_vector_store(bundle, NoLookupStore())

    def test_store_without_stable_id_writes_raises_type_error(self, bundle: OKFBundle) -> None:
        with pytest.raises(TypeError, match="ids"):
            sync_bundle_to_vector_store(bundle, NoStableIdStore())

    def test_capability_errors_name_the_store_class(self, bundle: OKFBundle) -> None:
        with pytest.raises(TypeError, match="NoLookupStore"):
            sync_bundle_to_vector_store(bundle, NoLookupStore())


class TestPartialFailures:
    def test_failed_write_batch_does_not_stop_later_batches(self, bundle: OKFBundle) -> None:
        poisoned = stable_document_id(bundle.root, "concepts/customers")
        store = FailingAddStore({poisoned})
        result = sync_bundle_to_vector_store(bundle, store, batch_size=1)
        assert (result.added, result.updated, result.skipped, result.failed) == (3, 0, 0, 1)
        assert poisoned not in store.storage
        assert stable_document_id(bundle.root, "guides/getting-started") in store.storage

    def test_failed_lookup_batch_does_not_stop_later_batches(self, bundle: OKFBundle) -> None:
        poisoned = stable_document_id(bundle.root, "concepts/customers")
        store = FailingLookupStore({poisoned})
        result = sync_bundle_to_vector_store(bundle, store, batch_size=1)
        assert (result.added, result.updated, result.skipped, result.failed) == (3, 0, 0, 1)
        assert len(result.errors) == 1

    def test_failed_batch_skips_only_its_own_documents(self, bundle: OKFBundle) -> None:
        poisoned = stable_document_id(bundle.root, "concepts/customers")
        store = FailingAddStore({poisoned})
        result = sync_bundle_to_vector_store(bundle, store, batch_size=2)
        # customers and orders share the first write batch; both fail together.
        assert (result.added, result.updated, result.skipped, result.failed) == (2, 0, 0, 2)
        assert len(store.storage) == 2

    def test_errors_are_sanitized_and_concept_specific(self, bundle: OKFBundle) -> None:
        poisoned = stable_document_id(bundle.root, "concepts/customers")
        result = sync_bundle_to_vector_store(bundle, FailingAddStore({poisoned}), batch_size=1)
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.startswith("concepts/customers: RuntimeError:")
        assert "\n" not in error
        assert "second line" not in error
