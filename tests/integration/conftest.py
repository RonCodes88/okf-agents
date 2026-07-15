"""Fixtures shared by the integration test tier.

``vector_store`` and ``embeddings`` back the offline integration tests
in :mod:`tests.integration.test_hybrid_retriever`: an in-process,
dependency-free vector store paired with deterministic fake embeddings
so those tests run by default, without secrets, network access, or a
numpy dependency. ``integration_chat_model`` gates the
provider-integration tests in :mod:`tests.integration.test_navigator`
behind ``RUN_INTEGRATION_TESTS=1`` and a real provider key, skipping
cleanly otherwise.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from tests.provider_support import build_chat_model

RUN_INTEGRATION_TESTS = os.environ.get("RUN_INTEGRATION_TESTS") == "1"

_EMBEDDING_DIMENSIONS = 64


class HashedBagOfWordsEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings for offline tests.

    Each text becomes an L2-normalized term-frequency vector over a
    fixed number of hashed buckets, so cosine similarity reflects word
    overlap without any model, network call, or randomness.
    """

    def __init__(self, dimensions: int = _EMBEDDING_DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"\w+", text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dimensions] += 1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class InProcessVectorStore(VectorStore):
    """A minimal, dependency-free in-process vector store for tests.

    Supports the idempotent-sync capabilities
    (:func:`~okf_agents.indexing.sync_bundle_to_vector_store` requires
    ``get_by_ids`` plus stable-ID writes) and real cosine-similarity
    search over embeddings from the configured :class:`Embeddings`, all
    in pure Python so no numpy or network dependency is required.
    """

    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding
        self._documents: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def storage(self) -> dict[str, Document]:
        return self._documents

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids: list[str] | None = kwargs.get("ids")
        if ids is None:
            raise ValueError("InProcessVectorStore requires explicit ids")
        vectors = self.embedding.embed_documents([document.page_content for document in documents])
        for doc_id, document, vector in zip(ids, documents, vectors, strict=True):
            self._documents[doc_id] = Document(
                page_content=document.page_content, metadata=dict(document.metadata), id=doc_id
            )
            self._vectors[doc_id] = vector
        return list(ids)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        return [self._documents[doc_id] for doc_id in ids if doc_id in self._documents]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        query_vector = self.embedding.embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, vector), doc_id)
            for doc_id, vector in self._vectors.items()
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._documents[doc_id] for _, doc_id in scored[:k]]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> InProcessVectorStore:
        raise NotImplementedError


@pytest.fixture()
def embeddings() -> HashedBagOfWordsEmbeddings:
    return HashedBagOfWordsEmbeddings()


@pytest.fixture()
def vector_store(embeddings: HashedBagOfWordsEmbeddings) -> InProcessVectorStore:
    """A fresh, isolated in-process vector store for each test."""
    return InProcessVectorStore(embeddings)


@pytest.fixture()
def integration_chat_model() -> BaseChatModel:
    """A real chat model, skipping the test when integration runs are disabled."""
    if not RUN_INTEGRATION_TESTS:
        pytest.skip("set RUN_INTEGRATION_TESTS=1 to run provider integration tests")
    return build_chat_model()
