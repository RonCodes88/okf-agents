"""LangChain retrievers backed by :class:`~okf_agents.bundle.OKFBundle`.

:class:`OKFRetriever` wraps the bundle's deterministic lexical search
behind LangChain's synchronous ``BaseRetriever`` protocol, and
:class:`OKFGraphRetriever` expands vector-store hits through the bundle
link graph. The shared :func:`concept_to_document` helper is the single
place a concept becomes a LangChain ``Document`` so retrieval and
vector-store indexing emit an identical metadata schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, Field

from okf_agents.bundle import OKFBundle
from okf_agents.exceptions import ConceptNotFoundError
from okf_agents.models import Concept

__all__ = ["OKFGraphRetriever", "OKFRetriever", "concept_to_document"]

_DOCUMENT_SOURCE = "okf_bundle"


def concept_to_document(concept: Concept, *, bundle_root: Path) -> Document:
    """Convert one concept into a LangChain ``Document``.

    ``page_content`` is the Markdown body. Metadata always carries
    ``concept_id``, ``title`` (falling back to the concept ID), ``type``,
    ``tags``, the absolute file ``path``, ``source="okf_bundle"``, and the
    absolute ``bundle_root``; ``description``, ``resource``, and an ISO
    8601 ``timestamp`` string appear only when present. All values are
    JSON-serializable.
    """
    frontmatter = concept.frontmatter
    metadata: dict[str, Any] = {
        "concept_id": concept.id,
        "title": frontmatter.title or concept.id,
        "type": frontmatter.type,
        "tags": list(frontmatter.tags),
        "path": concept.path,
        "source": _DOCUMENT_SOURCE,
        "bundle_root": str(bundle_root),
    }
    if frontmatter.description is not None:
        metadata["description"] = frontmatter.description
    if frontmatter.resource is not None:
        metadata["resource"] = frontmatter.resource
    if frontmatter.timestamp is not None:
        metadata["timestamp"] = frontmatter.timestamp.isoformat()
    return Document(page_content=concept.body, metadata=metadata)


class OKFRetriever(BaseRetriever):
    """Keyword retriever over a loaded OKF bundle.

    Delegates to :meth:`OKFBundle.search`, so results follow the bundle's
    weighted lexical ranking: descending score, then concept ID. Use the
    inherited public ``invoke`` API to retrieve documents.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle: OKFBundle
    top_k: int = Field(default=5, ge=1)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        concepts = self.bundle.search(query, top_k=self.top_k)
        return [
            concept_to_document(concept, bundle_root=self.bundle.root) for concept in concepts
        ]


class OKFGraphRetriever(BaseRetriever):
    """Vector retriever that expands hits through the OKF link graph.

    Semantic entry hits come from ``vector_store`` and keep vector-store
    order; each is then expanded breadth-first over the bundle's resolved
    links. Results are deduplicated by concept ID with entry hits first,
    followed by expanded concepts in distance, then concept-ID order.
    Every returned ``Document`` is rehydrated from the bundle so its
    metadata is canonical regardless of what the store persisted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    bundle: OKFBundle
    vector_store: VectorStore
    top_k: int = Field(default=5, ge=1)
    expand_hops: int = Field(default=1, ge=0)
    expand_direction: Literal["out", "in", "both"] = "out"

    def _entry_concept_ids(self, hits: list[Document]) -> list[str]:
        """Validate hits and return in-bundle concept IDs in hit order."""
        bundle_root = str(self.bundle.root)
        entry_ids: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            concept_id = hit.metadata.get("concept_id")
            if not isinstance(concept_id, str) or concept_id in seen:
                continue
            hit_root = hit.metadata.get("bundle_root")
            if hit_root is not None and hit_root != bundle_root:
                continue
            try:
                self.bundle.get(concept_id)
            except ConceptNotFoundError:
                continue
            seen.add(concept_id)
            entry_ids.append(concept_id)
        return entry_ids

    def _expand(self, entry_ids: list[str]) -> list[str]:
        """Multi-source breadth-first expansion from the entry hits.

        Distances are measured from the nearest entry hit, so ordering is
        global distance, then concept ID, and entry hits never reappear.
        """
        seen = set(entry_ids)
        expanded: list[str] = []
        frontier = entry_ids
        for _ in range(self.expand_hops):
            next_frontier = sorted(
                {
                    neighbor.id
                    for concept_id in frontier
                    for neighbor in self.bundle.neighbors(
                        concept_id, hops=1, direction=self.expand_direction
                    )
                    if neighbor.id not in seen
                }
            )
            if not next_frontier:
                break
            seen.update(next_frontier)
            expanded.extend(next_frontier)
            frontier = next_frontier
        return expanded

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.vector_store.similarity_search(query, k=self.top_k)
        entry_ids = self._entry_concept_ids(hits)
        concept_ids = [*entry_ids, *self._expand(entry_ids)]
        return [
            concept_to_document(self.bundle.get(concept_id), bundle_root=self.bundle.root)
            for concept_id in concept_ids
        ]
