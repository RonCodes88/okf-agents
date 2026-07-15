"""LangChain retrievers backed by :class:`~langgraph_okf.bundle.OKFBundle`.

:class:`OKFRetriever` wraps the bundle's deterministic lexical search
behind LangChain's synchronous ``BaseRetriever`` protocol. The shared
:func:`concept_to_document` helper is the single place a concept becomes
a LangChain ``Document`` so retrieval and vector-store indexing emit an
identical metadata schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.models import Concept

__all__ = ["OKFRetriever", "concept_to_document"]

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
