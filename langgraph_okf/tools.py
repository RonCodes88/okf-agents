"""Deterministic LangChain tools over a loaded OKF bundle.

:func:`create_okf_tools` exposes an :class:`~langgraph_okf.bundle.OKFBundle`
as four read-only tools — ``read_concept``, ``search_concepts``,
``list_links``, and ``read_index`` — whose outputs are stable plain text
fit for direct insertion into an LLM context. No model is required to
create or invoke them. Expected lookup and input-validation failures are
returned as concise strings beginning with ``Error:``; unexpected
exceptions propagate so defects stay observable.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, ValidationError
from pydantic.v1 import ValidationError as ValidationErrorV1

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.exceptions import ConceptNotFoundError

__all__ = ["create_okf_tools"]

_SNIPPET_MAX_CHARS = 200
_TOP_K_MAX = 25
_NONE = "none"

_Direction = Literal["out", "in", "both"]


class ReadConceptInput(BaseModel):
    """Arguments for the ``read_concept`` tool."""

    concept_id: str = Field(
        description="Concept ID relative to the bundle root, for example 'concepts/orders'."
    )


class SearchConceptsInput(BaseModel):
    """Arguments for the ``search_concepts`` tool."""

    query: str = Field(
        description=(
            "Free-text query matched against concept titles, tags, descriptions, and bodies."
        )
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=_TOP_K_MAX,
        description=f"Maximum number of matches to return, between 1 and {_TOP_K_MAX}.",
    )


class ListLinksInput(BaseModel):
    """Arguments for the ``list_links`` tool."""

    concept_id: str = Field(
        description="Concept ID relative to the bundle root, for example 'concepts/orders'."
    )
    direction: _Direction = Field(
        default="both",
        description="Which links to list: 'out' (outgoing), 'in' (incoming), or 'both'.",
    )


class ReadIndexInput(BaseModel):
    """Arguments for the ``read_index`` tool. It takes no arguments."""


def _format_validation_error(error: ValidationError | ValidationErrorV1) -> str:
    """Render the first schema-validation failure as a concise error string."""
    first = error.errors()[0]
    location = ".".join(str(part) for part in first["loc"]) or "input"
    return f"Error: invalid tool input at {location}: {first['msg']}"


def _label(bundle: OKFBundle, concept_id: str) -> str:
    """Return the concept's title, falling back to its ID when absent."""
    try:
        concept = bundle.get(concept_id)
    except ConceptNotFoundError:
        return concept_id
    return concept.frontmatter.title or concept_id


def _snippet(text: str) -> str:
    """Collapse ``text`` to one heading-free line of at most 200 characters."""
    prose = " ".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    collapsed = " ".join(prose.split())
    if len(collapsed) <= _SNIPPET_MAX_CHARS:
        return collapsed
    return collapsed[:_SNIPPET_MAX_CHARS].rstrip() + "..."


def _read_concept(bundle: OKFBundle, concept_id: str) -> str:
    try:
        concept = bundle.get(concept_id)
    except ConceptNotFoundError as exc:
        return f"Error: {exc}"
    frontmatter = concept.frontmatter
    lines = [f"# {frontmatter.title or concept.id}", ""]
    lines.append(f"ID: {concept.id}")
    lines.append(f"Type: {frontmatter.type}")
    if frontmatter.description is not None:
        lines.append(f"Description: {frontmatter.description}")
    if frontmatter.resource is not None:
        lines.append(f"Resource: {frontmatter.resource}")
    if frontmatter.tags:
        lines.append(f"Tags: {', '.join(frontmatter.tags)}")
    if frontmatter.timestamp is not None:
        lines.append(f"Timestamp: {frontmatter.timestamp.isoformat()}")
    edges = bundle.links_from(concept.id)
    resolved = sorted({edge.target_id for edge in edges if edge.resolved})
    unresolved = sorted({edge.target_id for edge in edges if not edge.resolved})
    lines.append(f"Related (resolved): {', '.join(resolved) or _NONE}")
    lines.append(f"Related (unresolved): {', '.join(unresolved) or _NONE}")
    lines.extend(["", "---", "", concept.body.strip()])
    return "\n".join(lines)


def _search_concepts(bundle: OKFBundle, query: str, top_k: int) -> str:
    try:
        matches = bundle.search(query, top_k=top_k)
    except ValueError as exc:
        return f"Error: {exc}"
    if not matches:
        return f"No concepts matched the query {query!r}."
    lines: list[str] = []
    for rank, concept in enumerate(matches, start=1):
        frontmatter = concept.frontmatter
        label = frontmatter.title or concept.id
        lines.append(f"{rank}. {concept.id} - {label} (type: {frontmatter.type})")
        snippet = _snippet(frontmatter.description or concept.body)
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


def _list_links(bundle: OKFBundle, concept_id: str, direction: _Direction) -> str:
    try:
        outbound = bundle.links_from(concept_id) if direction in ("out", "both") else []
        inbound = bundle.backlinks(concept_id) if direction in ("in", "both") else []
    except ConceptNotFoundError as exc:
        return f"Error: {exc}"
    lines: list[str] = []
    seen_targets: set[str] = set()
    for edge in outbound:
        if edge.target_id in seen_targets:
            continue
        seen_targets.add(edge.target_id)
        suffix = "" if edge.resolved else " (unresolved)"
        lines.append(f"-> {edge.target_id} - {_label(bundle, edge.target_id)}{suffix}")
    seen_sources: set[str] = set()
    for edge in inbound:
        if edge.source_id in seen_sources:
            continue
        seen_sources.add(edge.source_id)
        lines.append(f"<- {edge.source_id} - {_label(bundle, edge.source_id)}")
    if not lines:
        return f"No links found for {concept_id} (direction: {direction})."
    return "\n".join([f"Links for {concept_id} (direction: {direction}):", *lines])


def create_okf_tools(bundle: OKFBundle) -> list[BaseTool]:
    """Create the four OKF agent tools for a loaded bundle.

    Returns ``read_concept``, ``search_concepts``, ``list_links``, and
    ``read_index``, in that order. All tools are synchronous, offline,
    and deterministic; outputs are plain text and never expose absolute
    filesystem paths.
    """

    def read_concept(concept_id: str) -> str:
        return _read_concept(bundle, concept_id)

    def search_concepts(query: str, top_k: int = 5) -> str:
        return _search_concepts(bundle, query, top_k)

    def list_links(concept_id: str, direction: _Direction = "both") -> str:
        return _list_links(bundle, concept_id, direction)

    def read_index() -> str:
        return bundle.index().body

    return [
        StructuredTool.from_function(
            func=read_concept,
            name="read_concept",
            description=(
                "Read one concept from the knowledge bundle by concept ID. Returns its "
                "metadata, resolved and unresolved related concept IDs, and full Markdown body."
            ),
            args_schema=ReadConceptInput,
            handle_validation_error=_format_validation_error,
        ),
        StructuredTool.from_function(
            func=search_concepts,
            name="search_concepts",
            description=(
                "Search all concepts in the knowledge bundle with a free-text query. Returns "
                "a numbered list of the best matches with concept IDs, types, and snippets."
            ),
            args_schema=SearchConceptsInput,
            handle_validation_error=_format_validation_error,
        ),
        StructuredTool.from_function(
            func=list_links,
            name="list_links",
            description=(
                "List the links of a concept: outgoing ('out'), incoming ('in'), or 'both'. "
                "Broken links are marked as unresolved."
            ),
            args_schema=ListLinksInput,
            handle_validation_error=_format_validation_error,
        ),
        StructuredTool.from_function(
            func=read_index,
            name="read_index",
            description=(
                "Read the bundle's root index: a Markdown overview linking to the bundle's "
                "concepts. Takes no arguments."
            ),
            args_schema=ReadIndexInput,
            handle_validation_error=_format_validation_error,
        ),
    ]
