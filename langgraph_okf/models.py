"""Validated domain models for OKF v0.1 bundles.

Pure data contracts shared across the package: parsed concept frontmatter
and files, link-graph edges, the bundle index, and vector-store sync
results. Models validate structure only; filesystem concerns live in the
parser and bundle loader.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "BundleIndex",
    "Concept",
    "ConceptFrontmatter",
    "LinkEdge",
    "SyncResult",
]


class ConceptFrontmatter(BaseModel):
    """Parsed YAML frontmatter from a concept file.

    Standard OKF v0.1 metadata keys are typed fields. Unknown keys are
    stored only in ``extra`` and never duplicate the standard keys.
    """

    type: str
    title: str | None = None
    description: str | None = None
    resource: str | None = None
    tags: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _type_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _tags_none_is_empty(cls, value: object) -> object:
        # A missing YAML value (`tags:`) means "no tags"; scalars such as a
        # bare string are still rejected by the list[str] annotation.
        return [] if value is None else value


class Concept(BaseModel):
    """A single OKF concept file, fully parsed.

    ``raw`` preserves the exact file contents; ``body`` excludes the
    frontmatter and its delimiter lines. ``outbound_links`` holds target
    concept IDs deduplicated in first-seen order.
    """

    id: str
    path: str
    frontmatter: ConceptFrontmatter
    body: str
    outbound_links: list[str] = Field(default_factory=list)
    raw: str


class LinkEdge(BaseModel):
    """One directed inline-link occurrence between two concepts.

    ``resolved`` records whether ``target_id`` names a loaded concept; the
    parser emits edges unresolved and the bundle loader marks them.
    """

    source_id: str
    target_id: str
    anchor_text: str
    resolved: bool = False


class BundleIndex(BaseModel):
    """Parsed or synthesized root ``index.md``.

    ``body`` is the original or synthesized Markdown. ``title`` is the
    first H1 when present; ``description`` is the first non-heading
    paragraph when present. ``concept_ids`` holds normalized internal
    concept targets in first-seen order.
    """

    title: str | None = None
    description: str | None = None
    body: str
    concept_ids: list[str] = Field(default_factory=list)


class SyncResult(BaseModel):
    """Outcome counts from syncing a bundle into a vector store."""

    added: int = Field(default=0, ge=0)
    updated: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)
