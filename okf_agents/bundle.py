"""Eager, immutable loading and querying of OKF v0.1 bundles.

:class:`OKFBundle` discovers every concept Markdown file under a bundle
root, parses all of them up front, and builds directed-link indexes for
lookup, lexical search, and breadth-first traversal. Bundles are
immutable after loading; every collection-returning method returns a new
list so callers cannot mutate bundle internals.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

from okf_agents._internal.graph_utils import (
    breadth_first_reachable,
    build_adjacency,
    merge_adjacency,
)
from okf_agents._internal.parser import (
    extract_internal_links,
    parse_bundle_index,
    parse_concept,
    synthesize_bundle_index,
)
from okf_agents.exceptions import (
    BundleNotFoundError,
    BundleValidationError,
    ConceptNotFoundError,
)
from okf_agents.models import BundleIndex, Concept, LinkEdge

__all__ = ["OKFBundle"]

_RESERVED_FILENAMES = frozenset({"index.md", "log.md"})
_ROOT_INDEX = "index.md"
_DIRECTIONS = ("out", "in", "both")
_ON_ERROR_MODES = ("raise", "skip")

# Search field weights per the shared contracts: title > tags > description > body.
_TITLE_WEIGHT = 4
_TAGS_WEIGHT = 3
_DESCRIPTION_WEIGHT = 2
_BODY_WEIGHT = 1


class OKFBundle:
    """A fully loaded OKF bundle: concepts, link graph, and root index.

    Construct with :meth:`load`; the initializer is an internal detail.
    """

    def __init__(
        self,
        *,
        root: Path,
        concepts: dict[str, Concept],
        edges_from: dict[str, list[LinkEdge]],
        edges_to: dict[str, list[LinkEdge]],
        index: BundleIndex,
        skipped_files: dict[str, str] | None = None,
    ) -> None:
        self._root = root
        self._concepts = concepts
        self._edges_from = edges_from
        self._edges_to = edges_to
        self._index = index
        self._skipped_files = dict(skipped_files) if skipped_files else {}
        all_edges = [edge for edges in edges_from.values() for edge in edges]
        self._out_adjacency, self._in_adjacency = build_adjacency(all_edges)
        self._both_adjacency = merge_adjacency(self._out_adjacency, self._in_adjacency)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> OKFBundle:
        """Eagerly load the bundle rooted at ``path``.

        Every concept file (any ``.md`` except reserved ``index.md`` and
        ``log.md`` at any depth) is parsed up front. A root ``index.md``
        is parsed when present and synthesized in memory otherwise.

        By default (``on_error="raise"``), any invalid file aggregates
        into one :class:`BundleValidationError` and nothing loads. With
        ``on_error="skip"``, invalid files (including an invalid root
        ``index.md``) are excluded instead: the bundle loads from
        whatever files are valid, and the excluded paths and reasons are
        available afterwards via :attr:`skipped_files`. A link to a
        skipped concept becomes an unresolved edge, the same as a link to
        a concept that never existed.

        A bundle that ends up with zero concepts (no matching files, or
        every file skipped) emits a :class:`UserWarning` rather than
        failing silently, since this usually indicates a mistyped path.

        Args:
            path: The bundle root directory.
            on_error: ``"raise"`` (default) or ``"skip"``.

        Raises:
            BundleNotFoundError: If ``path`` does not exist, is not a
                directory, or cannot be read.
            BundleValidationError: If ``on_error="raise"`` and any concept
                file is invalid, keyed by stable root-relative paths.
            ValueError: If ``on_error`` is not ``"raise"`` or ``"skip"``.
        """
        if on_error not in _ON_ERROR_MODES:
            raise ValueError(f"on_error must be one of {_ON_ERROR_MODES}, got {on_error!r}")
        root = Path(path).resolve()
        if not root.exists():
            raise BundleNotFoundError(str(path))
        if not root.is_dir():
            raise BundleNotFoundError(str(path), reason="not_a_directory")
        try:
            concept_files = sorted(
                (
                    file
                    for file in root.rglob("*.md")
                    if file.is_file() and file.name not in _RESERVED_FILENAMES
                ),
                key=lambda file: file.relative_to(root).as_posix(),
            )
        except OSError as exc:
            raise BundleNotFoundError(str(path)) from exc

        concepts: dict[str, Concept] = {}
        failures: dict[str, str] = {}
        for file in concept_files:
            relative = file.relative_to(root).as_posix()
            try:
                raw = file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures[relative] = "file is not valid UTF-8"
                continue
            except OSError as exc:
                failures[relative] = f"file could not be read: {exc}"
                continue
            try:
                concept = parse_concept(raw, bundle_root=root, file_path=file)
            except BundleValidationError as exc:
                failures.update(exc.failed_files)
                continue
            concepts[concept.id] = concept

        index = cls._load_index(root, concepts, failures)
        if failures and on_error == "raise":
            raise BundleValidationError(failures)
        if index is None:
            # Either no index.md was present, or it was invalid and
            # on_error="skip" tolerated that failure: synthesize from
            # whatever concepts loaded successfully.
            index = synthesize_bundle_index(list(concepts.values()))

        if not concepts:
            warnings.warn(
                f"OKF bundle at {root} contains no concept files", UserWarning, stacklevel=2
            )

        edges_from: dict[str, list[LinkEdge]] = {}
        edges_to: dict[str, list[LinkEdge]] = {}
        for concept_id in sorted(concepts):
            concept = concepts[concept_id]
            source = f"{concept_id}.md"
            edges = [
                edge.model_copy(update={"resolved": edge.target_id in concepts})
                for edge in extract_internal_links(
                    concept.body, source_id=concept_id, source=source
                )
            ]
            edges_from[concept_id] = edges
            for edge in edges:
                if edge.resolved:
                    edges_to.setdefault(edge.target_id, []).append(edge)
        return cls(
            root=root,
            concepts=concepts,
            edges_from=edges_from,
            edges_to=edges_to,
            index=index,
            skipped_files=failures,
        )

    @staticmethod
    def _load_index(
        root: Path,
        concepts: dict[str, Concept],
        failures: dict[str, str],
    ) -> BundleIndex | None:
        """Parse the root ``index.md`` or synthesize one in memory."""
        index_path = root / _ROOT_INDEX
        if not index_path.is_file():
            return synthesize_bundle_index(list(concepts.values()))
        try:
            raw = index_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures[_ROOT_INDEX] = f"index could not be read: {exc}"
            return None
        try:
            return parse_bundle_index(raw, source=_ROOT_INDEX)
        except BundleValidationError as exc:
            failures.update(exc.failed_files)
            return None

    @property
    def root(self) -> Path:
        """The resolved bundle root directory."""
        return self._root

    @property
    def concept_count(self) -> int:
        """Number of loaded concepts."""
        return len(self._concepts)

    @property
    def skipped_files(self) -> dict[str, str]:
        """Root-relative paths of files excluded by ``on_error="skip"``.

        Maps each skipped path to the reason it was excluded, sorted the
        same way as :attr:`BundleValidationError.failed_files`. Always
        empty when the bundle was loaded with the default
        ``on_error="raise"``, since any failure would have raised instead
        of producing a loaded bundle.
        """
        return dict(self._skipped_files)

    def get(self, concept_id: str) -> Concept:
        """Return the concept with the given ID.

        Raises:
            ConceptNotFoundError: If ``concept_id`` is not in the bundle.
        """
        try:
            return self._concepts[concept_id]
        except KeyError:
            raise ConceptNotFoundError(concept_id) from None

    def index(self) -> BundleIndex:
        """Return a copy of the parsed or synthesized root index."""
        return self._index.model_copy(deep=True)

    def all_concepts(self) -> list[Concept]:
        """Return every concept, sorted by concept ID."""
        return [self._concepts[concept_id] for concept_id in sorted(self._concepts)]

    def search(self, query: str, top_k: int = 5) -> list[Concept]:
        """Weighted, case-insensitive lexical search over all concepts.

        Each case-folded query token partially matches (substring) the
        title (weight 4), tags (3), description (2), and body (1); a
        concept must match at least one token. Results rank by descending
        score, then concept ID, capped at ``top_k``.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        tokens = re.findall(r"\w+", query.casefold())
        if not tokens:
            return []
        scored: list[tuple[int, str]] = []
        for concept_id in sorted(self._concepts):
            concept = self._concepts[concept_id]
            frontmatter = concept.frontmatter
            title = (frontmatter.title or "").casefold()
            tags = [tag.casefold() for tag in frontmatter.tags]
            description = (frontmatter.description or "").casefold()
            body = concept.body.casefold()
            score = 0
            for token in tokens:
                if token in title:
                    score += _TITLE_WEIGHT
                if any(token in tag for tag in tags):
                    score += _TAGS_WEIGHT
                if token in description:
                    score += _DESCRIPTION_WEIGHT
                if token in body:
                    score += _BODY_WEIGHT
            if score > 0:
                scored.append((score, concept_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._concepts[concept_id] for _, concept_id in scored[:top_k]]

    def links_from(self, concept_id: str) -> list[LinkEdge]:
        """Return copies of the outbound edges of a concept in document order.

        Unresolved edges (broken links) are included with
        ``resolved=False``.

        Raises:
            ConceptNotFoundError: If ``concept_id`` is not in the bundle.
        """
        self.get(concept_id)
        return [edge.model_copy() for edge in self._edges_from.get(concept_id, [])]

    def backlinks(self, concept_id: str) -> list[LinkEdge]:
        """Return copies of the inbound edges of a concept.

        Edges are ordered by source concept ID, then document order
        within each source.

        Raises:
            ConceptNotFoundError: If ``concept_id`` is not in the bundle.
        """
        self.get(concept_id)
        return [edge.model_copy() for edge in self._edges_to.get(concept_id, [])]

    def neighbors(
        self,
        concept_id: str,
        hops: int = 1,
        direction: Literal["out", "in", "both"] = "out",
    ) -> list[Concept]:
        """Return concepts reachable within ``hops`` links of a concept.

        Breadth-first over resolved edges only; the starting concept is
        excluded and cycles terminate. Results are ordered by distance,
        then concept ID.

        Raises:
            ConceptNotFoundError: If ``concept_id`` is not in the bundle.
            ValueError: If ``hops`` is negative or ``direction`` is not
                one of ``"out"``, ``"in"``, or ``"both"``.
        """
        self.get(concept_id)
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
        if direction == "out":
            adjacency = self._out_adjacency
        elif direction == "in":
            adjacency = self._in_adjacency
        else:
            adjacency = self._both_adjacency
        reachable = breadth_first_reachable(adjacency, concept_id, hops)
        return [self._concepts[reached_id] for reached_id in reachable]
