"""Pure parsing functions for OKF v0.1 concept files and bundle indexes.

Every function is side-effect free: no filesystem writes and no network
access. Invalid input raises
:class:`~okf_agents.exceptions.BundleValidationError` keyed by the
root-relative source path so the bundle loader can aggregate failures.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from okf_agents.exceptions import BundleValidationError
from okf_agents.models import BundleIndex, Concept, ConceptFrontmatter, LinkEdge

__all__ = [
    "extract_internal_links",
    "normalize_link_target",
    "parse_bundle_index",
    "parse_concept",
    "parse_frontmatter",
    "parse_wikilink_target",
    "split_frontmatter",
    "synthesize_bundle_index",
]

_FRONTMATTER_DELIMITER = "---"
_STANDARD_FRONTMATTER_KEYS = (
    "type",
    "title",
    "description",
    "resource",
    "tags",
    "aliases",
    "timestamp",
)
_MD_SUFFIX = ".md"

# Standard inline links only: images are excluded by the lookbehind, and
# reference-style links and autolinks never match because they lack "(...)".
_INLINE_LINK_RE = re.compile(
    r"(?<!!)\[(?P<anchor>[^\]]*)\]\((?P<target>[^()\s]+)(?:\s+\"[^\"]*\")?\)"
)
# Obsidian-style wikilinks: [[target]], [[target|Display]],
# [[target#Heading]], [[target^blockid]]. The lookbehind excludes
# ![[embed]] file/block embeds, mirroring how "!" excludes image syntax
# from _INLINE_LINK_RE above.
_WIKILINK_RE = re.compile(r"(?<!!)\[\[(?P<inner>[^\[\]]+)\]\]")
_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")
_H1_RE = re.compile(r"^# +(?P<title>.+?) *#* *$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)]) ")


def _invalid(source: str, reason: str) -> BundleValidationError:
    return BundleValidationError({source: reason})


def split_frontmatter(raw: str, *, source: str) -> tuple[dict[str, Any], str]:
    """Split raw concept text into its YAML frontmatter mapping and body.

    The text must start with a ``---`` line and contain a matching closing
    ``---`` line; both delimiters accept LF or CRLF endings. The returned
    body excludes the frontmatter and both delimiter lines, preserving the
    original line endings.

    Raises:
        BundleValidationError: If a delimiter is missing, the YAML is
            malformed, or the frontmatter is not a mapping.
    """
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != _FRONTMATTER_DELIMITER:
        raise _invalid(source, "missing opening '---' frontmatter delimiter")
    close_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == _FRONTMATTER_DELIMITER
        ),
        None,
    )
    if close_index is None:
        raise _invalid(source, "missing closing '---' frontmatter delimiter")
    yaml_text = "".join(lines[1:close_index])
    body = "".join(lines[close_index + 1 :])
    try:
        mapping = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise _invalid(source, f"malformed YAML frontmatter: {exc}") from exc
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise _invalid(source, "frontmatter must be a YAML mapping")
    return mapping, body


def parse_frontmatter(mapping: dict[str, Any], *, source: str) -> ConceptFrontmatter:
    """Validate a frontmatter mapping into a :class:`ConceptFrontmatter`.

    Standard keys become typed fields; every unknown key is kept only in
    ``extra``.

    Raises:
        BundleValidationError: If ``type`` is missing or empty, or any
            field has an invalid type.
    """
    standard = {key: mapping[key] for key in _STANDARD_FRONTMATTER_KEYS if key in mapping}
    extra = {key: value for key, value in mapping.items() if key not in _STANDARD_FRONTMATTER_KEYS}
    if "type" not in standard:
        raise _invalid(source, "frontmatter is missing required key 'type'")
    try:
        return ConceptFrontmatter(**standard, extra=extra)
    except ValidationError as exc:
        reasons = "; ".join(
            f"{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise _invalid(source, f"invalid frontmatter: {reasons}") from exc


def normalize_link_target(target: str, *, source_id: str, source: str) -> str | None:
    """Normalize an inline-link target to a concept ID, or return ``None``.

    ``None`` marks targets that are not internal Markdown links: external
    URLs (any scheme or protocol-relative), fragment-only links, and
    targets without a ``.md`` suffix. Fragments are stripped first.
    Bundle-relative targets (``/path/file.md``) resolve from the bundle
    root; all other targets resolve from the source concept's directory.

    Raises:
        BundleValidationError: If the target traverses outside the bundle.
    """
    if _URL_SCHEME_RE.match(target) or target.startswith("//"):
        return None
    path_part = target.split("#", 1)[0]
    if not path_part.endswith(_MD_SUFFIX):
        return None
    if path_part.startswith("/"):
        candidate = posixpath.normpath(path_part.lstrip("/"))
    else:
        candidate = posixpath.normpath(posixpath.join(posixpath.dirname(source_id), path_part))
    if candidate == ".." or candidate.startswith("../"):
        raise _invalid(source, f"link target escapes the bundle: {target!r}")
    return candidate[: -len(_MD_SUFFIX)]


def parse_wikilink_target(inner: str) -> tuple[str, str]:
    """Split one ``[[...]]`` body into ``(lookup_key, display_text)``.

    ``inner`` is the raw text between the double brackets, e.g.
    ``"Note#Heading|Display"``. A ``|`` splits the target from its display
    alias; without one, the display text is the raw target as written
    (anchor included), matching what Obsidian renders. The lookup key
    drops any ``#heading`` or ``^blockid`` anchor and a trailing ``.md``
    suffix, then casefolds the result for case-insensitive resolution —
    anchors are not resolved to a location within the target; both
    ``[[Note]]`` and ``[[Note#Heading]]`` point at the whole target
    concept, since sub-file anchors are out of scope for this library's
    concept-level granularity.
    """
    if "|" in inner:
        target_part, display = inner.split("|", 1)
    else:
        target_part, display = inner, inner
    target_part = target_part.strip()
    display = display.strip()
    found = (target_part.find("#"), target_part.find("^"))
    anchor_index = min((index for index in found if index != -1), default=-1)
    base = target_part if anchor_index == -1 else target_part[:anchor_index]
    base = base.strip()
    if base.casefold().endswith(_MD_SUFFIX):
        base = base[: -len(_MD_SUFFIX)]
    return base.casefold(), display


def extract_internal_links(body: str, *, source_id: str, source: str) -> list[LinkEdge]:
    """Extract internal links as unresolved edges, in document order.

    Recognizes two syntaxes: standard inline Markdown links,
    ``[text](target.md)`` (``link_kind="markdown"``, ``target_id`` is the
    path-normalized concept ID candidate from
    :func:`normalize_link_target`), and Obsidian-style wikilinks,
    ``[[target]]``/``[[target|Display]]``/``[[target#Heading]]``
    (``link_kind="wiki"``, ``target_id`` is a casefolded lookup key —
    filename, title, or alias — resolved against the whole bundle later by
    :class:`~okf_agents.bundle.OKFBundle`, not by this function). Both
    kinds are returned unresolved (``resolved=False``); repeats are
    retained so callers can build a multigraph. Image syntax
    (``![...]()``), file/block embeds (``![[...]]``), links inside fenced
    code blocks, and Markdown targets rejected by
    :func:`normalize_link_target` are ignored.

    Raises:
        BundleValidationError: If any Markdown link target escapes the
            bundle.
    """
    edges: list[LinkEdge] = []
    fence_marker: str | None = None
    for line in body.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue
        line_matches: list[tuple[int, LinkEdge]] = []
        for match in _INLINE_LINK_RE.finditer(line):
            target_id = normalize_link_target(
                match.group("target"), source_id=source_id, source=source
            )
            if target_id is not None:
                line_matches.append(
                    (
                        match.start(),
                        LinkEdge(
                            source_id=source_id,
                            target_id=target_id,
                            anchor_text=match.group("anchor"),
                            link_kind="markdown",
                        ),
                    )
                )
        for match in _WIKILINK_RE.finditer(line):
            lookup_key, display = parse_wikilink_target(match.group("inner"))
            if not lookup_key:
                continue
            line_matches.append(
                (
                    match.start(),
                    LinkEdge(
                        source_id=source_id,
                        target_id=lookup_key,
                        anchor_text=display,
                        link_kind="wiki",
                    ),
                )
            )
        line_matches.sort(key=lambda item: item[0])
        edges.extend(edge for _, edge in line_matches)
    return edges


def parse_concept(raw: str, *, bundle_root: Path, file_path: Path) -> Concept:
    """Parse one concept file from raw text plus its bundle and file paths.

    The returned model preserves ``raw`` verbatim, excludes the
    frontmatter delimiters from ``body``, exposes ``path`` as the resolved
    absolute path string, and deduplicates ``outbound_links`` in
    first-seen order. ``outbound_links`` entries from wikilinks are the
    raw casefolded lookup key, not yet a real concept ID: resolving those
    against the rest of the bundle happens only in
    :meth:`~okf_agents.bundle.OKFBundle.load`, which is the first point
    with visibility into every concept's filename, title, and aliases.

    Raises:
        BundleValidationError: Keyed by the root-relative path when the
            frontmatter is invalid, a link escapes the bundle, or the file
            lies outside the bundle root.
    """
    resolved_root = bundle_root.resolve()
    resolved_file = file_path if file_path.is_absolute() else resolved_root / file_path
    resolved_file = resolved_file.resolve()
    try:
        relative = resolved_file.relative_to(resolved_root)
    except ValueError as exc:
        raise _invalid(str(file_path), "concept file lies outside the bundle root") from exc
    source = relative.as_posix()
    concept_id = source[: -len(_MD_SUFFIX)] if source.endswith(_MD_SUFFIX) else source

    mapping, body = split_frontmatter(raw, source=source)
    frontmatter = parse_frontmatter(mapping, source=source)
    edges = extract_internal_links(body, source_id=concept_id, source=source)
    outbound_links = list(dict.fromkeys(edge.target_id for edge in edges))
    return Concept(
        id=concept_id,
        path=str(resolved_file),
        frontmatter=frontmatter,
        body=body,
        outbound_links=outbound_links,
        raw=raw,
    )


def _extract_title_and_description(markdown: str) -> tuple[str | None, str | None]:
    """Return the first H1 title and first non-heading paragraph, if any."""
    title: str | None = None
    paragraph: list[str] = []
    fenced = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            if paragraph:
                break
            fenced = not fenced
            continue
        if fenced:
            continue
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            if paragraph:
                break
            h1 = _H1_RE.match(stripped)
            if h1 and title is None:
                title = h1.group("title")
            continue
        if not paragraph and _LIST_ITEM_RE.match(stripped):
            # A link list is not a descriptive paragraph.
            continue
        paragraph.append(stripped)
    description = " ".join(paragraph) if paragraph else None
    return title, description


def parse_bundle_index(raw: str, *, source: str = "index.md") -> BundleIndex:
    """Parse a root ``index.md`` into a :class:`BundleIndex`.

    ``body`` is the original Markdown, ``title`` the first H1 when
    present, ``description`` the first non-heading paragraph when present,
    and ``concept_ids`` the normalized internal link targets deduplicated
    in first-seen order.

    Raises:
        BundleValidationError: If any link target escapes the bundle.
    """
    title, description = _extract_title_and_description(raw)
    edges = extract_internal_links(raw, source_id="index", source=source)
    concept_ids = list(dict.fromkeys(edge.target_id for edge in edges))
    return BundleIndex(title=title, description=description, body=raw, concept_ids=concept_ids)


def synthesize_bundle_index(concepts: Sequence[Concept], *, title: str = "Index") -> BundleIndex:
    """Synthesize an in-memory index for a bundle without a root ``index.md``.

    Concepts are listed in ascending concept-ID order as bundle-relative
    Markdown links labelled by frontmatter title when available. Nothing
    is written to disk.
    """
    ordered = sorted(concepts, key=lambda concept: concept.id)
    lines = [f"# {title}", ""]
    lines.extend(
        f"- [{concept.frontmatter.title or concept.id}](/{concept.id}{_MD_SUFFIX})"
        for concept in ordered
    )
    body = "\n".join(lines) + "\n"
    return BundleIndex(
        title=title,
        description=None,
        body=body,
        concept_ids=[concept.id for concept in ordered],
    )
