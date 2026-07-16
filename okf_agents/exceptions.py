"""Exception hierarchy for okf-agents.

All package exceptions derive from :class:`OKFError` so callers can catch
one base type. Validation and lookup exceptions keep their structured
inputs as attributes and render deterministic human-readable messages.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

__all__ = [
    "BundleNotFoundError",
    "BundleValidationError",
    "ConceptNotFoundError",
    "LinkResolutionError",
    "OKFError",
]


class OKFError(Exception):
    """Base class for all okf-agents errors."""


class BundleNotFoundError(OKFError):
    """Raised when a bundle root does not exist or is not a directory.

    Attributes:
        path: The path that was passed to :meth:`OKFBundle.load`.
        reason: ``"missing"`` if ``path`` does not exist (or could not be
            read), ``"not_a_directory"`` if it exists but is a file.
    """

    def __init__(
        self, path: str, *, reason: Literal["missing", "not_a_directory"] = "missing"
    ) -> None:
        self.path = path
        self.reason = reason
        if reason == "not_a_directory":
            message = f"OKF bundle path exists but is not a directory: {path}"
        else:
            message = f"OKF bundle not found: {path}"
        super().__init__(message)


class BundleValidationError(OKFError):
    """Raised when one or more concept files in a bundle are invalid.

    Attributes:
        failed_files: Root-relative paths of invalid files, sorted, with
            one reason per path.
    """

    def __init__(self, failed_files: dict[str, str] | Sequence[str]) -> None:
        if isinstance(failed_files, dict):
            self.failed_files: dict[str, str] = dict(sorted(failed_files.items()))
        else:
            self.failed_files = dict.fromkeys(sorted(failed_files), "invalid concept file")
        count = len(self.failed_files)
        noun = "file" if count == 1 else "files"
        details = "; ".join(f"{path}: {reason}" for path, reason in self.failed_files.items())
        super().__init__(f"OKF bundle validation failed for {count} {noun}: {details}")


class ConceptNotFoundError(OKFError):
    """Raised when a concept ID is not present in the bundle."""

    def __init__(self, concept_id: str) -> None:
        self.concept_id = concept_id
        super().__init__(f"Concept not found in bundle: {concept_id}")


class LinkResolutionError(OKFError):
    """Deprecated: never raised by this library.

    Unresolvable internal links are tolerated, not raised: they surface as
    :class:`~okf_agents.models.LinkEdge` instances with ``resolved=False``
    from :meth:`OKFBundle.links_from` and :meth:`OKFBundle.backlinks`, and
    every consumer in this package (agent tools, the navigator) is built
    to handle that case rather than encounter this exception. This class
    is kept only so existing ``from okf_agents import LinkResolutionError``
    imports keep working; it is deprecated and will be removed in a future
    release. Instantiating it emits a ``DeprecationWarning``.
    """

    def __init__(self, source_id: str, target: str) -> None:
        warnings.warn(
            "LinkResolutionError is deprecated and never raised by okf-agents; "
            "unresolvable links are represented by LinkEdge(resolved=False) "
            "instead. It will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.source_id = source_id
        self.target = target
        super().__init__(f"Cannot resolve link {target!r} from concept {source_id!r}")
