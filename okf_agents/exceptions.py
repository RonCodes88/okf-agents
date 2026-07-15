"""Exception hierarchy for okf-agents.

All package exceptions derive from :class:`OKFError` so callers can catch
one base type. Validation and lookup exceptions keep their structured
inputs as attributes and render deterministic human-readable messages.
"""

from __future__ import annotations

from collections.abc import Sequence

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
    """Raised when a bundle root does not exist or is not a directory."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"OKF bundle not found: {path}")


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
    """Raised when an internal link cannot be resolved on demand."""

    def __init__(self, source_id: str, target: str) -> None:
        self.source_id = source_id
        self.target = target
        super().__init__(f"Cannot resolve link {target!r} from concept {source_id!r}")
