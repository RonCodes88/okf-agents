"""Unit tests for the okf-agents exception hierarchy."""

import pytest

from okf_agents import __version__
from okf_agents.exceptions import (
    BundleNotFoundError,
    BundleValidationError,
    ConceptNotFoundError,
    LinkResolutionError,
    OKFError,
)

pytestmark = pytest.mark.unit


def test_version() -> None:
    assert __version__ == "0.1.0"


@pytest.mark.parametrize(
    "exc",
    [
        BundleNotFoundError("/missing/bundle"),
        BundleValidationError({"a.md": "bad yaml"}),
        ConceptNotFoundError("concepts/orders"),
        LinkResolutionError("concepts/orders", "../missing.md"),
    ],
)
def test_all_exceptions_inherit_from_okf_error(exc: OKFError) -> None:
    assert isinstance(exc, OKFError)
    assert isinstance(exc, Exception)


def test_bundle_not_found_preserves_path() -> None:
    exc = BundleNotFoundError("/data/kb")
    assert exc.path == "/data/kb"
    assert str(exc) == "OKF bundle not found: /data/kb"


def test_bundle_validation_error_sorts_failed_files() -> None:
    exc = BundleValidationError({"b/two.md": "empty type", "a/one.md": "bad yaml"})
    assert list(exc.failed_files) == ["a/one.md", "b/two.md"]
    assert str(exc) == (
        "OKF bundle validation failed for 2 files: a/one.md: bad yaml; b/two.md: empty type"
    )


def test_bundle_validation_error_singular_message() -> None:
    exc = BundleValidationError({"a.md": "bad yaml"})
    assert str(exc) == "OKF bundle validation failed for 1 file: a.md: bad yaml"


def test_bundle_validation_error_accepts_path_sequence() -> None:
    exc = BundleValidationError(["b.md", "a.md"])
    assert exc.failed_files == {
        "a.md": "invalid concept file",
        "b.md": "invalid concept file",
    }
    assert str(exc) == (
        "OKF bundle validation failed for 2 files: "
        "a.md: invalid concept file; b.md: invalid concept file"
    )


def test_bundle_validation_error_message_is_deterministic() -> None:
    files = {"z.md": "r1", "a.md": "r2", "m.md": "r3"}
    reordered = dict(reversed(files.items()))
    assert str(BundleValidationError(files)) == str(BundleValidationError(reordered))


def test_concept_not_found_preserves_concept_id() -> None:
    exc = ConceptNotFoundError("concepts/orders")
    assert exc.concept_id == "concepts/orders"
    assert str(exc) == "Concept not found in bundle: concepts/orders"


def test_link_resolution_error_preserves_source_and_target() -> None:
    exc = LinkResolutionError("concepts/orders", "../missing.md")
    assert exc.source_id == "concepts/orders"
    assert exc.target == "../missing.md"
    assert str(exc) == "Cannot resolve link '../missing.md' from concept 'concepts/orders'"
