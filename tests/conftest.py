"""Shared pytest fixtures for the langgraph-okf test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_okf.bundle import OKFBundle

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_bundle_path() -> Path:
    """Path to the conformant sample bundle fixture."""
    return FIXTURES_DIR / "sample_bundle"


@pytest.fixture(scope="session")
def bundle(sample_bundle_path: Path) -> OKFBundle:
    """The sample bundle, loaded once per test session."""
    return OKFBundle.load(sample_bundle_path)
