"""Fixtures shared by the end-to-end test tier.

``e2e_chat_model`` gates :mod:`tests.e2e.test_full_workflow` behind
``RUN_E2E_TESTS=1`` and a real provider key, skipping cleanly otherwise
(including when the provider SDK is not installed).
"""

from __future__ import annotations

import os

import pytest
from langchain_core.language_models import BaseChatModel

from tests.provider_support import build_chat_model

RUN_E2E_TESTS = os.environ.get("RUN_E2E_TESTS") == "1"


@pytest.fixture()
def e2e_chat_model() -> BaseChatModel:
    """A real, tool-capable chat model, skipping the test when e2e runs are disabled."""
    if not RUN_E2E_TESTS:
        pytest.skip("set RUN_E2E_TESTS=1 to run end-to-end tests")
    return build_chat_model()
