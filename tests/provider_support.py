"""Shared helpers for opt-in provider-integration and e2e tests.

Provider SDKs (``langchain-anthropic``, ``langchain-openai``) are
imported lazily inside :func:`build_chat_model` so unit tests, offline
integration tests, and plain package imports never depend on them, and
so a missing SDK becomes a clean test skip rather than a collection
error.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar, cast

import pytest
from langchain_core.language_models import BaseChatModel

__all__ = ["build_chat_model", "run_with_timeout"]

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
}

_RESULT = TypeVar("_RESULT")


def run_with_timeout(func: Callable[[], _RESULT], *, seconds: float) -> _RESULT:
    """Run ``func`` on a worker thread and fail the test past ``seconds``.

    Used instead of a ``pytest-timeout`` dependency: a hung provider call
    fails the offending test instead of hanging the whole suite.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=seconds)
        except FutureTimeoutError:
            pytest.fail(f"operation exceeded {seconds:.0f}s timeout")


def build_chat_model() -> BaseChatModel:
    """Construct a real chat model from environment configuration.

    Provider is ``OKF_TEST_PROVIDER`` (``"anthropic"`` or ``"openai"``),
    defaulting to whichever supported API key is present. Model name is
    ``OKF_TEST_MODEL``, defaulting to a small, low-cost model per
    provider so tests never hard-code a specific paid model. Skips the
    calling test with a clear reason when no supported key or SDK is
    available.
    """
    provider = os.environ.get("OKF_TEST_PROVIDER")
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            pytest.skip(
                "no supported provider key set: export ANTHROPIC_API_KEY, "
                "OPENAI_API_KEY, or set OKF_TEST_PROVIDER"
            )
    model_name = os.environ.get("OKF_TEST_MODEL") or _DEFAULT_MODELS.get(provider, "")
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY is not set")
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("langchain-anthropic is not installed")
        return cast(
            BaseChatModel, ChatAnthropic(model=model_name, timeout=60.0, max_retries=0)
        )
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is not set")
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip("langchain-openai is not installed")
        return cast(BaseChatModel, ChatOpenAI(model=model_name, timeout=60.0, max_retries=0))
    pytest.skip(f"unsupported OKF_TEST_PROVIDER={provider!r}")
