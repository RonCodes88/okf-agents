"""Opt-in provider-integration tests for the navigator against a real model.

Gated by ``RUN_INTEGRATION_TESTS=1`` and a supported provider key
(``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``, or an explicit
``OKF_TEST_PROVIDER``); skips cleanly otherwise, including when the
provider SDK is not installed. Assertions target observable,
provider-agnostic outcomes (budgets, citation validity, whether a known
concept was reached) rather than exact prose, per the module's
reliability rules.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from okf_agents.bundle import OKFBundle
from okf_agents.navigator import create_okf_navigator
from tests.provider_support import run_with_timeout

pytestmark = pytest.mark.integration

_TIMEOUT_SECONDS = 60.0

_HEDGE_PHRASES = (
    "not found",
    "no information",
    "cannot find",
    "can't find",
    "don't know",
    "do not know",
    "does not appear",
    "doesn't appear",
    "no mention",
    "not available",
    "not covered",
    "unable to find",
    "no data",
    "not contain",
    "not part of",
    "outside",
)


class TestGroundedAnswers:
    def test_general_question_stays_within_budgets(
        self, bundle: OKFBundle, integration_chat_model: BaseChatModel
    ) -> None:
        navigator = create_okf_navigator(
            bundle, integration_chat_model, max_hops=3, max_concepts=6, token_budget=4000
        )
        result = run_with_timeout(
            lambda: navigator.invoke({"question": "What tables are in this bundle?"}),
            seconds=_TIMEOUT_SECONDS,
        )
        assert isinstance(result["answer"], str) and result["answer"].strip()
        assert result["traversal_path"][0] == "index"
        assert result["hops"] <= 3
        assert len(result["visited"]) <= 6
        assert result["tokens_used"] <= 4000
        assert set(result["citations"]) <= set(result["visited"])
        assert result["citations"]

    def test_max_hops_zero_still_produces_an_answer(
        self, bundle: OKFBundle, integration_chat_model: BaseChatModel
    ) -> None:
        navigator = create_okf_navigator(bundle, integration_chat_model, max_hops=0)
        result = run_with_timeout(
            lambda: navigator.invoke({"question": "What tables are in this bundle?"}),
            seconds=_TIMEOUT_SECONDS,
        )
        assert result["traversal_path"] == ["index"]
        assert result["visited"] == []
        assert isinstance(result["answer"], str) and result["answer"].strip()

    def test_payment_question_reaches_the_payments_concept(
        self, bundle: OKFBundle, integration_chat_model: BaseChatModel
    ) -> None:
        navigator = create_okf_navigator(
            bundle, integration_chat_model, max_hops=3, max_concepts=6
        )
        result = run_with_timeout(
            lambda: navigator.invoke(
                {"question": "How are payments captured and refunded?"}
            ),
            seconds=_TIMEOUT_SECONDS,
        )
        assert (
            "concepts/payments" in result["citations"]
            or "concepts/payments" in result["traversal_path"]
        )

    def test_unsupported_question_yields_a_hedged_answer(
        self, bundle: OKFBundle, integration_chat_model: BaseChatModel
    ) -> None:
        navigator = create_okf_navigator(
            bundle, integration_chat_model, max_hops=3, max_concepts=5
        )
        result = run_with_timeout(
            lambda: navigator.invoke({"question": "What is the CEO's home address?"}),
            seconds=_TIMEOUT_SECONDS,
        )
        answer = (result["answer"] or "").casefold()
        assert any(phrase in answer for phrase in _HEDGE_PHRASES), (
            f"expected a hedged/not-found answer, got: {answer!r}"
        )
        assert len(result["citations"]) <= 1
