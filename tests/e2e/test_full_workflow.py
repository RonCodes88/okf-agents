"""Opt-in end-to-end tests against a real chat model.

Gated by ``RUN_E2E_TESTS=1`` and a supported provider key
(``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``, or an explicit
``OKF_TEST_PROVIDER``); skips cleanly otherwise. Assertions target
observable outcomes and captured tool calls rather than exact prose,
per the module's reliability rules.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.navigator import create_okf_navigator
from langgraph_okf.router import Route, RouterState, create_okf_router
from langgraph_okf.tools import create_okf_tools
from tests.provider_support import run_with_timeout

pytestmark = pytest.mark.e2e

_TIMEOUT_SECONDS = 90.0
_READ_TOOLS = {"read_concept", "search_concepts"}


def _called_tool_names(messages: list[Any]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            names.add(call["name"])
    return names


class TestToolCallingAgent:
    def test_agent_explains_orders_and_customers_relationship(
        self, bundle: OKFBundle, e2e_chat_model: BaseChatModel
    ) -> None:
        tools = create_okf_tools(bundle)
        agent = create_react_agent(e2e_chat_model, tools)

        result = run_with_timeout(
            lambda: agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            "Explain the relationship between orders and customers."
                        )
                    ]
                }
            ),
            seconds=_TIMEOUT_SECONDS,
        )

        messages = result["messages"]
        assert _called_tool_names(messages) & _READ_TOOLS

        final = messages[-1]
        assert isinstance(final, AIMessage)
        content = final.content if isinstance(final.content, str) else str(final.content)
        lowered = content.casefold()
        assert "order" in lowered
        assert "customer" in lowered


class _ParentState(TypedDict):
    """Bridges the router's ``query`` and the navigator's ``question``."""

    query: str
    route: NotRequired[Route | None]
    question: NotRequired[str]
    visited: NotRequired[list[str]]
    context: NotRequired[list[str]]
    tokens_used: NotRequired[int]
    hops: NotRequired[int]
    answer: NotRequired[str | None]
    citations: NotRequired[list[str]]
    traversal_path: NotRequired[list[str]]


class TestRouterAndNavigatorParentGraph:
    def test_routing_happens_before_navigation_and_answer_is_grounded(
        self, bundle: OKFBundle, e2e_chat_model: BaseChatModel
    ) -> None:
        router_node = create_okf_router(bundle)
        navigator = create_okf_navigator(bundle, e2e_chat_model, max_hops=3, max_concepts=6)

        def route(state: _ParentState) -> dict[str, Route]:
            return router_node(cast(RouterState, {"query": state["query"]}))

        def to_navigator_input(state: _ParentState) -> dict[str, str]:
            assert state.get("route") is not None  # router must run before this node
            return {"question": state["query"]}

        graph: StateGraph[_ParentState, None, _ParentState, _ParentState] = StateGraph(
            _ParentState
        )
        graph.add_node("router", route)
        graph.add_node("prepare", to_navigator_input)
        graph.add_node("navigate", navigator)
        graph.add_edge(START, "router")
        graph.add_edge("router", "prepare")
        graph.add_edge("prepare", "navigate")
        graph.add_edge("navigate", END)
        app = graph.compile()

        result = run_with_timeout(
            lambda: app.invoke({"query": "How are payments captured and refunded?"}),
            seconds=_TIMEOUT_SECONDS,
        )

        assert result["route"] is not None
        assert isinstance(result["answer"], str) and result["answer"].strip()
        assert set(result["citations"]) <= set(result["visited"])
