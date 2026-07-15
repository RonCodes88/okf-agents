"""Offline integration tests for vector-store sync and the graph retriever.

These tests run by default in CI, without secrets: they use an
in-process ``InProcessVectorStore`` paired with deterministic, dependency
-free fake embeddings (see ``tests/integration/conftest.py``) instead of
a real embedding provider. They cover interoperation across component
boundaries that unit tests exercise in isolation: syncing a real bundle
into a real (in-process) vector store, expanding semantic hits through
the link graph, and composing the router node with the navigator
subgraph inside one compiled LangGraph graph.
"""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict, cast

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from langgraph_okf.bundle import OKFBundle
from langgraph_okf.indexing import sync_bundle_to_vector_store
from langgraph_okf.navigator import create_okf_navigator
from langgraph_okf.retriever import OKFGraphRetriever
from langgraph_okf.router import Route, RouterState, create_okf_router
from tests.integration.conftest import InProcessVectorStore

pytestmark = pytest.mark.integration


class ScriptedChatModel(BaseChatModel):
    """Replays scripted JSON responses; fails on any extra model call.

    Used only to drive the navigator deterministically while proving
    graph composition, not to test navigator behavior itself (that is
    Task 06's responsibility).
    """

    responses: list[str]
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        if len(self.calls) > len(self.responses):
            raise AssertionError(
                f"unexpected model call #{len(self.calls)}; only "
                f"{len(self.responses)} responses were scripted"
            )
        message = AIMessage(content=self.responses[len(self.calls) - 1])
        return ChatResult(generations=[ChatGeneration(message=message)])


def _plan_json(*concept_ids: str) -> str:
    return json.dumps({"concept_ids": list(concept_ids)})


def _answer_json(answer: str, *citations: str) -> str:
    return json.dumps({"answer": answer, "citations": list(citations)})


def _result_ids(documents: list[Any]) -> list[str]:
    return [document.metadata["concept_id"] for document in documents]


class TestIdempotentSync:
    def test_second_sync_only_skips(
        self, bundle: OKFBundle, vector_store: InProcessVectorStore
    ) -> None:
        first = sync_bundle_to_vector_store(bundle, vector_store)
        assert first.added == bundle.concept_count
        assert first.updated == 0
        assert first.skipped == 0
        assert first.failed == 0

        second = sync_bundle_to_vector_store(bundle, vector_store)
        assert second.added == 0
        assert second.updated == 0
        assert second.skipped == bundle.concept_count
        assert second.failed == 0

    def test_second_sync_does_not_duplicate_documents(
        self, bundle: OKFBundle, vector_store: InProcessVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, vector_store)
        sync_bundle_to_vector_store(bundle, vector_store)
        assert len(vector_store.storage) == bundle.concept_count


class TestGraphRetrieverOverRealVectorSearch:
    """The word "cycle" appears only in concepts/customers' body, which
    links to concepts/orders, so these assertions do not depend on a
    model choosing between multiple plausible routes.
    """

    def test_semantic_hit_expands_through_links(
        self, bundle: OKFBundle, vector_store: InProcessVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, vector_store)
        retriever = OKFGraphRetriever(bundle=bundle, vector_store=vector_store, top_k=1)
        results = _result_ids(retriever.invoke("cycle"))
        assert results[0] == "concepts/customers"
        assert "concepts/orders" in results

    def test_expand_hops_zero_returns_only_the_entry_hit(
        self, bundle: OKFBundle, vector_store: InProcessVectorStore
    ) -> None:
        sync_bundle_to_vector_store(bundle, vector_store)
        retriever = OKFGraphRetriever(
            bundle=bundle, vector_store=vector_store, top_k=1, expand_hops=0
        )
        results = _result_ids(retriever.invoke("cycle"))
        assert results == ["concepts/customers"]


class _HybridState(TypedDict):
    """Parent state bridging the router's ``query`` and navigator's ``question``."""

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


class TestRouterNavigatorComposition:
    def test_router_conditional_edge_feeds_the_navigator_subgraph(
        self, bundle: OKFBundle, vector_store: InProcessVectorStore
    ) -> None:
        router_node = create_okf_router(bundle, vector_store=vector_store)
        model = ScriptedChatModel(
            responses=[
                _plan_json("concepts/orders"),
                _answer_json("Orders track sales.", "concepts/orders"),
            ]
        )
        navigator = create_okf_navigator(bundle, model, max_hops=1)

        def route(state: _HybridState) -> dict[str, Route]:
            return router_node(cast(RouterState, {"query": state["query"]}))

        def to_navigator_input(state: _HybridState) -> dict[str, str]:
            return {"question": state["query"]}

        def pick_route(state: _HybridState) -> str:
            picked = state["route"]
            assert picked is not None
            return picked

        graph: StateGraph[_HybridState, None, _HybridState, _HybridState] = StateGraph(
            _HybridState
        )
        graph.add_node("router", route)
        graph.add_node("prepare", to_navigator_input)
        graph.add_node("navigate", navigator)
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router", pick_route, {"bundle": "prepare", "vector": "prepare", "both": "prepare"}
        )
        graph.add_edge("prepare", "navigate")
        graph.add_edge("navigate", END)
        app = graph.compile()

        result = app.invoke({"query": "Orders"})
        assert result["route"] == "bundle"
        assert result["traversal_path"][0] == "index"
        assert result["answer"] == "Orders track sales."
        assert result["citations"] == ["concepts/orders"]
        assert len(model.calls) == 2
