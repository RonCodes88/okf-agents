"""Unit tests for the query-routing LangGraph node."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from okf_agents.bundle import OKFBundle
from okf_agents.router import Route, RouterState, create_okf_router

pytestmark = pytest.mark.unit


class StubLLM(LLM):
    """Offline classifier stub with a canned reply and call recording."""

    response: str = "bundle"
    error: RuntimeError | None = None
    prompts: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response


class ListContentChatModel(BaseChatModel):
    """Chat model stub whose reply content is a block list, not a string."""

    @property
    def _llm_type(self) -> str:
        return "stub-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=[{"type": "text", "text": "vector"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class StubVectorStore(VectorStore):
    """Minimal concrete vector store; the router only checks presence."""

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return []

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[Any, Any]] | None = None,
        **kwargs: Any,
    ) -> StubVectorStore:
        return cls()


@pytest.fixture
def vector_store() -> StubVectorStore:
    return StubVectorStore()


class TestHeuristicRouting:
    @pytest.mark.parametrize("query", ["Orders", "orders", "  ORDERS  ", "getting-started"])
    def test_exact_title_match_routes_to_bundle(
        self, bundle: OKFBundle, vector_store: StubVectorStore, query: str
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)
        assert node({"query": query}) == {"route": "bundle"}

    @pytest.mark.parametrize("query", ["billing", "SALES"])
    def test_exact_tag_match_routes_to_bundle(
        self, bundle: OKFBundle, vector_store: StubVectorStore, query: str
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)
        assert node({"query": query}) == {"route": "bundle"}

    @pytest.mark.parametrize("query", ["order", "getting", "orders and refunds"])
    def test_partial_title_does_not_match(
        self, bundle: OKFBundle, vector_store: StubVectorStore, query: str
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)
        assert node({"query": query}) == {"route": "vector"}

    def test_vague_query_with_vector_store_routes_to_vector(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)
        assert node({"query": "how do refunds flow?"}) == {"route": "vector"}

    def test_vague_query_without_vector_store_routes_to_bundle(self, bundle: OKFBundle) -> None:
        node = create_okf_router(bundle)
        assert node({"query": "how do refunds flow?"}) == {"route": "bundle"}

    def test_heuristic_is_deterministic(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)
        assert [node({"query": "payments"}) for _ in range(3)] == [{"route": "bundle"}] * 3


class TestClassifierRouting:
    @pytest.mark.parametrize("route", ["bundle", "vector", "both"])
    def test_classifier_route_is_used(
        self, bundle: OKFBundle, vector_store: StubVectorStore, route: Route
    ) -> None:
        classifier = StubLLM(response=route)
        node = create_okf_router(bundle, vector_store=vector_store, classifier=classifier)
        assert node({"query": "how do refunds flow?"}) == {"route": route}
        assert len(classifier.prompts) == 1

    def test_classifier_prompt_contains_query(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        classifier = StubLLM(response="vector")
        node = create_okf_router(bundle, vector_store=vector_store, classifier=classifier)
        node({"query": "refund lifecycle"})
        assert "refund lifecycle" in classifier.prompts[0]

    @pytest.mark.parametrize("route", ["vector", "both"])
    def test_route_coerced_to_bundle_without_vector_store(
        self, bundle: OKFBundle, route: str
    ) -> None:
        classifier = StubLLM(response=route)
        node = create_okf_router(bundle, classifier=classifier)
        assert node({"query": "how do refunds flow?"}) == {"route": "bundle"}

    @pytest.mark.parametrize(
        "response",
        ["no idea", "", "bundle or vector, hard to say", '{"answer": 42}'],
    )
    def test_malformed_output_falls_back_to_heuristic_once(
        self, bundle: OKFBundle, vector_store: StubVectorStore, response: str
    ) -> None:
        classifier = StubLLM(response=response)
        node = create_okf_router(bundle, vector_store=vector_store, classifier=classifier)
        assert node({"query": "Orders"}) == {"route": "bundle"}
        assert node({"query": "how do refunds flow?"}) == {"route": "vector"}
        assert len(classifier.prompts) == 2  # one model call per invocation, no retries

    def test_non_string_content_falls_back_to_heuristic(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        node = create_okf_router(
            bundle, vector_store=vector_store, classifier=ListContentChatModel()
        )
        assert node({"query": "Orders"}) == {"route": "bundle"}

    def test_wordy_but_unambiguous_output_is_accepted(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        classifier = StubLLM(response="Route: VECTOR.")
        node = create_okf_router(bundle, vector_store=vector_store, classifier=classifier)
        assert node({"query": "Orders"}) == {"route": "vector"}

    def test_model_runtime_error_propagates(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        classifier = StubLLM(error=RuntimeError("model unavailable"))
        node = create_okf_router(bundle, vector_store=vector_store, classifier=classifier)
        with pytest.raises(RuntimeError, match="model unavailable"):
            node({"query": "Orders"})


class TestNodeContract:
    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_query_raises_value_error(self, bundle: OKFBundle, query: str) -> None:
        classifier = StubLLM()
        node = create_okf_router(bundle, classifier=classifier)
        with pytest.raises(ValueError, match="query"):
            node({"query": query})
        assert classifier.prompts == []

    def test_update_contains_only_route_and_state_is_untouched(self, bundle: OKFBundle) -> None:
        node = create_okf_router(bundle)
        state: RouterState = {
            "query": "Orders",
            "route": None,
            "retriever_result": [Document(page_content="stale")],
        }
        snapshot = dict(state)
        update = node(state)
        assert update == {"route": "bundle"}
        assert dict(state) == snapshot

    def test_conditional_edges_in_compiled_graph(
        self, bundle: OKFBundle, vector_store: StubVectorStore
    ) -> None:
        node = create_okf_router(bundle, vector_store=vector_store)

        def router(state: RouterState) -> dict[str, Route]:
            return node(state)

        def pick(state: RouterState) -> Route:
            route = state.get("route")
            assert route is not None
            return route

        def branch(name: str) -> Any:
            def run(state: RouterState) -> dict[str, list[Document]]:
                return {"retriever_result": [Document(page_content=name)]}

            return run

        graph = StateGraph(RouterState)
        graph.add_node("router", router)
        graph.add_node("bundle_node", branch("bundle"))
        graph.add_node("vector_node", branch("vector"))
        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router", pick, {"bundle": "bundle_node", "vector": "vector_node"}
        )
        graph.add_edge("bundle_node", END)
        graph.add_edge("vector_node", END)
        compiled = graph.compile()

        exact = compiled.invoke({"query": "Orders"})
        assert exact["route"] == "bundle"
        assert [doc.page_content for doc in exact["retriever_result"]] == ["bundle"]

        vague = compiled.invoke({"query": "how do refunds flow?"})
        assert vague["route"] == "vector"
        assert [doc.page_content for doc in vague["retriever_result"]] == ["vector"]
