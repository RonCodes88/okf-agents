"""LangGraph router node that classifies queries without retrieving.

:func:`create_okf_router` returns a node function that labels a query as
``bundle``, ``vector``, or ``both`` so downstream conditional edges can
branch to the matching retrieval path. Routing is a deterministic
offline heuristic by default; an optional language-model classifier can
override it, with validated output and a single heuristic fallback when
the model output is malformed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict, cast, get_args

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore

from okf_agents.bundle import OKFBundle

__all__ = ["Route", "RouterState", "create_okf_router"]

Route = Literal["bundle", "vector", "both"]

_ROUTES: tuple[str, ...] = get_args(Route)

_ROUTE_TOKEN = re.compile(r"\b(bundle|vector|both)\b")

_CLASSIFIER_PROMPT = """\
Classify how to retrieve knowledge for a user query over a documentation bundle.
Reply with exactly one word:
- bundle: the query names a specific known concept or fact
- vector: the query is vague or semantic and suits similarity search
- both: the query needs a known concept and semantic context

Query: {query}
Route:"""


class RouterState(TypedDict):
    """State schema consumed and updated by the router node.

    ``query`` is required. ``route`` and ``retriever_result`` belong to
    the parent graph; the node only ever emits a ``route`` update and
    never erases the other keys.
    """

    query: str
    route: NotRequired[Route | None]
    retriever_result: NotRequired[list[Document] | None]


def _normalize(text: str) -> str:
    """Case-fold and reduce text to space-separated word tokens."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def _extract_route(output: object) -> Route | None:
    """Validate classifier output down to exactly one route, else ``None``.

    Accepts a message-like object or plain string. The text must mention
    exactly one distinct route name; anything else is malformed.
    """
    content = getattr(output, "content", output)
    if not isinstance(content, str):
        return None
    found = set(_ROUTE_TOKEN.findall(content.casefold()))
    if len(found) != 1:
        return None
    route = found.pop()
    return cast(Route, route) if route in _ROUTES else None


def create_okf_router(
    bundle: OKFBundle,
    vector_store: VectorStore | None = None,
    classifier: BaseLanguageModel[Any] | None = None,
) -> Callable[[RouterState], dict[str, Route]]:
    """Build a LangGraph node that sets ``route`` for an incoming query.

    Without a classifier the route is a deterministic heuristic: the
    normalized query exact-matches a complete concept title or tag means
    ``bundle``; otherwise ``vector`` when a vector store exists, else
    ``bundle``. With a classifier, the model is invoked at most once and
    its validated output supplies the route; malformed output falls back
    to the heuristic exactly once, while model runtime errors propagate.
    Routes needing an absent vector store are coerced to ``bundle``.

    The node performs no retrieval, never mutates the incoming state,
    and returns an update containing only ``route``.
    """
    exact_terms = frozenset(
        _normalize(term)
        for concept in bundle.all_concepts()
        for term in [concept.frontmatter.title or "", *concept.frontmatter.tags]
        if term
    )

    def heuristic(query: str) -> Route:
        if _normalize(query) in exact_terms:
            return "bundle"
        return "vector" if vector_store is not None else "bundle"

    def router_node(state: RouterState) -> dict[str, Route]:
        query = state["query"]
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        route: Route | None = None
        if classifier is not None:
            output = classifier.invoke(_CLASSIFIER_PROMPT.format(query=query))
            route = _extract_route(output)
        if route is None:
            route = heuristic(query)
        if vector_store is None and route != "bundle":
            route = "bundle"
        return {"route": route}

    return router_node
