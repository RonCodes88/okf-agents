"""Bounded LangGraph navigator subgraph over an OKF bundle.

:func:`create_okf_navigator` compiles a self-contained LangGraph state
graph that reads the bundle index, asks a model which concepts to read,
expands linked concepts within hard budgets, and generates a grounded
answer with citations validated against the concepts actually read.

Every route through the graph has a provable bound:

- Read rounds (``hops``) never exceed ``max_hops``; each expansion round
  increments ``hops`` whether or not a read succeeded.
- Concepts read never exceed ``max_concepts``.
- A read that would push ``tokens_used`` past ``token_budget`` is never
  performed and terminates traversal.
- Model calls are at most ``1`` (plan) ``+ max(0, max_hops - 1)``
  (progressive decide rounds) ``+ 1`` (generate). Exhaustive mode makes
  no decide calls. No model call is ever retried.

Malformed planning output falls back to deterministic lexical search
candidates; malformed decision output terminates traversal. A model
exception always propagates to the caller.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, NotRequired, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from langgraph_okf.bundle import OKFBundle

__all__ = ["NavigatorState", "create_okf_navigator"]

_STRATEGIES = ("progressive", "exhaustive")
_INDEX_STEP = "index"
# Progressive rounds expand at most two links, per the navigator spec.
_MAX_PICKS_PER_ROUND = 2

_FENCED_BLOCK_RE = re.compile(r"^```[a-zA-Z]*\s*(?P<inner>.*?)\s*```$", re.DOTALL)


class NavigatorState(TypedDict):
    """State carried through one navigator invocation.

    Invoke with only ``question``; every other field is marked
    ``NotRequired`` because it is initialized internally, and all fields
    are present when the graph finishes.

    - ``question``: the user question; must be a non-empty string.
    - ``visited``: concept IDs already read, in read order.
    - ``context``: accumulated text, starting with the index body.
    - ``tokens_used``: estimated tokens across everything in ``context``.
    - ``hops``: completed read rounds, including the initial planned read.
    - ``answer``: the final answer, ``None`` until generated.
    - ``citations``: concept IDs cited by the answer, filtered to
      ``visited`` so invented citations are discarded.
    - ``traversal_path``: ordered steps taken, always starting with
      ``"index"`` followed by each concept ID as it was read.
    """

    question: str
    visited: NotRequired[list[str]]
    context: NotRequired[list[str]]
    tokens_used: NotRequired[int]
    hops: NotRequired[int]
    answer: NotRequired[str | None]
    citations: NotRequired[list[str]]
    traversal_path: NotRequired[list[str]]


class _PlanOutput(BaseModel):
    concept_ids: list[str]


class _DecideOutput(BaseModel):
    sufficient: bool
    next_concept_ids: list[str] = Field(default_factory=list)


class _AnswerOutput(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)


def _default_token_estimator(text: str) -> int:
    return max(1, len(text) // 4)


def _response_text(response: object) -> str:
    """Extract plain text from a chat message or raw string response."""
    if isinstance(response, BaseMessage):
        content = response.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(response)


def _parse_json_payload(text: str) -> Any | None:
    """Best-effort extraction of one JSON value from model output."""
    stripped = text.strip()
    fenced = _FENCED_BLOCK_RE.match(stripped)
    if fenced:
        stripped = fenced.group("inner")
    try:
        return json.loads(stripped)
    except ValueError:
        pass
    for opener, closer in ("{}", "[]"):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(stripped[start : end + 1])
            except ValueError:
                continue
    return None


def _dedupe(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def _joined_context(context: Sequence[str]) -> str:
    return "\n\n".join(context)


def _plan_messages(question: str, index_body: str) -> list[BaseMessage]:
    return [
        SystemMessage(
            "You navigate an OKF knowledge bundle. Given a question and the "
            "bundle index, choose which concepts to read first. Respond with "
            'JSON only: {"concept_ids": ["<concept-id>", ...]} using concept '
            "IDs exactly as they appear in the index."
        ),
        HumanMessage(f"Question: {question}\n\nBundle index:\n{index_body}"),
    ]


def _decide_messages(
    question: str, context: Sequence[str], candidates: Sequence[str]
) -> list[BaseMessage]:
    return [
        SystemMessage(
            "You judge whether the accumulated context answers the question. "
            'Respond with JSON only: {"sufficient": true} when it does, or '
            '{"sufficient": false, "next_concept_ids": ["<concept-id>", ...]} '
            "choosing only from the candidate concept IDs."
        ),
        HumanMessage(
            f"Question: {question}\n\nContext:\n{_joined_context(context)}\n\n"
            f"Candidate concept IDs: {json.dumps(list(candidates))}"
        ),
    ]


def _generate_messages(
    question: str, context: Sequence[str], visited: Sequence[str]
) -> list[BaseMessage]:
    return [
        SystemMessage(
            "Answer the question using only the provided context. Respond "
            'with JSON only: {"answer": "<answer>", "citations": '
            '["<concept-id>", ...]} citing only the listed concept IDs. If '
            "the context is insufficient, say so in the answer."
        ),
        HumanMessage(
            f"Question: {question}\n\nContext:\n{_joined_context(context)}\n\n"
            f"Concept IDs you may cite: {json.dumps(list(visited))}"
        ),
    ]


def create_okf_navigator(
    bundle: OKFBundle,
    model: BaseLanguageModel[Any],
    max_hops: int = 3,
    max_concepts: int = 10,
    token_budget: int = 8000,
    strategy: Literal["progressive", "exhaustive"] = "progressive",
    token_estimator: Callable[[str], int] | None = None,
) -> CompiledStateGraph[NavigatorState, None, NavigatorState, NavigatorState]:
    """Compile a bounded navigator subgraph for ``bundle``.

    The graph reads the original or synthesized index, asks ``model`` for
    initial candidate concept IDs, reads known unvisited concepts within
    budgets, and then either asks once per round whether context is
    sufficient (``progressive``) or traverses reachable links
    breadth-first (``exhaustive``) before generating a final answer.
    ``max_hops=0`` permits the index but no concept reads. Invoke the
    compiled graph with ``{"question": ...}``; it can also be embedded as
    a node in a parent graph whose state shares the ``question`` and
    output keys.

    Args:
        bundle: The loaded OKF bundle to navigate.
        model: Chat or LLM model used for planning, deciding, and
            answering. Its exceptions propagate; calls are never retried.
        max_hops: Maximum read rounds (the planned read counts as one).
        max_concepts: Maximum concepts read across the whole run.
        token_budget: Hard cap on estimated tokens accumulated in
            ``context``; a read that would exceed it never happens.
        strategy: ``"progressive"`` stops when the model says context is
            sufficient; ``"exhaustive"`` expands links breadth-first
            without confidence-based early stopping.
        token_estimator: Optional ``text -> estimated token count``
            callable, defaulting to ``max(1, len(text) // 4)``.

    Raises:
        ValueError: If ``max_hops`` or ``max_concepts`` is negative,
            ``token_budget`` is less than 1, or ``strategy`` is unknown.
            An empty or missing ``question`` raises at invocation time.
    """
    if max_hops < 0:
        raise ValueError(f"max_hops must be non-negative, got {max_hops}")
    if max_concepts < 0:
        raise ValueError(f"max_concepts must be non-negative, got {max_concepts}")
    if token_budget < 1:
        raise ValueError(f"token_budget must be at least 1, got {token_budget}")
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")
    estimate = token_estimator if token_estimator is not None else _default_token_estimator
    known_ids = {concept.id for concept in bundle.all_concepts()}

    def can_expand(hops: int, visited_count: int, tokens_used: int) -> bool:
        return hops < max_hops and visited_count < max_concepts and tokens_used < token_budget

    def link_candidates(traversal_path: Sequence[str], visited: Sequence[str]) -> list[str]:
        """Unvisited resolved-link targets in traversal, then document, order."""
        excluded = set(visited)
        candidates: list[str] = []
        for step in traversal_path:
            if step == _INDEX_STEP:
                continue
            for edge in bundle.links_from(step):
                if (
                    edge.resolved
                    and edge.target_id not in excluded
                    and edge.target_id not in candidates
                ):
                    candidates.append(edge.target_id)
        return candidates

    def lexical_candidates(question: str) -> list[str]:
        return [concept.id for concept in bundle.search(question, top_k=max(1, max_concepts))]

    def read_batch(
        candidate_ids: Sequence[str], visited: Sequence[str], tokens_used: int
    ) -> tuple[list[str], list[str], int, bool]:
        """Read candidates in order within budgets.

        Returns ``(context_entries, read_ids, tokens_added, blocked)``
        where ``blocked`` means a read would have exceeded
        ``token_budget`` and traversal must terminate.
        """
        entries: list[str] = []
        read_ids: list[str] = []
        tokens_added = 0
        for concept_id in candidate_ids:
            if len(visited) + len(read_ids) >= max_concepts:
                break
            concept = bundle.get(concept_id)
            entry = f"## Concept: {concept_id}\n\n{concept.body}"
            cost = estimate(entry)
            if tokens_used + tokens_added + cost > token_budget:
                return entries, read_ids, tokens_added, True
            entries.append(entry)
            read_ids.append(concept_id)
            tokens_added += cost
        return entries, read_ids, tokens_added, False

    def apply_read_round(
        state: NavigatorState, candidate_ids: Sequence[str], loop_node: str
    ) -> Command[str]:
        """Read one round of candidates and route onward."""
        entries, read_ids, tokens_added, blocked = read_batch(
            candidate_ids, state["visited"], state["tokens_used"]
        )
        visited = state["visited"] + read_ids
        tokens_used = state["tokens_used"] + tokens_added
        hops = state["hops"] + 1
        traversal_path = state["traversal_path"] + read_ids
        goto = "generate"
        if (
            not blocked
            and can_expand(hops, len(visited), tokens_used)
            and link_candidates(traversal_path, visited)
        ):
            goto = loop_node
        return Command(
            update={
                "visited": visited,
                "context": state["context"] + entries,
                "tokens_used": tokens_used,
                "hops": hops,
                "traversal_path": traversal_path,
            },
            goto=goto,
        )

    def read_index(state: NavigatorState) -> Command[str]:
        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        index_body = bundle.index().body
        tokens_used = estimate(index_body)
        can_read = (
            max_hops >= 1
            and max_concepts >= 1
            and tokens_used < token_budget
            and bundle.concept_count > 0
        )
        return Command(
            update={
                "visited": [],
                "context": [index_body],
                "tokens_used": tokens_used,
                "hops": 0,
                "answer": None,
                "citations": [],
                "traversal_path": [_INDEX_STEP],
            },
            goto="plan" if can_read else "generate",
        )

    def plan(state: NavigatorState) -> Command[str]:
        response = model.invoke(_plan_messages(state["question"], state["context"][0]))
        payload = _parse_json_payload(_response_text(response))
        if isinstance(payload, list):
            payload = {"concept_ids": payload}
        candidate_ids: list[str] = []
        try:
            parsed = _PlanOutput.model_validate(payload)
        except ValidationError:
            parsed = None
        if parsed is not None:
            candidate_ids = _dedupe(cid for cid in parsed.concept_ids if cid in known_ids)
        if not candidate_ids:
            candidate_ids = lexical_candidates(state["question"])
        loop_node = "decide" if strategy == "progressive" else "expand"
        return apply_read_round(state, candidate_ids, loop_node)

    def decide(state: NavigatorState) -> Command[str]:
        candidates = link_candidates(state["traversal_path"], state["visited"])
        response = model.invoke(
            _decide_messages(state["question"], state["context"], candidates)
        )
        try:
            parsed = _DecideOutput.model_validate(_parse_json_payload(_response_text(response)))
        except ValidationError:
            parsed = None
        if parsed is None or parsed.sufficient:
            return Command(goto="generate")
        picks = _dedupe(cid for cid in parsed.next_concept_ids if cid in candidates)
        picks = picks[:_MAX_PICKS_PER_ROUND]
        if not picks:
            picks = candidates[:_MAX_PICKS_PER_ROUND]
        return apply_read_round(state, picks, "decide")

    def expand(state: NavigatorState) -> Command[str]:
        frontier = link_candidates(state["traversal_path"], state["visited"])
        return apply_read_round(state, frontier, "expand")

    def generate(state: NavigatorState) -> dict[str, Any]:
        response = model.invoke(
            _generate_messages(state["question"], state["context"], state["visited"])
        )
        text = _response_text(response)
        try:
            parsed = _AnswerOutput.model_validate(_parse_json_payload(text))
        except ValidationError:
            parsed = None
        if parsed is None:
            return {"answer": text, "citations": []}
        visited = set(state["visited"])
        citations = _dedupe(cid for cid in parsed.citations if cid in visited)
        return {"answer": parsed.answer, "citations": citations}

    loop_node = "decide" if strategy == "progressive" else "expand"
    builder = StateGraph(NavigatorState)
    builder.add_node("read_index", read_index, destinations=("plan", "generate"))
    builder.add_node("plan", plan, destinations=(loop_node, "generate"))
    builder.add_node(
        loop_node,
        decide if strategy == "progressive" else expand,
        destinations=(loop_node, "generate"),
    )
    builder.add_node("generate", generate)
    builder.add_edge(START, "read_index")
    builder.add_edge("generate", END)
    return builder.compile()
