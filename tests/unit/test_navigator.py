"""Unit tests for the bounded OKF navigator subgraph.

All tests use deterministic scripted chat models; no external model or
service is ever called. The scripted model raises if invoked more times
than scripted, so every test also guards against accidental loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from okf_agents.bundle import OKFBundle
from okf_agents.navigator import NavigatorState, create_okf_navigator

pytestmark = pytest.mark.unit

STATE_KEYS = {
    "question",
    "visited",
    "context",
    "tokens_used",
    "hops",
    "answer",
    "citations",
    "traversal_path",
}


class ScriptedChatModel(BaseChatModel):
    """Replays scripted responses and fails on any extra model call."""

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


class BlockContentChatModel(ScriptedChatModel):
    """Wraps each scripted response in a content-block list message."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        text = str(result.generations[0].message.content)
        blocks: list[str | dict[str, Any]] = ["Sure: ", {"type": "text", "text": text}]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=blocks))])


class FailingChatModel(BaseChatModel):
    """Raises on every call to prove model exceptions propagate."""

    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("model unavailable")


def concept_text(title: str, body: str) -> str:
    return f"---\ntype: entity\ntitle: {title}\n---\n\n{body}\n"


def make_bundle(root: Path, files: dict[str, str]) -> OKFBundle:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return OKFBundle.load(root)


@pytest.fixture
def chain_bundle(tmp_path: Path) -> OKFBundle:
    """a -> b -> {c, a}: a three-concept chain with a cycle back to a."""
    return make_bundle(
        tmp_path,
        {
            "concepts/a.md": concept_text("Alpha", "See [b](/concepts/b.md)."),
            "concepts/b.md": concept_text(
                "Beta", "See [c](/concepts/c.md) and [a](/concepts/a.md)."
            ),
            "concepts/c.md": concept_text("Gamma", "No links here."),
        },
    )


@pytest.fixture
def star_bundle(tmp_path: Path) -> OKFBundle:
    """a links to b, c, and d; the leaves have no outbound links."""
    return make_bundle(
        tmp_path,
        {
            "concepts/a.md": concept_text(
                "Alpha",
                "See [b](/concepts/b.md), [c](/concepts/c.md), [d](/concepts/d.md).",
            ),
            "concepts/b.md": concept_text("Beta", "Leaf."),
            "concepts/c.md": concept_text("Gamma", "Leaf."),
            "concepts/d.md": concept_text("Delta", "Leaf."),
        },
    )


def plan_json(*concept_ids: str) -> str:
    return json.dumps({"concept_ids": list(concept_ids)})


def decide_json(sufficient: bool, *concept_ids: str) -> str:
    return json.dumps({"sufficient": sufficient, "next_concept_ids": list(concept_ids)})


def answer_json(answer: str, *citations: str) -> str:
    return json.dumps({"answer": answer, "citations": list(citations)})


class TestConfigValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"max_hops": -1},
            {"max_concepts": -1},
            {"token_budget": 0},
            {"strategy": "eager"},
        ],
    )
    def test_invalid_configuration_raises(
        self, chain_bundle: OKFBundle, overrides: dict[str, Any]
    ) -> None:
        model = ScriptedChatModel(responses=[])
        with pytest.raises(ValueError):
            create_okf_navigator(chain_bundle, model, **overrides)

    @pytest.mark.parametrize("bad_bundle", [None, "not a bundle", 42, object()])
    def test_rejects_non_bundle_immediately(self, bad_bundle: object) -> None:
        model = ScriptedChatModel(responses=[])
        with pytest.raises(TypeError, match="OKFBundle"):
            create_okf_navigator(bad_bundle, model)  # type: ignore[arg-type]

    @pytest.mark.parametrize("state", [{}, {"question": ""}, {"question": "   "}])
    def test_empty_question_raises(
        self, chain_bundle: OKFBundle, state: dict[str, Any]
    ) -> None:
        model = ScriptedChatModel(responses=[])
        navigator = create_okf_navigator(chain_bundle, model)
        with pytest.raises(ValueError, match="question"):
            navigator.invoke(cast(NavigatorState, state))


class TestProgressiveFlow:
    def test_state_initialized_from_question_only(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(True),
                answer_json("Alpha.", "concepts/a"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "What is alpha?"})
        assert set(result) >= STATE_KEYS
        assert result["question"] == "What is alpha?"
        assert result["traversal_path"][0] == "index"
        assert result["visited"] == ["concepts/a"]
        assert result["hops"] == 1
        assert result["tokens_used"] > 0
        assert result["answer"] == "Alpha."
        assert result["citations"] == ["concepts/a"]

    def test_plan_read_decide_generate_flow(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(False, "concepts/b"),
                decide_json(True),
                answer_json("Alpha and beta.", "concepts/b", "concepts/a"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b"]
        assert result["traversal_path"] == ["index", "concepts/a", "concepts/b"]
        assert result["hops"] == 2
        assert result["citations"] == ["concepts/b", "concepts/a"]
        assert len(model.calls) == 4

    def test_early_stop_leaves_links_unread(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(True),
                answer_json("Enough.", "concepts/a"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert len(model.calls) == 3

    def test_decide_prompt_offers_linked_candidates(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(True),
                answer_json("Done."),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        navigator.invoke({"question": "alpha?"})
        decide_prompt = "".join(str(message.content) for message in model.calls[1])
        assert "concepts/b" in decide_prompt

    def test_no_link_candidates_skips_decide(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/c"), answer_json("Gamma.", "concepts/c")]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "gamma?"})
        assert result["visited"] == ["concepts/c"]
        assert len(model.calls) == 2

    def test_max_hops_one_skips_decide(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), answer_json("Alpha.", "concepts/a")]
        )
        navigator = create_okf_navigator(chain_bundle, model, max_hops=1)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert result["hops"] == 1
        assert len(model.calls) == 2


class TestExhaustiveFlow:
    def test_traverses_reachable_links_without_deciding(
        self, chain_bundle: OKFBundle
    ) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), answer_json("All.", "concepts/c")]
        )
        navigator = create_okf_navigator(chain_bundle, model, strategy="exhaustive")
        result = navigator.invoke({"question": "alpha?"})
        assert result["traversal_path"] == [
            "index",
            "concepts/a",
            "concepts/b",
            "concepts/c",
        ]
        assert result["hops"] == 3
        assert len(model.calls) == 2

    def test_breadth_first_distance_ordering(self, tmp_path: Path) -> None:
        bundle = make_bundle(
            tmp_path,
            {
                "concepts/a.md": concept_text(
                    "Alpha", "See [b](/concepts/b.md) and [c](/concepts/c.md)."
                ),
                "concepts/b.md": concept_text("Beta", "See [d](/concepts/d.md)."),
                "concepts/c.md": concept_text("Gamma", "Leaf."),
                "concepts/d.md": concept_text("Delta", "Leaf."),
            },
        )
        model = ScriptedChatModel(responses=[plan_json("concepts/a"), answer_json("All.")])
        navigator = create_okf_navigator(bundle, model, strategy="exhaustive")
        result = navigator.invoke({"question": "alpha?"})
        assert result["traversal_path"] == [
            "index",
            "concepts/a",
            "concepts/b",
            "concepts/c",
            "concepts/d",
        ]

    def test_cycle_terminates(self, tmp_path: Path) -> None:
        bundle = make_bundle(
            tmp_path,
            {
                "concepts/a.md": concept_text("Alpha", "See [b](/concepts/b.md)."),
                "concepts/b.md": concept_text("Beta", "See [a](/concepts/a.md)."),
            },
        )
        model = ScriptedChatModel(responses=[plan_json("concepts/a"), answer_json("Both.")])
        navigator = create_okf_navigator(bundle, model, strategy="exhaustive")
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b"]
        assert len(model.calls) == 2

    def test_max_hops_bounds_expansion(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(responses=[plan_json("concepts/a"), answer_json("Some.")])
        navigator = create_okf_navigator(
            chain_bundle, model, max_hops=2, strategy="exhaustive"
        )
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b"]
        assert result["hops"] == 2

    def test_max_concepts_caps_frontier_mid_round(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(responses=[plan_json("concepts/a"), answer_json("Some.")])
        navigator = create_okf_navigator(
            chain_bundle, model, max_concepts=2, strategy="exhaustive"
        )
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b"]
        assert len(model.calls) == 2


class TestModelOutputFallbacks:
    def test_malformed_plan_falls_back_to_lexical_candidates(
        self, chain_bundle: OKFBundle
    ) -> None:
        model = ScriptedChatModel(
            responses=["not json at all", decide_json(True), answer_json("Alpha.")]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha"})
        assert result["visited"][0] == "concepts/a"

    def test_unknown_plan_ids_are_dropped(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/ghost", "concepts/a"),
                decide_json(True),
                answer_json("Alpha."),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]

    def test_all_unknown_plan_ids_fall_back_to_lexical(
        self, chain_bundle: OKFBundle
    ) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/ghost", "nope"),
                decide_json(True),
                answer_json("Alpha."),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha"})
        assert result["visited"][0] == "concepts/a"

    def test_malformed_decide_terminates_to_generate(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), "garbage", answer_json("Alpha.")]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert result["answer"] == "Alpha."
        assert len(model.calls) == 3

    def test_invalid_decide_picks_fall_back_to_link_candidates(
        self, chain_bundle: OKFBundle
    ) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(False, "concepts/ghost"),
                decide_json(True),
                answer_json("Both."),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b"]
        assert len(model.calls) == 4

    def test_decide_picks_capped_at_two_per_round(self, star_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(False, "concepts/b", "concepts/c", "concepts/d"),
                decide_json(True),
                answer_json("Enough."),
            ]
        )
        navigator = create_okf_navigator(star_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a", "concepts/b", "concepts/c"]
        assert len(model.calls) == 4

    def test_model_exception_propagates(self, chain_bundle: OKFBundle) -> None:
        navigator = create_okf_navigator(chain_bundle, FailingChatModel())
        with pytest.raises(RuntimeError, match="model unavailable"):
            navigator.invoke({"question": "alpha?"})


class TestGeneration:
    def test_invented_citations_are_discarded(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(True),
                answer_json(
                    "Alpha.", "concepts/a", "concepts/ghost", "index", "concepts/a"
                ),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["citations"] == ["concepts/a"]

    def test_malformed_answer_uses_raw_text(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), decide_json(True), "Plain text answer."]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["answer"] == "Plain text answer."
        assert result["citations"] == []

    def test_fenced_json_answer_is_parsed(self, chain_bundle: OKFBundle) -> None:
        fenced = f"```json\n{answer_json('Alpha.', 'concepts/a')}\n```"
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), decide_json(True), fenced]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["answer"] == "Alpha."
        assert result["citations"] == ["concepts/a"]

    def test_json_embedded_in_prose_is_parsed(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                f"Read these: {plan_json('concepts/a')} as requested.",
                decide_json(True),
                f"Here is my answer: {answer_json('Alpha.', 'concepts/a')}",
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert result["answer"] == "Alpha."
        assert result["citations"] == ["concepts/a"]

    def test_bare_json_list_plan_is_accepted(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                json.dumps(["concepts/c"]),
                answer_json("Gamma.", "concepts/c"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "gamma?"})
        assert result["visited"] == ["concepts/c"]

    def test_content_block_responses_are_handled(self, chain_bundle: OKFBundle) -> None:
        model = BlockContentChatModel(
            responses=[
                plan_json("concepts/a"),
                decide_json(True),
                answer_json("Alpha.", "concepts/a"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert result["answer"] == "Alpha."
        assert result["citations"] == ["concepts/a"]

    def test_empty_bundle_still_generates_answer(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="no concept files"):
            bundle = OKFBundle.load(tmp_path)
        model = ScriptedChatModel(responses=[answer_json("Nothing to read.")])
        navigator = create_okf_navigator(bundle, model)
        result = navigator.invoke({"question": "anything?"})
        assert result["answer"] == "Nothing to read."
        assert result["visited"] == []
        assert result["traversal_path"] == ["index"]
        assert len(model.calls) == 1


class TestBudgets:
    def test_max_hops_zero_reads_only_index(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(responses=[answer_json("No reads.")])
        navigator = create_okf_navigator(chain_bundle, model, max_hops=0)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == []
        assert result["traversal_path"] == ["index"]
        assert result["hops"] == 0
        assert result["answer"] == "No reads."
        assert len(model.calls) == 1

    def test_max_concepts_zero_skips_planning(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(responses=[answer_json("No reads.")])
        navigator = create_okf_navigator(chain_bundle, model, max_concepts=0)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == []
        assert len(model.calls) == 1

    def test_max_concepts_caps_planned_reads(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[
                plan_json("concepts/a", "concepts/b"),
                answer_json("Alpha only.", "concepts/a"),
            ]
        )
        navigator = create_okf_navigator(chain_bundle, model, max_concepts=1)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert len(model.calls) == 2

    def test_blocked_read_terminates_before_exceeding_budget(
        self, chain_bundle: OKFBundle
    ) -> None:
        def estimator(text: str) -> int:
            return 500 if text.startswith("## Concept:") else 1

        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), answer_json("Index only.")]
        )
        navigator = create_okf_navigator(
            chain_bundle, model, token_budget=100, token_estimator=estimator
        )
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == []
        assert result["tokens_used"] == 1
        assert result["answer"] == "Index only."
        assert len(model.calls) == 2

    def test_read_exactly_filling_budget_is_allowed(self, chain_bundle: OKFBundle) -> None:
        def estimator(text: str) -> int:
            return 10 if text.startswith("## Concept:") else 1

        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), answer_json("Alpha.", "concepts/a")]
        )
        navigator = create_okf_navigator(
            chain_bundle, model, token_budget=11, token_estimator=estimator
        )
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == ["concepts/a"]
        assert result["tokens_used"] == 11
        assert len(model.calls) == 2

    def test_index_consuming_budget_skips_planning(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(responses=[answer_json("Budget spent.")])
        navigator = create_okf_navigator(chain_bundle, model, token_budget=1)
        result = navigator.invoke({"question": "alpha?"})
        assert result["visited"] == []
        assert result["tokens_used"] >= 1
        assert len(model.calls) == 1

    def test_token_estimator_is_injectable(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/a"), decide_json(True), answer_json("Alpha.")]
        )
        navigator = create_okf_navigator(
            chain_bundle, model, token_estimator=lambda text: 1
        )
        result = navigator.invoke({"question": "alpha?"})
        assert result["tokens_used"] == 2  # index + one concept read


class ParentState(TypedDict):
    question: str
    answer: NotRequired[str | None]
    citations: NotRequired[list[str]]


class TestParentGraphEmbedding:
    def test_navigator_runs_as_parent_node(self, chain_bundle: OKFBundle) -> None:
        model = ScriptedChatModel(
            responses=[plan_json("concepts/c"), answer_json("Gamma.", "concepts/c")]
        )
        navigator = create_okf_navigator(chain_bundle, model)
        parent: StateGraph[ParentState, None, ParentState, ParentState] = StateGraph(
            ParentState
        )
        parent.add_node("navigate", navigator)
        parent.add_edge(START, "navigate")
        parent.add_edge("navigate", END)
        app = parent.compile()
        result = app.invoke({"question": "gamma?"})
        assert result["answer"] == "Gamma."
        assert result["citations"] == ["concepts/c"]


class TestNavigatorStateContract:
    def test_state_keys_match_documentation(self) -> None:
        assert set(NavigatorState.__annotations__) == STATE_KEYS
