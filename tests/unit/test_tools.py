"""Unit tests for the LangChain agent tools over an OKF bundle."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from okf_agents.bundle import OKFBundle
from okf_agents.tools import create_okf_tools

pytestmark = pytest.mark.unit

TOOL_NAMES = ["read_concept", "search_concepts", "list_links", "read_index"]


def write_bundle(root: Path, files: dict[str, str]) -> OKFBundle:
    for relative, content in files.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content, encoding="utf-8")
    return OKFBundle.load(root)


@pytest.fixture(scope="module")
def tools(bundle: OKFBundle) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in create_okf_tools(bundle)}


class TestRegistry:
    def test_returns_four_tools_in_order(self, bundle: OKFBundle) -> None:
        tools = create_okf_tools(bundle)
        assert [tool.name for tool in tools] == TOOL_NAMES
        assert all(isinstance(tool, BaseTool) for tool in tools)

    @pytest.mark.parametrize("bad_bundle", [None, "not a bundle", 42, object()])
    def test_rejects_non_bundle_immediately(self, bad_bundle: object) -> None:
        with pytest.raises(TypeError, match="OKFBundle"):
            create_okf_tools(bad_bundle)  # type: ignore[arg-type]

    def test_descriptions_are_llm_friendly(self, tools: dict[str, BaseTool]) -> None:
        assert "concept ID" in tools["read_concept"].description
        assert "free-text query" in tools["search_concepts"].description
        assert "unresolved" in tools["list_links"].description
        assert "root index" in tools["read_index"].description

    def test_read_concept_schema(self, tools: dict[str, BaseTool]) -> None:
        assert list(tools["read_concept"].args) == ["concept_id"]
        assert tools["read_concept"].args["concept_id"]["type"] == "string"

    def test_search_schema_constrains_top_k(self, tools: dict[str, BaseTool]) -> None:
        args = tools["search_concepts"].args
        assert list(args) == ["query", "top_k"]
        assert args["top_k"]["minimum"] == 1
        assert args["top_k"]["maximum"] == 25
        assert args["top_k"]["default"] == 5

    def test_list_links_schema_constrains_direction(self, tools: dict[str, BaseTool]) -> None:
        args = tools["list_links"].args
        assert list(args) == ["concept_id", "direction"]
        assert args["direction"]["enum"] == ["out", "in", "both"]
        assert args["direction"]["default"] == "both"

    def test_read_index_schema_has_no_arguments(self, tools: dict[str, BaseTool]) -> None:
        assert tools["read_index"].args == {}


class TestReadConcept:
    def test_valid_read_renders_all_sections(self, tools: dict[str, BaseTool]) -> None:
        output = tools["read_concept"].invoke({"concept_id": "concepts/orders"})
        assert isinstance(output, str)
        assert output.startswith("# Orders\n")
        assert "ID: concepts/orders" in output
        assert "Type: table" in output
        assert "Description: Order lifecycle and fulfilment records." in output
        assert "Tags: sales, commerce" in output
        assert "Timestamp: 2026-07-01T09:30:00" in output
        assert "Related (resolved): concepts/customers, concepts/payments" in output
        assert "Related (unresolved): concepts/missing" in output
        assert output.index("ID:") < output.index("\n---\n") < output.index("Each order belongs")

    def test_resource_field_is_rendered(self, tmp_path: Path) -> None:
        bundle = write_bundle(
            tmp_path,
            {"a.md": "---\ntype: note\nresource: https://example.com/a\n---\n\nBody.\n"},
        )
        output = create_okf_tools(bundle)[0].invoke({"concept_id": "a"})
        assert "Resource: https://example.com/a" in output

    def test_optional_metadata_lines_absent_when_missing(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path, {"a.md": "---\ntype: note\n---\n\nBody.\n"})
        output = create_okf_tools(bundle)[0].invoke({"concept_id": "a"})
        assert output.startswith("# a\n")
        for line in ("Description:", "Resource:", "Tags:", "Timestamp:"):
            assert line not in output
        assert "Related (resolved): none" in output
        assert "Related (unresolved): none" in output

    def test_unknown_concept_returns_error_string(self, tools: dict[str, BaseTool]) -> None:
        output = tools["read_concept"].invoke({"concept_id": "concepts/ghost"})
        assert output.startswith("Error:")
        assert "concepts/ghost" in output

    def test_missing_argument_returns_error_string(self, tools: dict[str, BaseTool]) -> None:
        output = tools["read_concept"].invoke({})
        assert output.startswith("Error:")
        assert "concept_id" in output


class TestSearchConcepts:
    def test_matches_are_numbered_with_id_title_and_type(
        self, tools: dict[str, BaseTool]
    ) -> None:
        output = tools["search_concepts"].invoke({"query": "orders"})
        assert output.startswith("1. concepts/orders - Orders (type: table)")
        assert "Order lifecycle and fulfilment records." in output

    def test_no_match_is_reported_explicitly(self, tools: dict[str, BaseTool]) -> None:
        output = tools["search_concepts"].invoke({"query": "zzzquark"})
        assert output == "No concepts matched the query 'zzzquark'."

    def test_top_k_limits_results(self, tools: dict[str, BaseTool]) -> None:
        output = tools["search_concepts"].invoke({"query": "table", "top_k": 1})
        assert output.startswith("1. ")
        assert "2. " not in output

    @pytest.mark.parametrize("top_k", [0, -3, 26])
    def test_out_of_bounds_top_k_returns_error_string(
        self, tools: dict[str, BaseTool], top_k: int
    ) -> None:
        output = tools["search_concepts"].invoke({"query": "orders", "top_k": top_k})
        assert output.startswith("Error:")
        assert "top_k" in output

    def test_title_falls_back_to_concept_id(self, tmp_path: Path) -> None:
        bundle = write_bundle(tmp_path, {"a.md": "---\ntype: note\n---\n\nUnique alpaca.\n"})
        output = create_okf_tools(bundle)[1].invoke({"query": "alpaca"})
        assert output.startswith("1. a - a (type: note)")

    def test_body_snippet_is_bounded_and_heading_free(self, tmp_path: Path) -> None:
        body = "# Heading\n\n" + "word " * 100
        bundle = write_bundle(tmp_path, {"a.md": f"---\ntype: note\n---\n\n{body}\n"})
        output = create_okf_tools(bundle)[1].invoke({"query": "word"})
        snippet = output.splitlines()[1].strip()
        assert snippet.endswith("...")
        assert len(snippet) <= 203
        assert "Heading" not in snippet


class TestListLinks:
    def test_out_direction_lists_targets_and_marks_broken_links(
        self, tools: dict[str, BaseTool]
    ) -> None:
        output = tools["list_links"].invoke(
            {"concept_id": "concepts/orders", "direction": "out"}
        )
        assert output.splitlines()[0] == "Links for concepts/orders (direction: out):"
        assert "-> concepts/customers - Customers" in output
        assert "-> concepts/payments - Payments" in output
        assert "-> concepts/missing - concepts/missing (unresolved)" in output
        assert "<-" not in output

    def test_in_direction_lists_sources(self, tools: dict[str, BaseTool]) -> None:
        output = tools["list_links"].invoke(
            {"concept_id": "concepts/customers", "direction": "in"}
        )
        assert "<- concepts/orders - Orders" in output
        assert "->" not in output

    def test_both_direction_is_default_and_merges(self, tools: dict[str, BaseTool]) -> None:
        output = tools["list_links"].invoke({"concept_id": "concepts/orders"})
        assert "(direction: both)" in output
        assert "-> concepts/customers - Customers" in output
        assert "<- concepts/customers - Customers" in output
        assert "<- guides/getting-started - Getting Started" in output

    def test_repeated_links_are_deduplicated(self, tmp_path: Path) -> None:
        bundle = write_bundle(
            tmp_path,
            {
                "a.md": "---\ntype: note\n---\n\nSee [b](b.md) and again [b](b.md).\n",
                "b.md": "---\ntype: note\n---\n\nLeaf.\n",
            },
        )
        list_links = create_okf_tools(bundle)[2]
        for direction in ("out", "both"):
            output = list_links.invoke({"concept_id": "a", "direction": direction})
            assert output.count("-> b") == 1
        assert list_links.invoke({"concept_id": "b", "direction": "in"}).count("<- a") == 1

    def test_no_links_is_reported_explicitly(self, tools: dict[str, BaseTool]) -> None:
        output = tools["list_links"].invoke(
            {"concept_id": "concepts/payments", "direction": "out"}
        )
        assert output == "No links found for concepts/payments (direction: out)."

    def test_unknown_concept_returns_error_string(self, tools: dict[str, BaseTool]) -> None:
        output = tools["list_links"].invoke({"concept_id": "nowhere"})
        assert output.startswith("Error:")
        assert "nowhere" in output

    def test_invalid_direction_returns_error_string(self, tools: dict[str, BaseTool]) -> None:
        output = tools["list_links"].invoke(
            {"concept_id": "concepts/orders", "direction": "sideways"}
        )
        assert output.startswith("Error:")
        assert "direction" in output


class TestReadIndex:
    def test_returns_parsed_root_index_body(self, tools: dict[str, BaseTool]) -> None:
        output = tools["read_index"].invoke({})
        assert "# Sample Store Knowledge" in output
        assert "[Orders](/concepts/orders.md)" in output

    def test_returns_synthesized_index_when_root_index_absent(self, tmp_path: Path) -> None:
        bundle = write_bundle(
            tmp_path,
            {
                "a.md": "---\ntype: note\ntitle: Alpha\n---\n\nBody.\n",
                "b.md": "---\ntype: note\n---\n\nBody.\n",
            },
        )
        output = create_okf_tools(bundle)[3].invoke({})
        assert output.startswith("# Index\n")
        assert "- [Alpha](/a.md)" in output
        assert "- [b](/b.md)" in output
        assert not (tmp_path / "index.md").exists()


class TestNoAbsolutePaths:
    def test_no_tool_output_contains_the_bundle_root(
        self, bundle: OKFBundle, tools: dict[str, BaseTool]
    ) -> None:
        outputs = [
            tools["read_concept"].invoke({"concept_id": "concepts/orders"}),
            tools["search_concepts"].invoke({"query": "orders"}),
            tools["list_links"].invoke({"concept_id": "concepts/orders"}),
            tools["read_index"].invoke({}),
        ]
        for output in outputs:
            assert str(bundle.root) not in output
