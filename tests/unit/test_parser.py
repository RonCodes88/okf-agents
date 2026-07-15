"""Unit tests for the internal OKF concept and index parser."""

from pathlib import Path

import pytest

from okf_agents._internal.parser import (
    extract_internal_links,
    normalize_link_target,
    parse_bundle_index,
    parse_concept,
    parse_frontmatter,
    split_frontmatter,
    synthesize_bundle_index,
)
from okf_agents.exceptions import BundleValidationError
from okf_agents.models import Concept, ConceptFrontmatter

pytestmark = pytest.mark.unit

ROOT = Path("/kb")
SOURCE = "concepts/orders.md"


def parse(raw: str, rel: str = SOURCE) -> Concept:
    return parse_concept(raw, bundle_root=ROOT, file_path=ROOT / rel)


def make_concept(concept_id: str, title: str | None = None) -> Concept:
    return Concept(
        id=concept_id,
        path=f"/kb/{concept_id}.md",
        frontmatter=ConceptFrontmatter(type="note", title=title),
        body="",
        raw="---\ntype: note\n---\n",
    )


class TestSplitFrontmatter:
    def test_empty_frontmatter_yields_empty_mapping(self) -> None:
        mapping, body = split_frontmatter("---\n---\nBody.\n", source=SOURCE)
        assert mapping == {}
        assert body == "Body.\n"

    def test_body_excludes_delimiter_lines(self) -> None:
        raw = "---\ntype: note\n---\nFirst.\n\n---\n\nA thematic break stays.\n"
        _, body = split_frontmatter(raw, source=SOURCE)
        assert body == "First.\n\n---\n\nA thematic break stays.\n"

    @pytest.mark.parametrize(
        "raw",
        ["", "type: note\n", "# Just markdown\n", "--\ntype: note\n---\n"],
        ids=["empty", "no-delimiters", "plain-markdown", "short-dashes"],
    )
    def test_missing_opening_delimiter(self, raw: str) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            split_frontmatter(raw, source=SOURCE)
        assert SOURCE in excinfo.value.failed_files

    def test_missing_closing_delimiter(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            split_frontmatter("---\ntype: note\n", source=SOURCE)
        assert "closing" in excinfo.value.failed_files[SOURCE]

    def test_malformed_yaml(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            split_frontmatter("---\ntype: [unclosed\n---\nBody.\n", source=SOURCE)
        assert "malformed YAML" in excinfo.value.failed_files[SOURCE]

    def test_non_mapping_frontmatter(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            split_frontmatter("---\n- a\n- b\n---\nBody.\n", source=SOURCE)
        assert "mapping" in excinfo.value.failed_files[SOURCE]


class TestParseFrontmatter:
    def test_unknown_keys_go_only_to_extra(self) -> None:
        fm = parse_frontmatter(
            {"type": "note", "title": "T", "owner": "data-team", "priority": 3},
            source=SOURCE,
        )
        assert fm.extra == {"owner": "data-team", "priority": 3}
        assert fm.title == "T"

    def test_missing_type(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            parse_frontmatter({"title": "No type"}, source=SOURCE)
        assert "type" in excinfo.value.failed_files[SOURCE]

    @pytest.mark.parametrize(
        "mapping",
        [
            {"type": ""},
            {"type": "   "},
            {"type": None},
            {"type": 123},
            {"type": "note", "tags": "production"},
            {"type": "note", "tags": 5},
            {"type": "note", "title": ["not", "a", "string"]},
            {"type": "note", "timestamp": "not-a-timestamp"},
        ],
        ids=[
            "empty-type",
            "blank-type",
            "null-type",
            "int-type",
            "scalar-string-tags",
            "scalar-int-tags",
            "list-title",
            "bad-timestamp",
        ],
    )
    def test_invalid_values_raise_with_source(self, mapping: dict[str, object]) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            parse_frontmatter(mapping, source=SOURCE)
        assert SOURCE in excinfo.value.failed_files


class TestParseConcept:
    def test_minimal_frontmatter(self) -> None:
        raw = "---\ntype: note\n---\nBody text.\n"
        concept = parse(raw)
        assert concept.id == "concepts/orders"
        assert concept.path == str(ROOT / SOURCE)
        assert concept.frontmatter.type == "note"
        assert concept.frontmatter.tags == []
        assert concept.frontmatter.timestamp is None
        assert concept.body == "Body text.\n"
        assert concept.raw == raw
        assert concept.outbound_links == []

    def test_all_standard_fields(self) -> None:
        raw = (
            "---\n"
            "type: table\n"
            "title: Orders\n"
            "description: Order fact table\n"
            "resource: warehouse.orders\n"
            "tags:\n"
            "  - sales\n"
            "  - core\n"
            "timestamp: 2026-05-01T10:30:00+02:00\n"
            "---\n"
            "# Orders\n"
        )
        fm = parse(raw).frontmatter
        assert fm.type == "table"
        assert fm.title == "Orders"
        assert fm.description == "Order fact table"
        assert fm.resource == "warehouse.orders"
        assert fm.tags == ["sales", "core"]
        assert fm.timestamp is not None
        assert fm.timestamp.tzinfo is not None
        assert fm.extra == {}

    def test_naive_timestamp_stays_naive(self) -> None:
        fm = parse("---\ntype: note\ntimestamp: 2026-05-01 10:30:00\n---\n").frontmatter
        assert fm.timestamp is not None
        assert fm.timestamp.tzinfo is None

    def test_unknown_fields_do_not_duplicate_standard_keys(self) -> None:
        raw = "---\ntype: note\ntitle: T\nowner: data-team\n---\n"
        fm = parse(raw).frontmatter
        assert fm.extra == {"owner": "data-team"}

    def test_raw_preserved_and_body_excludes_delimiters(self) -> None:
        raw = "---\ntype: note\n---\nLine one.\nLine two.\n"
        concept = parse(raw)
        assert concept.raw == raw
        assert concept.body == "Line one.\nLine two.\n"
        assert "---" not in concept.body

    def test_crlf_input(self) -> None:
        raw = "---\r\ntype: note\r\ntitle: Orders\r\n---\r\nFirst.\r\n[Link](/tables/x.md)\r\n"
        concept = parse(raw)
        assert concept.frontmatter.title == "Orders"
        assert concept.raw == raw
        assert concept.body == "First.\r\n[Link](/tables/x.md)\r\n"
        assert concept.outbound_links == ["tables/x"]

    def test_non_ascii_content(self) -> None:
        raw = (
            "---\ntype: note\ntitle: Café Zürich\ntags: [日本語]\n---\n"
            "Résumé — naïve façade. 中文正文 [链接](../menu.md)\n"
        )
        concept = parse(raw)
        assert concept.frontmatter.title == "Café Zürich"
        assert concept.frontmatter.tags == ["日本語"]
        assert "中文正文" in concept.body
        assert concept.outbound_links == ["menu"]

    def test_duplicate_links_dedupe_first_seen(self) -> None:
        raw = "---\ntype: note\n---\n[First](/b.md) then [Other](/a.md) then [Again](/b.md).\n"
        assert parse(raw).outbound_links == ["b", "a"]

    def test_broken_link_is_retained(self) -> None:
        raw = "---\ntype: note\n---\n[Missing](/nowhere/missing.md)\n"
        assert parse(raw).outbound_links == ["nowhere/missing"]

    def test_file_outside_bundle_root(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            parse_concept(
                "---\ntype: note\n---\n",
                bundle_root=ROOT,
                file_path=Path("/elsewhere/x.md"),
            )
        assert "/elsewhere/x.md" in excinfo.value.failed_files

    def test_invalid_frontmatter_reports_relative_source(self) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            parse("---\ntitle: no type\n---\n")
        assert list(excinfo.value.failed_files) == [SOURCE]


class TestNormalizeLinkTarget:
    @pytest.mark.parametrize(
        ("target", "source_id", "expected"),
        [
            ("/tables/orders.md", "concepts/customers", "tables/orders"),
            ("/orders.md", "deep/nested/leaf", "orders"),
            ("../orders.md", "concepts/customers", "orders"),
            ("../tables/dim.md", "concepts/customers", "tables/dim"),
            ("./payments.md", "concepts/customers", "concepts/payments"),
            ("payments.md", "concepts/customers", "concepts/payments"),
            ("sibling.md", "root-note", "sibling"),
            ("/tables/orders.md#schema", "concepts/customers", "tables/orders"),
            ("orders.md#totals", "concepts/customers", "concepts/orders"),
            ("#local-section", "concepts/customers", None),
            ("https://example.com/page.md", "concepts/customers", None),
            ("http://example.com", "concepts/customers", None),
            ("mailto:team@example.com", "concepts/customers", None),
            ("//cdn.example.com/page.md", "concepts/customers", None),
            ("/img/logo.png", "concepts/customers", None),
            ("../data.csv", "concepts/customers", None),
        ],
        ids=[
            "bundle-relative",
            "bundle-relative-from-deep-source",
            "parent-relative",
            "parent-relative-sibling-dir",
            "dot-relative",
            "bare-relative",
            "bare-relative-from-root",
            "bundle-relative-fragment",
            "relative-fragment",
            "fragment-only",
            "https",
            "http-no-md",
            "mailto",
            "protocol-relative",
            "non-markdown-target",
            "non-markdown-relative",
        ],
    )
    def test_normalization_table(self, target: str, source_id: str, expected: str | None) -> None:
        assert normalize_link_target(target, source_id=source_id, source=SOURCE) == expected

    @pytest.mark.parametrize(
        ("target", "source_id"),
        [
            ("../../secret.md", "concepts/customers"),
            ("../secret.md", "orders"),
            ("/../outside.md", "concepts/customers"),
            ("../../../etc/passwd.md", "concepts/customers"),
        ],
        ids=["two-up", "up-from-root", "root-escape", "deep-escape"],
    )
    def test_escape_attempts_raise(self, target: str, source_id: str) -> None:
        with pytest.raises(BundleValidationError) as excinfo:
            normalize_link_target(target, source_id=source_id, source=SOURCE)
        assert "escapes" in excinfo.value.failed_files[SOURCE]


class TestExtractInternalLinks:
    def test_edges_in_document_order_with_anchor_text(self) -> None:
        body = "See [Orders](/tables/orders.md) and [Customers](./customers.md).\n"
        edges = extract_internal_links(body, source_id="concepts/guide", source="concepts/guide.md")
        assert [(e.source_id, e.target_id, e.anchor_text, e.resolved) for e in edges] == [
            ("concepts/guide", "tables/orders", "Orders", False),
            ("concepts/guide", "concepts/customers", "Customers", False),
        ]

    def test_repeated_links_keep_every_edge(self) -> None:
        body = "[A](/a.md) [A again](/a.md)\n"
        edges = extract_internal_links(body, source_id="src", source="src.md")
        assert [e.target_id for e in edges] == ["a", "a"]
        assert [e.anchor_text for e in edges] == ["A", "A again"]

    def test_images_are_ignored(self) -> None:
        body = "![Diagram](/tables/orders.md) but [Real](/tables/orders.md)\n"
        edges = extract_internal_links(body, source_id="src", source="src.md")
        assert [e.anchor_text for e in edges] == ["Real"]

    def test_fenced_code_links_are_ignored(self) -> None:
        body = (
            "[Before](/a.md)\n"
            "```markdown\n"
            "[Inside backticks](/b.md)\n"
            "```\n"
            "[Between](/c.md)\n"
            "~~~\n"
            "[Inside tildes](/d.md)\n"
            "~~~\n"
            "[After](/e.md)\n"
        )
        edges = extract_internal_links(body, source_id="src", source="src.md")
        assert [e.target_id for e in edges] == ["a", "c", "e"]

    def test_link_with_quoted_title(self) -> None:
        edges = extract_internal_links(
            '[Docs](/a.md "Orders documentation")\n', source_id="src", source="src.md"
        )
        assert [e.target_id for e in edges] == ["a"]

    def test_reference_links_and_autolinks_are_ignored(self) -> None:
        body = "[ref style][1]\n<https://example.com/x.md>\n\n[1]: /a.md\n"
        assert extract_internal_links(body, source_id="src", source="src.md") == []

    def test_external_and_fragment_only_links_are_ignored(self) -> None:
        body = "[Ext](https://example.com/x.md) [Frag](#section)\n"
        assert extract_internal_links(body, source_id="src", source="src.md") == []

    def test_empty_anchor_text(self) -> None:
        edges = extract_internal_links("[](/a.md)\n", source_id="src", source="src.md")
        assert [(e.target_id, e.anchor_text) for e in edges] == [("a", "")]


class TestParseBundleIndex:
    def test_full_index(self) -> None:
        raw = (
            "# Sales KB\n"
            "\n"
            "Curated sales knowledge.\n"
            "\n"
            "- [Orders](/concepts/orders.md)\n"
            "- [Customers](concepts/customers.md)\n"
            "- [Orders again](/concepts/orders.md)\n"
            "- [External](https://example.com)\n"
        )
        index = parse_bundle_index(raw)
        assert index.title == "Sales KB"
        assert index.description == "Curated sales knowledge."
        assert index.body == raw
        assert index.concept_ids == ["concepts/orders", "concepts/customers"]

    def test_index_without_title_or_description(self) -> None:
        index = parse_bundle_index("- [A](/a.md)\n")
        assert index.title is None
        assert index.description is None
        assert index.concept_ids == ["a"]

    def test_description_skips_link_lists(self) -> None:
        raw = "# T\n\n- [A](/a.md)\n\nAll about the bundle.\n"
        index = parse_bundle_index(raw)
        assert index.description == "All about the bundle."

    def test_multiline_description_is_joined(self) -> None:
        raw = "# T\n\nLine one\ncontinues here.\n\nSecond paragraph.\n"
        assert parse_bundle_index(raw).description == "Line one continues here."

    def test_fenced_code_in_index_is_ignored(self) -> None:
        raw = "# T\n\n```\n[Hidden](/a.md)\n# Not a title\n```\n\nReal description.\n"
        index = parse_bundle_index(raw)
        assert index.title == "T"
        assert index.description == "Real description."
        assert index.concept_ids == []


class TestSynthesizeBundleIndex:
    def test_ordering_is_sorted_by_concept_id(self) -> None:
        concepts = [
            make_concept("concepts/orders", "Orders"),
            make_concept("aaa/alpha"),
            make_concept("concepts/customers", "Customers"),
        ]
        index = synthesize_bundle_index(concepts)
        assert index.concept_ids == ["aaa/alpha", "concepts/customers", "concepts/orders"]
        assert index.title == "Index"
        assert index.description is None

    def test_body_links_use_titles_and_round_trip(self) -> None:
        concepts = [make_concept("concepts/orders", "Orders"), make_concept("aaa/alpha")]
        index = synthesize_bundle_index(concepts, title="Sales KB")
        assert "- [Orders](/concepts/orders.md)" in index.body
        assert "- [aaa/alpha](/aaa/alpha.md)" in index.body
        assert index.body.startswith("# Sales KB\n")
        assert parse_bundle_index(index.body).concept_ids == index.concept_ids

    def test_empty_bundle(self) -> None:
        index = synthesize_bundle_index([])
        assert index.concept_ids == []
        assert index.title == "Index"
