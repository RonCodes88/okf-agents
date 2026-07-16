"""Unit tests for okf-agents domain models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from okf_agents.models import (
    BundleIndex,
    Concept,
    ConceptFrontmatter,
    LinkEdge,
    SyncResult,
)

pytestmark = pytest.mark.unit


class TestConceptFrontmatter:
    def test_minimal_only_type(self) -> None:
        fm = ConceptFrontmatter(type="note")
        assert fm.type == "note"
        assert fm.title is None
        assert fm.description is None
        assert fm.resource is None
        assert fm.tags == []
        assert fm.aliases == []
        assert fm.timestamp is None
        assert fm.extra == {}

    def test_all_standard_fields(self) -> None:
        timestamp = datetime(2026, 5, 1, 10, 30, tzinfo=UTC)
        fm = ConceptFrontmatter(
            type="table",
            title="Orders",
            description="Order fact table",
            resource="warehouse.orders",
            tags=["sales", "core"],
            aliases=["Orders Table", "Sales Orders"],
            timestamp=timestamp,
        )
        assert fm.title == "Orders"
        assert fm.description == "Order fact table"
        assert fm.resource == "warehouse.orders"
        assert fm.tags == ["sales", "core"]
        assert fm.aliases == ["Orders Table", "Sales Orders"]
        assert fm.timestamp == timestamp
        assert fm.extra == {}

    def test_unknown_fields_live_in_extra(self) -> None:
        fm = ConceptFrontmatter(type="note", extra={"owner": "data-team", "priority": 3})
        assert fm.extra == {"owner": "data-team", "priority": 3}

    def test_timezone_aware_iso_string(self) -> None:
        fm = ConceptFrontmatter.model_validate(
            {"type": "note", "timestamp": "2026-05-01T10:30:00+02:00"}
        )
        assert fm.timestamp is not None
        assert fm.timestamp.tzinfo is not None
        assert fm.timestamp.astimezone(UTC) == datetime(2026, 5, 1, 8, 30, tzinfo=UTC)

    def test_naive_iso_string_stays_naive(self) -> None:
        fm = ConceptFrontmatter.model_validate({"type": "note", "timestamp": "2026-05-01T10:30:00"})
        assert fm.timestamp == datetime(2026, 5, 1, 10, 30)
        assert fm.timestamp.tzinfo is None

    def test_timestamp_none_is_preserved(self) -> None:
        assert ConceptFrontmatter(type="note", timestamp=None).timestamp is None

    def test_tags_none_normalizes_to_empty_list(self) -> None:
        assert ConceptFrontmatter.model_validate({"type": "note", "tags": None}).tags == []

    def test_aliases_none_normalizes_to_empty_list(self) -> None:
        assert ConceptFrontmatter.model_validate({"type": "note", "aliases": None}).aliases == []

    @pytest.mark.parametrize("bad_type", ["", "   ", "\t\n"])
    def test_empty_type_rejected(self, bad_type: str) -> None:
        with pytest.raises(ValidationError):
            ConceptFrontmatter(type=bad_type)

    def test_missing_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConceptFrontmatter.model_validate({})

    @pytest.mark.parametrize(
        "data",
        [
            {"type": 123},
            {"type": "note", "tags": "production"},
            {"type": "note", "tags": 5},
            {"type": "note", "tags": ["ok", 7]},
            {"type": "note", "title": ["not", "a", "string"]},
            {"type": "note", "description": {"nested": True}},
            {"type": "note", "timestamp": "not-a-timestamp"},
        ],
        ids=[
            "int-type",
            "scalar-string-tags",
            "scalar-int-tags",
            "non-string-tag-item",
            "list-title",
            "mapping-description",
            "bad-timestamp",
        ],
    )
    def test_wrong_field_types_rejected(self, data: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            ConceptFrontmatter.model_validate(data)


class TestConcept:
    def test_construction_and_defaults(self) -> None:
        raw = "---\ntype: note\n---\nBody.\n"
        concept = Concept(
            id="concepts/orders",
            path="/kb/concepts/orders.md",
            frontmatter=ConceptFrontmatter(type="note"),
            body="Body.\n",
            raw=raw,
        )
        assert concept.outbound_links == []
        assert concept.raw == raw
        assert concept.body == "Body.\n"


class TestLinkEdge:
    def test_resolved_defaults_to_false(self) -> None:
        edge = LinkEdge(source_id="a", target_id="b", anchor_text="B")
        assert edge.resolved is False

    def test_resolved_can_be_set(self) -> None:
        edge = LinkEdge(source_id="a", target_id="b", anchor_text="B", resolved=True)
        assert edge.resolved is True

    def test_link_kind_defaults_to_markdown(self) -> None:
        edge = LinkEdge(source_id="a", target_id="b", anchor_text="B")
        assert edge.link_kind == "markdown"

    def test_link_kind_can_be_wiki(self) -> None:
        edge = LinkEdge(source_id="a", target_id="b", anchor_text="B", link_kind="wiki")
        assert edge.link_kind == "wiki"

    def test_invalid_link_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LinkEdge(source_id="a", target_id="b", anchor_text="B", link_kind="notion")  # type: ignore[arg-type]

    def test_ambiguous_defaults_to_false(self) -> None:
        edge = LinkEdge(source_id="a", target_id="b", anchor_text="B")
        assert edge.ambiguous is False

    def test_ambiguous_can_be_set(self) -> None:
        edge = LinkEdge(
            source_id="a", target_id="b", anchor_text="B", link_kind="wiki", ambiguous=True
        )
        assert edge.ambiguous is True


class TestBundleIndex:
    def test_defaults(self) -> None:
        index = BundleIndex(body="# Index\n")
        assert index.title is None
        assert index.description is None
        assert index.concept_ids == []

    def test_full_construction(self) -> None:
        index = BundleIndex(
            title="Sales KB",
            description="Curated sales knowledge.",
            body="# Sales KB\n",
            concept_ids=["concepts/orders"],
        )
        assert index.concept_ids == ["concepts/orders"]


class TestSyncResult:
    def test_defaults_are_zero(self) -> None:
        result = SyncResult()
        assert (result.added, result.updated, result.skipped, result.failed) == (0, 0, 0, 0)
        assert result.errors == []

    def test_counts_and_errors(self) -> None:
        result = SyncResult(added=2, updated=1, skipped=3, failed=1, errors=["boom"])
        assert result.added == 2
        assert result.errors == ["boom"]

    @pytest.mark.parametrize("field", ["added", "updated", "skipped", "failed"])
    def test_negative_counts_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            SyncResult.model_validate({field: -1})
