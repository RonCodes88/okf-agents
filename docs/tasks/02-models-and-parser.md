# Task 02 — Models and parser

## Goal

Define validated domain models and pure parsing functions for OKF v0.1 documents and links.

## Depends on

Task 01.

## Owned files

- `langgraph_okf/models.py`
- `langgraph_okf/_internal/parser.py`
- `tests/unit/test_models.py`
- `tests/unit/test_parser.py`

## Public and internal contracts

Implement Pydantic v2 models:

- `ConceptFrontmatter(type, title, description, resource, tags, timestamp, extra)`
- `Concept(id, path, frontmatter, body, outbound_links, raw)`
- `LinkEdge(source_id, target_id, anchor_text, resolved)`
- `BundleIndex(title, description, body, concept_ids)`
- `SyncResult(added, updated, skipped, failed, errors)`; defined here for reuse by Task 08

Implement internal pure functions for:

- Splitting and validating YAML frontmatter.
- Parsing a concept from raw text plus bundle/file paths.
- Parsing or synthesizing a `BundleIndex`.
- Extracting internal inline Markdown links with anchor text.
- Normalizing a link target to a concept ID while rejecting traversal outside the bundle.

Names of internal functions may vary, but type signatures and docstrings must make their responsibilities explicit.

## Required behavior

- Follow all metadata and link rules in `00-shared-contracts.md`.
- Preserve the exact full source in `raw` and exclude frontmatter delimiters from `body`.
- Store unknown frontmatter keys only in `extra`, without duplicating standard keys.
- Normalize tags to a list of strings; reject scalar tags rather than silently guessing.
- Parse timezone-aware or naive ISO 8601 timestamps as `datetime`; preserve `None`.
- Ignore links in fenced code and image syntax.
- Resolve `/tables/orders.md` from the bundle root and `../orders.md` from the source file's directory.
- Deduplicate `outbound_links` in first-seen order while retaining each concrete `LinkEdge` for graph construction.

## Tests

Cover minimal frontmatter, all standard fields, unknown fields, malformed YAML, missing/empty `type`, wrong field types, CRLF input, non-ASCII text, links with fragments, root-relative and relative links, external links, broken links, fenced-code links, path escape attempts, index parsing, and synthesized index ordering.

## Acceptance criteria

- Parsing functions perform no filesystem writes and make no network calls.
- Invalid input raises `BundleValidationError` with the source path.
- Tests use table-driven parametrization for path/link cases.
- `pytest -m unit tests/unit/test_models.py tests/unit/test_parser.py`, Ruff, and mypy pass.

## Out of scope

Directory discovery, graph traversal, search ranking, and LangChain documents.
