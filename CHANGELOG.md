# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- README's `OKFGraphRetriever` example used positional arguments, which
  raises `TypeError` because `OKFGraphRetriever` is a Pydantic model and
  only accepts keyword arguments. Corrected to keyword arguments.
- Removed a dead link to `docs/tasks/00-shared-contracts.md` in
  `docs/concepts.md` and `CONTRIBUTING.md` — that file was deleted from
  version control before the `0.1.0` release; its content already lives
  inline in `docs/concepts.md`.

### Added

- A zero-extra-dependency vector-store example in
  `docs/vector-stores.md`, using a small pure-Python `VectorStore` +
  `Embeddings` pair (no `numpy`, no `chromadb`) as an alternative to the
  Chroma-based example.
- A few PyPI keywords (`obsidian`, `notion`, `wiki`, `runbooks`)
  reflecting the library's actual target use cases.

## [0.1.2] - 2026-07-15

### Added

- Dynamic package versioning: the version is now read from
  `okf_agents.__version__` via `[tool.hatch.version]` instead of being
  hardcoded in `pyproject.toml`, so a single source of truth drives both
  the package metadata and the release automation.

### Changed

- Rewrote `.github/workflows/release.yml` to trigger automatically when
  `okf_agents/__init__.py` changes on `main`, detect whether that version
  has already been tagged, build and publish to PyPI via trusted
  publishing, and push the matching `vX.Y.Z` tag itself — rather than
  relying on a maintainer to push a version tag by hand against a
  hardcoded `pyproject.toml` version.
- Expanded the README quick-start and navigator examples to be fully
  self-contained and runnable (temp-directory bundle setup, a working
  `FakeListChatModel` navigator example) instead of referencing
  placeholder variables.
- Loosened `test_version` in `tests/unit/test_exceptions.py` to assert the
  version string is well-formed (`X.Y.Z`, digits) instead of pinning to a
  literal `"0.1.0"`, so the test doesn't need updating on every release.

### Fixed

- The version mismatch that had prevented `0.1.1` from being published to
  PyPI (see `[0.1.1]` below): `pyproject.toml`'s hardcoded version had not
  been bumped alongside the `v0.1.1` tag, so the old tag-triggered
  workflow could never build a `0.1.1` wheel.

## [0.1.1] - 2026-07-15

Tagged, but **never actually published to PyPI** — `pyproject.toml` still
hardcoded `version = "0.1.0"` at this tag (dynamic versioning didn't land
until `0.1.2`), so the release workflow of the time had no new version to
build and publish. PyPI's version history reflects this: it lists `0.1.0`
and `0.1.2`, but not `0.1.1`.

### Added

- `AGENTS.md`, a guide for AI coding agents (Cursor, Claude Code, Codex,
  etc.) integrating against this library.

### Changed

- Corrected project URLs in `pyproject.toml` that still pointed at the
  library's old repository name (`RonCodes88/langgraph-okf` →
  `RonCodes88/okf-agents`).
- Trimmed `README.md` and rewrote `llms.txt` for discoverability —
  shorter section summaries, updated doc descriptions.

### Fixed

- Marked the PyPI name-claim checklist item in `docs/launch-checklist.md`
  as complete, reflecting that `okf-agents` had already been claimed via
  the `0.1.0` release.

## [0.1.0] - 2026-07-14

Initial alpha release.

### Added

- Deterministic OKF v0.1 bundle loading, parsing, link-graph construction,
  and weighted lexical search (`OKFBundle`).
- Domain models for concepts, frontmatter, link edges, and the bundle
  index (`Concept`, `ConceptFrontmatter`, `LinkEdge`, `BundleIndex`).
- Four deterministic LangChain agent tools over a loaded bundle
  (`create_okf_tools`): `read_concept`, `search_concepts`, `list_links`,
  `read_index`.
- A keyword `BaseRetriever` backed by bundle search (`OKFRetriever`) and a
  shared concept-to-`Document` conversion helper.
- A query-routing LangGraph node that classifies queries as `bundle`,
  `vector`, or `both`, with an offline heuristic and optional classifier
  fallback (`create_okf_router`).
- A bounded LangGraph navigator subgraph that reads the bundle index,
  plans traversal, expands linked concepts within hard budgets, and
  returns a grounded, citation-validated answer (`create_okf_navigator`).
- Idempotent synchronization of bundles into LangChain vector stores
  (`sync_bundle_to_vector_store`) and a graph-expanding retriever
  (`OKFGraphRetriever`) that expands semantic hits through the link graph.
- Offline integration tests, opt-in provider-integration tests
  (`RUN_INTEGRATION_TESTS=1`), and opt-in end-to-end tests
  (`RUN_E2E_TESTS=1`).

[Unreleased]: https://github.com/RonCodes88/okf-agents/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/RonCodes88/okf-agents/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/RonCodes88/okf-agents/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/RonCodes88/okf-agents/releases/tag/v0.1.0
