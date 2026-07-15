# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/RonCodes88/okf-agents/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RonCodes88/okf-agents/releases/tag/v0.1.0
