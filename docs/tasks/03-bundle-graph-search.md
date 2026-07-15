# Task 03 — Bundle, graph, and lexical search

## Goal

Implement eager, immutable bundle loading; directed-link graph operations; and deterministic lexical search.

## Depends on

Tasks 01–02.

## Owned files

- `langgraph_okf/bundle.py`
- `langgraph_okf/_internal/graph_utils.py`
- `tests/conftest.py`
- `tests/fixtures/sample_bundle/**`
- `tests/unit/test_bundle.py`
- `tests/unit/test_graph_utils.py`

## API

Implement `OKFBundle.load(path)`, `get`, `search`, `links_from`, `backlinks`, `neighbors`, `index`, `all_concepts`, `concept_count`, and `root` as described by the source specification and clarified by shared contracts.

## Work

1. Validate that the supplied path exists, is a readable directory, and remains the resolved bundle root.
2. Recursively discover concept Markdown files while excluding every `index.md` and `log.md`.
3. Parse every concept eagerly, aggregate validation failures, then construct immutable lookup and edge indexes.
4. Parse root `index.md` when present; otherwise synthesize it without writing to disk.
5. Mark each edge `resolved` according to whether its target is a loaded concept.
6. Implement outbound, inbound, and both-direction breadth-first traversal without NetworkX.
7. Implement weighted lexical search exactly as specified in `00-shared-contracts.md`.
8. Return deterministic copies from collection APIs and do not expose mutable internal dictionaries.

## Fixture

Create a conformant sample bundle with root `index.md`, optional `log.md`, and at least `concepts/orders.md`, `concepts/customers.md`, and `concepts/payments.md`. Include a cycle, a leaf, a broken link, root-relative and relative links, standard metadata, and one nested reserved file to prove it is excluded.

## Tests

Cover loading, missing/non-directory paths, optional root index synthesis, aggregate invalid files, exact concept lookup, deterministic listing, weighted and case-insensitive partial search, empty/no-match queries, limit validation, outbound/backlinks, unresolved edges, all traversal directions, zero and multiple hops, cycles, unknown IDs, and immutability-by-copy.

## Acceptance criteria

- No external service, LLM, graph package, or vector store is used.
- Discovery and every public result are deterministic across runs.
- Bundles with hundreds of concepts load eagerly without obviously quadratic link resolution.
- `pytest -m unit tests/unit/test_bundle.py tests/unit/test_graph_utils.py`, Ruff, and mypy pass.

## Out of scope

Watching files, writes, cache invalidation, advanced query syntax, and fuzzy/embedding search.
