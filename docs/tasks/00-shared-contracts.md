# Shared implementation contracts

These decisions remove ambiguities in `langgraph-okf-spec.md`. They apply to every task.

## Standards baseline

- Target the official [OKF v0.1 specification](https://okf.md/spec/) as accessed July 14, 2026.
- A concept is any UTF-8 `.md` file except a reserved `index.md` or `log.md` at any directory depth.
- Every concept requires parseable YAML frontmatter with a non-empty `type`.
- Supported standard metadata is `type`, `title`, `description`, `resource`, `tags`, and `timestamp`. `timestamp` is an ISO 8601 datetime. Unknown fields are retained in `extra`; `updated` has no special meaning.
- Internal links can be bundle-relative (`/path/file.md`) or file-relative (`../file.md`). External URLs and non-Markdown targets are not graph edges.
- Broken internal links are retained as unresolved edges and never make a bundle invalid.
- A root `index.md` is optional. If absent, the loader synthesizes an in-memory index from sorted concepts. No file is written.

These choices intentionally supersede the draft's `updated: date`, mandatory root index, and root-only reserved-file behavior.

## Core data contracts

- Concept IDs are normalized POSIX paths relative to the bundle root with the final `.md` removed, for example `concepts/orders`.
- Paths exposed in models and document metadata are absolute, resolved filesystem paths serialized as strings.
- `LinkEdge` contains `source_id`, `target_id`, `anchor_text`, and `resolved: bool`.
- `BundleIndex.body` is the original or synthesized Markdown. `concept_ids` contains normalized internal concept targets in first-seen order. `title` is the first H1 when present; `description` is the first non-heading paragraph when present.
- Returned concept and edge collections are deterministic: sort by concept ID unless relevance score or traversal order is part of the API.
- Public collection-returning methods return new lists so callers cannot mutate bundle internals.

## Parsing and validation

- Use `PyYAML` for YAML rather than a hand-written subset parser. Because production code directly imports Pydantic and PyYAML, both must be declared direct dependencies even though this is stricter than the draft's “two required dependencies” goal.
- Decode as UTF-8 and report malformed YAML, invalid standard field types, empty `type`, unreadable files, and paths escaping the bundle as `BundleValidationError` entries.
- Aggregate all invalid concept files during eager loading and raise one `BundleValidationError` with stable, root-relative failed paths.
- Do not add a “strict” mode in v0.1. Missing/empty `type` is always invalid; broken links are always tolerated.
- Parse standard inline Markdown links. Images, reference-style links, autolinks, links inside fenced code, fragments-only links, and external URLs are excluded in v0.1.

## Search and traversal

- Implement dependency-free weighted lexical search, not TF-IDF: case-folded partial-token matching across title (weight 4), tags (3), description (2), and body (1).
- A concept must match at least one query token. Rank by descending score, then concept ID. Invalid `top_k`, `hops`, or enum values raise `ValueError`.
- Breadth-first traversal marks nodes when enqueued, excludes the root, handles cycles, and preserves distance then concept-ID ordering.

## LangChain and LangGraph boundaries

- Sync APIs are primary in v0.1. Do not invent async wrappers.
- Every retriever emits `Document` objects with the metadata schema from the draft plus `resource` and `timestamp` when present.
- A LangChain `VectorStore` is configured with its embedding implementation when constructed. Therefore `sync_bundle_to_vector_store` does not accept a separate `embeddings` argument.
- Model outputs used for routing or navigation must be parsed and validated; malformed output gets deterministic fallback behavior rather than an unbounded retry.
- Token budgets use an injectable estimator, defaulting to `max(1, len(text) // 4)`, so no tokenizer dependency is required.

## Error and quality policy

- Public filesystem and lookup failures use package exceptions. Programmer errors such as negative limits use `ValueError`.
- Tool-facing functions convert expected package exceptions into concise error strings; library APIs raise typed exceptions.
- Unit tests are offline and deterministic. Integration and end-to-end tests are opt-in through environment flags.
- Ruff, strict mypy, and pytest must pass for files owned by a task. Overall branch coverage target is at least 85%.
