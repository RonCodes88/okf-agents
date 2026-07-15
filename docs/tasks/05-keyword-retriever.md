# Task 05 — Keyword retriever

## Goal

Provide a LangChain `BaseRetriever` backed only by `OKFBundle.search`.

## Depends on

Task 03.

## Owned files

- `langgraph_okf/retriever.py` for shared conversion helpers and `OKFRetriever`
- `tests/unit/test_retriever.py` for `OKFRetriever` cases

Task 08 later extends both files for the graph retriever; it must preserve this task's behavior.

## API

Implement `OKFRetriever(BaseRetriever)` with Pydantic-compatible fields:

- `bundle: OKFBundle`
- `top_k: int = 5`

Implement LangChain's current synchronous protected retrieval hook and rely on the base class public `invoke` API. Do not add deprecated public aliases solely for the tests.

## Document contract

Create one `Document` per concept:

- `page_content` is the Markdown body.
- Required metadata: `concept_id`, title fallback, `type`, `tags`, absolute `path`, `source="okf_bundle"`, and absolute `bundle_root`.
- Include `description`, `resource`, and ISO 8601 `timestamp` only when present.
- Keep the concept-to-document conversion in one internal helper so Task 08 and indexing cannot drift.

## Tests

Cover result limits and ordering, no matches, content, complete/minimal metadata, optional metadata serialization, top result relevance, invalid `top_k`, and public retriever invocation. Tests must not use embeddings.

## Acceptance criteria

- Importing and using `OKFRetriever` requires no vector-store package.
- Returned metadata is serializable using ordinary JSON encoders.
- `pytest -m unit tests/unit/test_retriever.py`, Ruff, and mypy pass.

## Out of scope

Vector search, graph expansion, reranking, chunking a concept into sections, and async retrieval.
