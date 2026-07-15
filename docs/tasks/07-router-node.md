# Task 07 — Router node

## Goal

Create a LangGraph-compatible node that classifies a query as `bundle`, `vector`, or `both` without performing retrieval.

## Depends on

Task 03.

## Owned files

- `langgraph_okf/router.py`
- `tests/unit/test_router.py`

## API

Implement:

```python
create_okf_router(
    bundle: OKFBundle,
    vector_store: VectorStore | None = None,
    classifier: BaseLanguageModel | None = None,
) -> Callable[[RouterState], dict[str, Route]]
```

`RouterState` accepts `query`, optional `route`, and optional `retriever_result`. The node must return a state update containing only `route`; it must not erase unrelated parent-state keys or perform retrieval.

## Routing contract

- Without a classifier, normalize query terms and exact-match them against complete concept titles or tags. Match means `bundle`.
- No exact match plus a vector store means `vector`.
- No exact match and no vector store means `bundle`.
- With a classifier, request validated structured output containing one route.
- Coerce `vector` to `bundle` if no vector store exists. `both` is allowed without a vector store only as `bundle`.
- Empty queries raise `ValueError`.
- Malformed classifier output falls back to the heuristic exactly once; model runtime errors propagate.

The router never reads concepts, calls a retriever, or mutates the incoming state object.

## Tests

Cover normalized exact title and tag matching, partial-title non-match, vague query with/without a vector store, all classifier routes, unavailable-vector coercion, malformed output fallback, propagated model failure, empty query, input immutability, and conditional-edge use in a minimal compiled graph.

## Acceptance criteria

- Heuristic routing is deterministic and offline.
- At most one classifier call occurs per node invocation.
- `pytest -m unit tests/unit/test_router.py`, Ruff, and mypy pass.

## Out of scope

Executing branches, merging documents, query rewriting, confidence scores, and model retries.
