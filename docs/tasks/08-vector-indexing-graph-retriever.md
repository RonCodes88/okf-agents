# Task 08 — Vector indexing and graph retriever

## Goal

Synchronize concepts into compatible LangChain vector stores and expand semantic hits through the OKF link graph.

## Depends on

Tasks 03 and 05.

## Owned files

- `langgraph_okf/indexing.py`
- `langgraph_okf/retriever.py` for `OKFGraphRetriever`
- `tests/unit/test_indexing.py`
- `tests/unit/test_retriever.py` for graph-retriever cases

## Indexing API

Implement:

```python
sync_bundle_to_vector_store(
    bundle: OKFBundle,
    vector_store: VectorStore,
    batch_size: int = 50,
    overwrite: bool = False,
) -> SyncResult
```

The vector store is responsible for embeddings. Use stable IDs derived from a namespace-safe hash of resolved bundle root plus concept ID. Add a deterministic content hash to document metadata.

Feature-detect `get_by_ids` and stable-ID writes before mutating the store. If the store cannot support idempotent synchronization, raise a clear `TypeError` naming the missing capability. Do not claim generic compatibility that the base interface cannot guarantee.

Classify documents as:

- `added`: stable ID absent.
- `skipped`: present with equal content hash and `overwrite=False`.
- `updated`: present but changed, or `overwrite=True`.
- `failed`: an attempted document operation raised; record a sanitized concept-specific error and continue with later batches.

Validate positive `batch_size`. Return zero counts for an empty bundle.

## Graph retriever API

Implement `OKFGraphRetriever(BaseRetriever)` with `bundle`, `vector_store`, `top_k=5`, `expand_hops=1`, and `expand_direction="out"`.

1. Request `top_k` vector hits.
2. Validate `concept_id` metadata and ignore hits outside this bundle.
3. Keep entry hits in vector-store order.
4. Expand each entry hit breadth-first using bundle neighbors.
5. Deduplicate by concept ID; entry hits precede expanded concepts, and expansions follow distance/concept-ID order.
6. Rehydrate every returned `Document` from the bundle so metadata is canonical.

## Tests

Build an in-memory fake vector store implementing only the required capabilities. Cover add/update/skip/overwrite counts, stable IDs, batching, partial failures, unsupported stores, empty bundles, vector-hit ordering, malformed/foreign metadata, all directions, zero/multiple hops, cycles, deduplication, and canonical document metadata.

## Acceptance criteria

- Unit tests require no live store, embedding model, or network.
- Base package imports still work when optional vector-test dependencies are absent.
- Synchronization is demonstrably idempotent in the fake store.
- Relevant unit tests, Ruff, and mypy pass.

## Out of scope

Deleting stale vector documents, chunk-level embeddings, score fusion/reranking, store-specific adapters, and async batching.
