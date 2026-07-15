# Vector stores, idempotent sync, and graph-aware retrieval

`langgraph-okf` never embeds text itself. It writes plain LangChain
`Document` objects into a `VectorStore` you configure, and that store owns
its embedding implementation. This page covers what a store needs to
support, how synchronization stays idempotent across repeated runs, and
how `OKFGraphRetriever` expands semantic hits through the bundle's link
graph.

## What `sync_bundle_to_vector_store` requires

```python
from langgraph_okf import OKFBundle, sync_bundle_to_vector_store

bundle = OKFBundle.load("./my_bundle")
result = sync_bundle_to_vector_store(bundle, vector_store)
print(result.added, result.updated, result.skipped, result.failed)
```

The `vector_store` argument is a fully constructed LangChain `VectorStore`
— **with its embedding function already configured** — so this function
takes no separate `embeddings` argument. Before writing anything, it
feature-detects two capabilities the base `VectorStore` interface does
not guarantee:

- **`get_by_ids`**: an overridden implementation, so previously written
  documents can be looked up by stable ID.
- **Stable-ID writes**: `add_documents` or `add_texts` must accept an
  `ids=` keyword.

If either is missing, `sync_bundle_to_vector_store` raises `TypeError`
naming the specific missing capability, rather than silently degrading to
non-idempotent behavior or claiming compatibility the store can't back
up. `langchain_core.vectorstores.InMemoryVectorStore` and most
production stores (Chroma, pgvector, etc.) support both.

## How idempotency works

Every concept gets one document with a **stable ID**: a UUIDv5 derived
from the resolved bundle root plus the concept ID, so the same concept in
the same bundle always maps to the same document, and different bundles
never collide. Each document also carries a deterministic
`content_hash` in its metadata — a SHA-256 over its page content and
metadata.

On each sync, every concept is classified by comparing its stable ID and
content hash against what the store already has:

| Outcome   | Condition                                                    |
| --------- | ------------------------------------------------------------- |
| `added`   | Stable ID not present in the store yet.                       |
| `skipped` | Present, with an equal content hash, and `overwrite=False`.    |
| `updated` | Present but changed, or present with `overwrite=True`.         |
| `failed`  | A store operation for that concept raised; sync continues with later batches, and the error is recorded (sanitized, one line, no traceback) in `SyncResult.errors`. |

Running sync twice in a row with `overwrite=False` therefore produces
`added=N, skipped=0` the first time and `added=0, skipped=N` the second —
no duplicate documents are ever created. Writes happen in `batch_size`
groups (default 50) so a large bundle does not require one enormous
store call, and one failing batch never stops later batches from being
attempted.

## `OKFGraphRetriever`: semantic search plus graph expansion

```python
from langgraph_okf import OKFBundle, OKFGraphRetriever

retriever = OKFGraphRetriever(
    bundle=bundle,
    vector_store=vector_store,
    top_k=5,
    expand_hops=1,
    expand_direction="out",
)
documents = retriever.invoke("how are refunds handled?")
```

`OKFGraphRetriever` runs `vector_store.similarity_search(query, k=top_k)`
to get **entry hits**, validates that each hit's `concept_id` metadata
names a concept that actually exists in this bundle (foreign or malformed
hits are dropped), and keeps entry hits in the vector store's own
ranking order. It then expands each entry hit breadth-first over the
bundle's link graph, up to `expand_hops` hops, in the chosen
`expand_direction` (`"out"`, `"in"`, or `"both"`).

Results are deduplicated by concept ID: entry hits always come first, and
expanded concepts follow in distance-then-concept-ID order, so a concept
reachable from two different entry hits only appears once, and cycles in
the link graph terminate normally instead of looping.

Every returned `Document` is **rehydrated from the bundle** — not read
back from whatever the vector store happened to persist — so its
metadata is always canonical and current, even if the store's copy is
stale relative to the bundle on disk. `expand_hops=0` disables expansion
entirely and returns only the entry hits, unchanged.

## Document metadata contract

Both `OKFRetriever` (keyword search) and `OKFGraphRetriever` (semantic
search + expansion) build `Document` objects through the same shared
`concept_to_document()` helper, so retrieval and vector-store indexing
never drift apart. `page_content` is the concept's Markdown body.
Metadata always includes `concept_id`, `title` (falling back to the
concept ID), `type`, `tags`, an absolute `path`, `source="okf_bundle"`,
and an absolute `bundle_root`; `description`, `resource`, and an ISO 8601
`timestamp` are included only when the concept's frontmatter has them.
Every value is plain, JSON-serializable data — no custom objects — so
metadata round-trips cleanly through any store's own serialization.
