# AGENTS.md — okf-agents

This file is for AI coding agents (Cursor, Claude Code, Codex, Copilot,
etc.) that are writing code using the `okf-agents` library. Follow these
instructions exactly.

## What this library does

`okf-agents` connects OKF (Open Knowledge Format) Markdown bundles to
LangGraph/LangChain. An OKF bundle is a directory of `.md` files with
YAML frontmatter. This library parses them, builds a link graph, and
exposes typed LangChain tools, retrievers, a router, and a navigator
subgraph.

## Install

```
pip install okf-agents
```

## Public exports

Everything is importable from the top-level package:

```python
from okf_agents import (
    OKFBundle,
    Concept, ConceptFrontmatter, LinkEdge, BundleIndex, SyncResult,
    create_okf_tools,
    OKFRetriever, OKFGraphRetriever,
    create_okf_router,
    create_okf_navigator,
    sync_bundle_to_vector_store,
    OKFError, BundleNotFoundError, BundleValidationError,
    ConceptNotFoundError, LinkResolutionError,  # LinkResolutionError is deprecated, never raised
)
```

Do NOT import from submodules like `okf_agents.bundle` or
`okf_agents.models`. Always import from `okf_agents`.

`create_okf_tools`, `create_okf_router`, `create_okf_navigator`,
`OKFRetriever`, and `OKFGraphRetriever` all validate their `bundle`
argument eagerly and raise immediately (`TypeError`, or
`pydantic.ValidationError` for the two retrievers) if it is not an
`OKFBundle` — they never wait until first use to fail.

## Patterns

### Pattern 1: Load a bundle

```python
from okf_agents import OKFBundle

bundle = OKFBundle.load("path/to/bundle")
concept = bundle.get("concepts/orders")
results = bundle.search("customer", top_k=5)
```

`OKFBundle.load()` takes a `str | Path`. It eagerly parses all `.md`
files on init. The bundle is immutable after loading.

### Pattern 2: Create agent tools

```python
from okf_agents import OKFBundle, create_okf_tools

bundle = OKFBundle.load("path/to/bundle")
tools = create_okf_tools(bundle)
# Returns 4 tools: read_concept, search_concepts, list_links, read_index
# Pass `tools` to any LangChain/LangGraph tool-calling agent.
```

No model is required. All tools are deterministic.

### Pattern 3: Keyword retriever

```python
from okf_agents import OKFBundle, OKFRetriever

bundle = OKFBundle.load("path/to/bundle")
retriever = OKFRetriever(bundle=bundle, top_k=5)
docs = retriever.invoke("search query")
```

Returns LangChain `Document` objects. No vector store required.

### Pattern 4: Graph-aware retrieval (requires a vector store)

```python
from okf_agents import OKFBundle, OKFGraphRetriever, sync_bundle_to_vector_store

bundle = OKFBundle.load("path/to/bundle")
sync_bundle_to_vector_store(bundle, vector_store)

retriever = OKFGraphRetriever(
    bundle=bundle,
    vector_store=vector_store,
    top_k=5,
    expand_hops=1,
)
docs = retriever.invoke("search query")
```

`sync_bundle_to_vector_store` is idempotent. `vector_store` must be any
LangChain `VectorStore` with `get_by_ids()` and stable-ID writes.

### Pattern 5: Router

```python
from okf_agents import OKFBundle, create_okf_router

bundle = OKFBundle.load("path/to/bundle")
router = create_okf_router(bundle, vector_store=vs, classifier=model)
result = router({"query": "some question"})
# result["route"] is "bundle", "vector", or "both"
```

`vector_store` and `classifier` are both optional.

### Pattern 6: Navigator subgraph

```python
from okf_agents import OKFBundle, create_okf_navigator

bundle = OKFBundle.load("path/to/bundle")
navigator = create_okf_navigator(bundle, model, max_hops=3, max_concepts=10)
result = navigator.invoke({"question": "How do orders work?"})
# result["answer"]: str
# result["citations"]: list[str]  (concept IDs)
```

`model` must be a LangChain `BaseLanguageModel`. The navigator is a
`CompiledStateGraph` that can be embedded as a subgraph node.

## Common mistakes

- Wrong: `from okf_agents.bundle import OKFBundle`
- Correct: `from okf_agents import OKFBundle`

- Wrong: passing a file path to `OKFBundle.load()`
- Correct: passing a **directory** path to `OKFBundle.load()`

- Wrong: `OKFBundle.load()` then modifying the bundle
- Correct: the bundle is **immutable** after loading

- Wrong: assuming `sync_bundle_to_vector_store` takes an `embeddings` argument
- Correct: the vector store must already have its embeddings configured; the function uses the store's existing embedding function

- Wrong: `navigator.invoke("question")` (passing a string)
- Correct: `navigator.invoke({"question": "..."})` (passing a dict)

- Wrong: treating `create_okf_router` as a retriever
- Correct: the router only classifies queries and sets `route` in state; it does not retrieve documents

## Key types

- `Concept.id` — `str`, relative path without `.md` (e.g. `"concepts/orders"`)
- `Concept.frontmatter` — `ConceptFrontmatter` with `.type`, `.title`, `.tags`, etc.
- `Concept.body` — `str`, Markdown body without frontmatter
- `LinkEdge` — `.source_id`, `.target_id`, `.anchor_text`, `.resolved`
- `NavigatorState` — `TypedDict` with `question`, `answer`, `citations`, `traversal_path`

## Version

Current version: 0.1.0 (pre-1.0 alpha, APIs may change between minor versions).
