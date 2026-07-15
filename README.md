<div align="center">

# okf-agents

**LangGraph and LangChain tools, retriever, and navigator for OKF knowledge bundles.**

[![CI](https://github.com/RonCodes88/okf-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/RonCodes88/okf-agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/okf-agents)](https://pypi.org/project/okf-agents/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Installation](#installation) &nbsp;•&nbsp;
[Quick Start](#quick-start) &nbsp;•&nbsp;
[Examples](#examples) &nbsp;•&nbsp;
[Docs](docs/) &nbsp;•&nbsp;
[Contributing](CONTRIBUTING.md)

</div>

---

`okf-agents` turns an [Open Knowledge Format (OKF)](https://okf.md) bundle
— a directory of linked Markdown concepts with YAML frontmatter — into
typed, composable LangGraph and LangChain building blocks. Load a bundle,
get agent tools, a keyword retriever, a graph-aware semantic retriever,
a query router, and a bounded navigator subgraph — all independently
usable, all offline by default.

## Why okf-agents?

Most RAG pipelines chunk documents into a vector store and throw away the
link structure. `okf-agents` keeps the bundle's link graph as a
first-class citizen:

- **Graph-aware retrieval.** Vector search finds entry points; the link
  graph expands the neighborhood. Search for "orders" and automatically
  pull in "customers" because orders *links to* customers.
- **Deterministic by default.** Search, traversal, and routing are
  offline and dependency-free. A model is only involved where you
  explicitly ask for one.
- **Composable pieces, not a framework.** Use just the bundle loader,
  just the tools, or just the retriever — nothing forces you to wire the
  whole stack.
- **Real budget enforcement.** The navigator's hop, concept, and token
  budgets are hard limits on the graph, not soft guidelines a model can
  blow through.

## Installation

```bash
pip install okf-agents
```

Only `langgraph`, `langchain-core`, `pydantic`, and `pyyaml` are
required. Provider SDKs and vector-store packages are never hard
dependencies — bring the ones you need.

## Quick Start

```python
from pathlib import Path
from okf_agents import OKFBundle

bundle = OKFBundle.load("path/to/my-bundle")
print(bundle.concept_count, "concepts loaded")

for concept in bundle.search("customer", top_k=3):
    print(concept.id, concept.frontmatter.title)
```

Point `OKFBundle.load()` at any directory of Markdown files with `type`
frontmatter. No index file is required — see
[docs/concepts.md](docs/concepts.md).

## Examples

### Agent tools

```python
from okf_agents import create_okf_tools

tools = create_okf_tools(bundle)  # read_concept, search_concepts, list_links, read_index
```

Drop these into any tool-calling agent. All four are deterministic and
require no model.

### Keyword retriever

```python
from okf_agents import OKFRetriever

retriever = OKFRetriever(bundle=bundle, top_k=3)
docs = retriever.invoke("orders")
```

### Router

```python
from okf_agents import create_okf_router

router = create_okf_router(bundle)
router({"query": "Orders"})               # exact title match → "bundle"
router({"query": "how do refunds work?"})  # vague, no vector store → "bundle"
```

Pass `vector_store=` to route vague queries to `"vector"`, or
`classifier=` to let a model choose.

### Navigator subgraph

```python
from okf_agents import create_okf_navigator

navigator = create_okf_navigator(bundle, model, max_hops=2)
result = navigator.invoke({"question": "How do orders relate to customers?"})
print(result["answer"], result["citations"])
```

The navigator reads concepts, follows links breadth-first, and produces
a cited answer — all within hard token/hop budgets. See
[docs/navigator-and-budgets.md](docs/navigator-and-budgets.md).

### Graph-aware semantic retrieval

```python
from okf_agents import OKFGraphRetriever, sync_bundle_to_vector_store

sync_bundle_to_vector_store(bundle, vector_store)

graph_retriever = OKFGraphRetriever(
    bundle=bundle, vector_store=vector_store, top_k=3, expand_hops=1,
)
docs = graph_retriever.invoke("order belongs")
# → concepts/orders + concepts/customers (reached via the link graph)
```

See [docs/vector-stores.md](docs/vector-stores.md) for the full sync
and store-capability contract.

## Architecture

```text
OKF bundle (directory of Markdown)
        │
        ▼
   OKFBundle.load()            deterministic parse + link graph + lexical search
        │
        ├── create_okf_tools()        four LangChain tools (no model required)
        ├── OKFRetriever               keyword BaseRetriever
        ├── sync_bundle_to_vector_store + OKFGraphRetriever
        │                              idempotent sync + graph-aware semantic retrieval
        ├── create_okf_router()        bundle / vector / both classification node
        └── create_okf_navigator()     bounded read → expand → cite subgraph
```

Every piece is independently usable.

## Compatibility

- Python 3.11, 3.12, 3.13
- `langgraph` >= 0.2, `langchain-core` >= 0.3, `pydantic` >= 2.0

Full API reference: [docs/api-reference.md](docs/api-reference.md) &nbsp;|&nbsp;
Optional extras and dev setup: [docs/testing.md](docs/testing.md)

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © Ronald Li

---

<div align="center">

If this project is useful to you, a ⭐ helps others find it.

</div>
