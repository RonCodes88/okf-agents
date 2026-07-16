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

Have a directory of Markdown files with YAML frontmatter that link to
each other — a wiki, a knowledge base, internal docs, runbooks? `okf-agents`
parses the whole directory, builds the link graph, and gives you typed
LangGraph and LangChain building blocks: agent tools, a keyword retriever,
a graph-aware semantic retriever, a query router, and a bounded navigator
subgraph. Everything is composable, and everything except the navigator
works offline with no model.

The Markdown files follow the [Open Knowledge Format (OKF)](https://okf.md)
convention — any `.md` file with a `type` field in its YAML frontmatter
qualifies.

## Why graph-aware retrieval?

Most RAG pipelines chunk documents into a vector store and throw away
the link structure. `okf-agents` keeps the link graph as a first-class
citizen:

| | Traditional RAG | okf-agents |
|---|---|---|
| **Input** | Flat document chunks | Linked Markdown files with metadata |
| **Retrieval** | Vector similarity only | Vector search + link-graph expansion |
| **Structure** | Lost after chunking | Preserved — titles, tags, types, links |
| **Multi-hop** | Requires multiple retrievals + prompt engineering | Built-in navigator walks links within hard budgets |
| **Determinism** | Depends on embeddings | Search, traversal, and routing are fully deterministic |
| **Dependencies** | Embedding model required | No model required for tools, search, or routing |

## Installation

```bash
pip install okf-agents
```

Only `langgraph`, `langchain-core`, `pydantic`, and `pyyaml` are
required. Provider SDKs and vector-store packages are never hard
dependencies — bring the ones you need.

## Quick Start

```python
import tempfile
from pathlib import Path
from okf_agents import OKFBundle

tmp = Path(tempfile.mkdtemp()) / "concepts"
tmp.mkdir()
(tmp / "orders.md").write_text(
    "---\ntype: table\ntitle: Orders\ntags: [sales]\n---\n"
    "# Orders\n\nEach order belongs to a [customer](customers.md).\n"
)
(tmp / "customers.md").write_text(
    "---\ntype: table\ntitle: Customers\ntags: [crm]\n---\n"
    "# Customers\n\nCustomer accounts and contact details.\n"
)

bundle = OKFBundle.load(tmp.parent)
print(bundle.concept_count, "concepts loaded")

for concept in bundle.search("customer", top_k=3):
    print(concept.id, concept.frontmatter.title)
```

Point `OKFBundle.load()` at any directory of Markdown files with `type`
frontmatter. No index file is required — see
[docs/concepts.md](docs/concepts.md).

## What your Markdown files look like

Any `.md` file with a `type` field in YAML frontmatter works:

```markdown
---
type: runbook
title: Deploying to Production
tags: [devops, deploy]
---

# Deploying to Production

Before deploying, verify the [staging checklist](staging-checklist.md)
and confirm [monitoring](../monitoring/alerts.md) is green.
```

Links between files (`[text](relative-path.md)`) become edges in the
graph. `okf-agents` resolves them automatically and exposes them
through the tools, retriever, and navigator.

## Examples

### LangGraph agent with knowledge base tools

```python
from okf_agents import create_okf_tools

tools = create_okf_tools(bundle)  # read_concept, search_concepts, list_links, read_index
```

Drop these into any tool-calling agent. All four are deterministic and
require no model.

### Keyword retriever (no vector store)

```python
from okf_agents import OKFRetriever

retriever = OKFRetriever(bundle=bundle, top_k=3)
docs = retriever.invoke("orders")
```

Returns LangChain `Document` objects ranked by title > tags >
description > body.

### Query router

```python
from okf_agents import create_okf_router

router = create_okf_router(bundle)
router({"query": "Orders"})               # exact title match → "bundle"
router({"query": "how do refunds work?"})  # vague, no vector store → "bundle"
```

Pass `vector_store=` to route vague queries to `"vector"`, or
`classifier=` to let a model choose.

### Navigator subgraph (autonomous multi-hop traversal)

```python
import json
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from okf_agents import create_okf_navigator

model = FakeListChatModel(
    responses=[
        json.dumps({"concept_ids": ["concepts/orders"]}),
        json.dumps({"sufficient": True}),
        json.dumps({
            "answer": "Orders belong to customers.",
            "citations": ["concepts/orders"],
        }),
    ]
)
navigator = create_okf_navigator(bundle, model, max_hops=2)
result = navigator.invoke({"question": "How do orders relate to customers?"})
print(result["answer"], result["citations"])
```

The navigator reads concepts, follows links breadth-first, and produces
a cited answer — all within hard token/hop/concept budgets. Swap
`FakeListChatModel` for `ChatAnthropic`, `ChatOpenAI`, or any
`BaseChatModel` in production. See
[docs/navigator-and-budgets.md](docs/navigator-and-budgets.md).

### Graph-aware semantic retrieval

`sync_bundle_to_vector_store` + `OKFGraphRetriever` work with any
LangChain `VectorStore` that supports ID-based lookup:

```text
sync_bundle_to_vector_store(bundle, vector_store)  # idempotent upsert

retriever = OKFGraphRetriever(bundle=bundle, vector_store=vector_store, top_k=3, expand_hops=1)
docs = retriever.invoke("order belongs")
# → concepts/orders + concepts/customers (reached via the link graph)
```

See [docs/vector-stores.md](docs/vector-stores.md) for a full working
example and the store-capability contract.

## Architecture

```text
Markdown directory (your wiki / knowledge base / docs)
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

Every piece is independently usable. Use just the bundle loader, just
the tools, or just the retriever — nothing forces you to wire the whole
stack.

## Use cases

- **Internal wikis** — point at a Confluence export or a docs directory
  and get an instant Q&A agent
- **Runbooks and SOPs** — navigable, citable answers grounded in your
  actual procedures
- **Product knowledge bases** — support agents that follow links between
  related articles instead of returning isolated chunks
- **Research notes** — Obsidian-style vaults with linked concepts get
  graph-aware retrieval out of the box
- **API/SDK documentation** — linked reference docs become searchable,
  traversable agent tools

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

If this project saved you time, a ⭐ helps others find it.

</div>
