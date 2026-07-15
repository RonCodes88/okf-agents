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
from pathlib import Path
from okf_agents import OKFBundle, create_okf_tools

# Point at any directory of Markdown files with `type` frontmatter
bundle = OKFBundle.load("path/to/your/markdown/directory")

# Get 4 ready-to-use LangChain tools — no model required
tools = create_okf_tools(bundle)
# tools: [read_concept, search_concepts, list_links, read_index]

# Pass them straight to any LangGraph or LangChain agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools)
```

That's it. Your agent can now read, search, and traverse your Markdown
knowledge base. No chunking, no embedding, no vector store needed for
the basic flow.

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
from okf_agents import OKFBundle, create_okf_tools

bundle = OKFBundle.load("docs/knowledge-base")
tools = create_okf_tools(bundle)
# Returns: read_concept, search_concepts, list_links, read_index
# All deterministic, all offline — pass to any tool-calling agent.
```

### Keyword retriever (no vector store)

```python
from okf_agents import OKFBundle, OKFRetriever

bundle = OKFBundle.load("docs/knowledge-base")
retriever = OKFRetriever(bundle=bundle, top_k=3)
docs = retriever.invoke("deployment process")
# Returns LangChain Documents ranked by title > tags > description > body
```

### Graph-aware semantic retrieval (vector store + link expansion)

```python
from okf_agents import OKFBundle, OKFGraphRetriever, sync_bundle_to_vector_store

bundle = OKFBundle.load("docs/knowledge-base")
sync_bundle_to_vector_store(bundle, vector_store)  # idempotent upsert

retriever = OKFGraphRetriever(
    bundle=bundle,
    vector_store=vector_store,
    top_k=3,
    expand_hops=1,
)
docs = retriever.invoke("deployment")
# Finds "deploying-to-production" via vector search,
# then expands to "staging-checklist" and "alerts" via the link graph.
```

### Query router

```python
from okf_agents import OKFBundle, create_okf_router

bundle = OKFBundle.load("docs/knowledge-base")
router = create_okf_router(bundle)
router({"query": "Deploying to Production"})     # exact title → "bundle"
router({"query": "how do I roll back a deploy?"}) # vague → "bundle" (or "vector" if a vector store is configured)
```

### Navigator subgraph (autonomous multi-hop traversal)

```python
from okf_agents import OKFBundle, create_okf_navigator

bundle = OKFBundle.load("docs/knowledge-base")
navigator = create_okf_navigator(bundle, model, max_hops=3, max_concepts=10)
result = navigator.invoke({"question": "What's the full deploy process?"})

print(result["answer"])          # grounded answer from bundle content
print(result["citations"])       # ["concepts/deploying-to-production", ...]
print(result["traversal_path"])  # ["index", "concepts/deploying-to-production", ...]
```

The navigator reads concepts, follows links breadth-first, and produces
a cited answer — all within hard token/hop/concept budgets. See
[docs/navigator-and-budgets.md](docs/navigator-and-budgets.md).

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
