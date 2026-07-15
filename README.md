<div align="center">

# okf-agents

**Open Knowledge Format bundles as first-class LangGraph and LangChain building blocks.**

[![CI](https://github.com/RonCodes88/okf-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/RonCodes88/okf-agents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)

[Installation](#installation) &nbsp;•&nbsp;
[Quick Start](#quick-start) &nbsp;•&nbsp;
[Documentation](docs/) &nbsp;•&nbsp;
[Examples](#examples) &nbsp;•&nbsp;
[Contributing](CONTRIBUTING.md) &nbsp;•&nbsp;
[Security](SECURITY.md) &nbsp;•&nbsp;
[Changelog](CHANGELOG.md)

[![PyPI](https://img.shields.io/pypi/v/okf-agents)](https://pypi.org/project/okf-agents/)

</div>

---

`okf-agents` turns an [Open Knowledge Format (OKF)](https://okf.md) bundle
— a directory of linked, frontmattered Markdown concepts — into deterministic
LangGraph and LangChain building blocks: bundle loading and weighted lexical
search, four ready-made agent tools, a keyword retriever, a bounded navigator
subgraph that reads and cites its sources, a query router, and idempotent
vector-store synchronization with graph-aware retrieval. It's for teams
building retrieval or agentic workflows over structured Markdown knowledge
bases who want small, typed, independently testable pieces rather than
another end-to-end RAG framework to learn.

Loading arbitrary Markdown gives you text; a vector store alone gives you
semantically similar chunks with no sense of how concepts relate to each
other. `okf-agents` keeps the bundle's link graph as a first-class
citizen: search results and semantic hits can be expanded through resolved
links (`OKFGraphRetriever`), the navigator subgraph follows links breadth
-first or on model guidance within hard token/hop budgets, and every
answer's citations are validated against concepts it actually read — so
retrieval reflects the bundle's real structure, not just embedding-space
proximity.

This library implements the [OKF specification](https://okf.md/spec/); it
is an integration layer for OKF, not the standard itself.

## Installation

```bash
pip install okf-agents
```

Only `langgraph`, `langchain-core`, `pydantic`, and `pyyaml` are required.
Provider SDKs (`langchain-anthropic`, `langchain-openai`, ...) and
vector-store packages (`chromadb`, ...) are never installed as hard
dependencies — bring the ones you need.

For local development instead, see [Development](#development).

## Quick Start

```python
import tempfile
from pathlib import Path
from okf_agents import OKFBundle

tmp_dir = tempfile.mkdtemp()
concepts_dir = Path(tmp_dir) / "concepts"
concepts_dir.mkdir()
(concepts_dir / "orders.md").write_text(
    "---\ntype: table\ntitle: Orders\ntags: [sales]\n---\n\n"
    "# Orders\n\nEach order belongs to a [customer](customers.md).\n"
)
(concepts_dir / "customers.md").write_text(
    "---\ntype: table\ntitle: Customers\ntags: [crm]\n---\n\n"
    "# Customers\n\nCustomer accounts and contact details.\n"
)

bundle = OKFBundle.load(tmp_dir)
print(bundle.concept_count, "concepts loaded")
print(sorted(concept.id for concept in bundle.all_concepts()))
# 2 concepts loaded
# ['concepts/customers', 'concepts/orders']
```

That's the whole contract: point `OKFBundle.load()` at a directory of
Markdown files with `type` frontmatter, and get back a typed, queryable
bundle. No index file is required — see [docs/concepts.md](docs/concepts.md).

## Why use okf-agents?

- **Deterministic by default.** Search, traversal, and routing heuristics
  are dependency-free and offline; a real model is only involved where you
  explicitly ask for one (the navigator, an optional router classifier).
- **The link graph is not thrown away.** Broken links are tolerated and
  surfaced, not hidden; resolved links drive graph expansion and
  traversal, not just chunk similarity.
- **Small, typed, independently useful pieces.** Use just the bundle
  loader, just the retriever, just the tools — nothing requires the
  navigator or a vector store.
- **Budgets are real limits.** The navigator's hop/concept/token budgets
  are provable bounds on the graph, not soft guidelines a model can blow
  through.

## When should I not use it?

- You need a general-purpose document loader for arbitrary file formats
  (PDFs, HTML, docx) — `okf-agents` only reads OKF-shaped Markdown
  bundles.
- You need semantic chunking, reranking, or multi-vector-store fan-out —
  this library intentionally keeps vector-store integration minimal (see
  [docs/vector-stores.md](docs/vector-stores.md)).
- You want a hosted, batteries-included RAG product — this is a library
  of composable pieces for your own LangGraph app, not an application.

## Examples

Each example below continues from the `bundle` loaded in
[Quick Start](#quick-start).

### LangChain agent tools

```python
from okf_agents import create_okf_tools

tools = create_okf_tools(bundle)
search_tool = next(tool for tool in tools if tool.name == "search_concepts")
print(search_tool.invoke({"query": "customer"}))
```

`create_okf_tools` also returns `read_concept`, `list_links`, and
`read_index` — deterministic, plain-text tools ready to bind to any
tool-calling agent.

### Keyword retriever

```python
from okf_agents import OKFRetriever

retriever = OKFRetriever(bundle=bundle, top_k=3)
for document in retriever.invoke("orders"):
    print(document.metadata["concept_id"], "-", document.metadata["title"])
```

### Router

```python
from okf_agents import create_okf_router

router = create_okf_router(bundle)
print(router({"query": "Orders"}))                 # exact title match -> "bundle"
print(router({"query": "how do refunds work?"}))    # vague, no vector store -> "bundle"
```

Pass `vector_store=` to route vague queries to `"vector"` instead, or
`classifier=` to let a model choose `"bundle"` / `"vector"` / `"both"`.
`create_okf_router` never performs retrieval itself — it only labels the
query for a downstream conditional edge.

### Navigator subgraph

The navigator needs a chat model. This example uses LangChain's
`FakeListChatModel` so it runs fully offline; swap in `ChatAnthropic`,
`ChatOpenAI`, or any other `BaseChatModel` in production.

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
print(result["answer"])
print(result["citations"])
```

See [docs/navigator-and-budgets.md](docs/navigator-and-budgets.md) for the
full traversal and budget contract.

### Vector-store sync and graph-aware retrieval (optional)

`sync_bundle_to_vector_store` and `OKFGraphRetriever` work with any
LangChain `VectorStore` that supports ID-based lookup and stable-ID
writes. This example uses a tiny in-process store and hashed-word
embeddings so it runs offline; swap in Chroma, pgvector, or another real
`VectorStore` with a real embeddings model in production.

```python
import hashlib
import math
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from okf_agents import OKFGraphRetriever, sync_bundle_to_vector_store


class DemoEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str, dimensions: int = 32) -> list[float]:
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.casefold()):
            vector[int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class DemoVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings) -> None:
        self.embedding = embedding
        self._docs: dict[str, Document] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        ids = kwargs["ids"]
        vectors = self.embedding.embed_documents([d.page_content for d in documents])
        for doc_id, document, vector in zip(ids, documents, vectors, strict=True):
            self._docs[doc_id] = document
            self._vectors[doc_id] = vector
        return list(ids)

    def get_by_ids(self, ids: list[str], /) -> list[Document]:
        return [self._docs[i] for i in ids if i in self._docs]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        query_vector = self.embedding.embed_query(query)
        ranked = sorted(
            self._vectors,
            key=lambda i: -sum(x * y for x, y in zip(query_vector, self._vectors[i], strict=True)),
        )
        return [self._docs[i] for i in ranked[:k]]

    @classmethod
    def from_texts(cls, texts: Any, embedding: Any, metadatas: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError


vector_store = DemoVectorStore(DemoEmbeddings())
sync_bundle_to_vector_store(bundle, vector_store)

graph_retriever = OKFGraphRetriever(
    bundle=bundle, vector_store=vector_store, top_k=1, expand_hops=1
)
for document in graph_retriever.invoke("order belongs"):
    print(document.metadata["concept_id"])
# concepts/orders
# concepts/customers   (reached via the "customer" link, not just similarity)
```

See [docs/vector-stores.md](docs/vector-stores.md) for the idempotency and
store-capability contract.

## Architecture overview

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

Every arrow above is independently usable; nothing requires wiring the
whole diagram together.

## Feature status

| Feature                                   | Status |
| ------------------------------------------ | ------ |
| Bundle loading, link graph, lexical search  | Stable |
| LangChain agent tools                       | Stable |
| Keyword retriever                           | Stable |
| Router node                                 | Stable |
| Navigator subgraph                          | Stable |
| Vector-store sync + graph-aware retriever   | Stable |
| Async APIs                                  | Not implemented (v0.1 is sync-only by design) |
| Multi-vector-store adapters                 | Out of scope — any LangChain `VectorStore` with ID lookup + stable-ID writes works |

"Stable" means covered by unit and offline integration tests and used
across the examples above — the package itself is still pre-1.0 alpha, so
APIs may change between minor versions.

## Public API map

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
    ConceptNotFoundError, LinkResolutionError,
)
```

Full signatures and behavior contracts: [docs/api-reference.md](docs/api-reference.md).

## Optional dependencies

| Extra       | Installs                       | Needed for                          |
| ----------- | ------------------------------- | ------------------------------------ |
| `dev`       | test + lint + typecheck tools   | Local development                    |
| `test`      | `pytest`, `pytest-cov`          | Running the test suite               |
| `lint`      | `ruff`                          | Linting                              |
| `typecheck` | `mypy`, `types-PyYAML`         | Strict type checking                 |
| `vector-test` | `chromadb`, `langchain-chroma` | Vector-store integration testing |
| `release`   | `build`, `twine`                | Building/publishing packages         |

None of these are required to use `OKFBundle`, the tools, the retriever,
the router, or the navigator with your own chat model and vector store.

## Compatibility

- Python 3.11, 3.12, 3.13
- `langgraph` >= 0.2
- `langchain-core` >= 0.3
- `pydantic` >= 2.0

## Limitations

- v0.1 parses standard inline Markdown links only — reference-style
  links, images, and autolinks are not graph edges (see
  [docs/concepts.md](docs/concepts.md#links-and-resolution)).
- Lexical search is a weighted substring match, not TF-IDF or embeddings
  — use `OKFGraphRetriever` with a real vector store for semantic recall.
- The navigator does not stream, checkpoint, or support human-in-the-loop
  approval in v0.1.

## Development

```bash
git clone https://github.com/RonCodes88/okf-agents.git
cd okf-agents
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Testing

```bash
pytest              # unit tests + offline integration tests, no secrets required
pytest --cov        # with coverage (>= 85% required)
ruff check .
mypy okf_agents tests
```

Provider-integration and end-to-end tests are opt-in
(`RUN_INTEGRATION_TESTS=1`, `RUN_E2E_TESTS=1`) and skip cleanly without a
provider key. See [docs/testing.md](docs/testing.md).

## Contributing

Contributions are welcome — please read [CONTRIBUTING.md](CONTRIBUTING.md)
for branch/commit conventions and local setup, and open an issue before
starting a large change.

## Security

See [SECURITY.md](SECURITY.md) for how to privately report a
vulnerability.

## License

[MIT](LICENSE) © Ronald Li

## Acknowledgements

Built against the [Open Knowledge Format specification](https://okf.md/spec/)
and the [LangGraph](https://github.com/langchain-ai/langgraph) /
[LangChain](https://github.com/langchain-ai/langchain) ecosystem.

---

<div align="center">

If this project is useful to you, a ⭐ helps others find it.

</div>
