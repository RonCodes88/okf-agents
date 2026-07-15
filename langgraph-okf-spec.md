# langgraph-okf — Technical Specification

**Version:** 0.1.0-draft  
**Status:** Spec-Driven Development  
**Author:** TBD  
**Target Python:** 3.11+

---

## 1. Overview

`langgraph-okf` is a Python library that provides first-class integration between [Open Knowledge Format (OKF)](https://okf.md) bundles and LangGraph/LangChain agent workflows. It exposes four layers of integration, from a simple bundle reader to a prebuilt LangGraph navigator subgraph with hybrid retrieval.

### Goals

- Enable any LangGraph agent to consume an OKF bundle with minimal boilerplate
- Exploit OKF's link graph structure for multi-hop, context-aware retrieval
- Provide a drop-in hybrid retriever that routes between direct bundle reads and vector search
- Be fully testable with no LLM dependency in layers 1 and 2
- Ship zero required deps beyond `langgraph` and `langchain-core`

### Non-Goals (v0)

- Writing or mutating concept files
- Bundle publishing or registry integration
- Authentication or access control on bundles
- Async-first API (sync first, async in v0.2)

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User's LangGraph Agent              │
└────────────────────────┬────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐           ┌──────────▼──────────┐
│  Layer 2: Tools │           │  Layer 3: Navigator  │
│  (agent tools)  │           │  (prebuilt subgraph) │
└────────┬────────┘           └──────────┬──────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
              ┌──────────▼──────────┐
              │  Layer 1: Bundle    │
              │  (loader + graph)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Layer 4: Retriever │
              │  (hybrid OKF+vector)│
              └─────────────────────┘
```

---

## 3. Directory Structure

```
langgraph_okf/
├── __init__.py
├── bundle.py          # OKFBundle: loader, link graph, concept access
├── models.py          # Pydantic models: Concept, BundleIndex, LinkEdge
├── retriever.py       # OKFRetriever, OKFGraphRetriever (LangChain BaseRetriever)
├── tools.py           # create_okf_tools() → list[BaseTool]
├── navigator.py       # create_okf_navigator() → CompiledGraph
├── router.py          # create_okf_router() → LangGraph node fn
├── indexing.py        # sync_bundle_to_vector_store()
├── exceptions.py      # BundleNotFoundError, ConceptNotFoundError, etc.
└── _internal/
    ├── parser.py      # Frontmatter + markdown parsing
    └── graph_utils.py # Link graph helpers (BFS, backlinks, neighborhoods)

tests/
├── conftest.py                  # Shared fixtures: sample bundle, mock model
├── fixtures/
│   └── sample_bundle/           # Minimal valid OKF bundle for tests
│       ├── index.md
│       ├── log.md
│       └── concepts/
│           ├── orders.md
│           ├── customers.md
│           └── payments.md
├── unit/
│   ├── test_bundle.py
│   ├── test_parser.py
│   ├── test_models.py
│   ├── test_tools.py
│   ├── test_retriever.py
│   ├── test_router.py
│   └── test_graph_utils.py
├── integration/
│   ├── test_navigator.py        # Requires real LLM (gated by env var)
│   └── test_hybrid_retriever.py # Requires vector store
└── e2e/
    └── test_full_workflow.py    # Full agent + bundle + retrieval
```

---

## 4. Data Models (`models.py`)

All models use Pydantic v2.

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date


class ConceptFrontmatter(BaseModel):
    """Parsed YAML frontmatter from a concept file."""
    type: str                              # required by OKF spec
    title: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    updated: Optional[date] = None
    extra: dict = Field(default_factory=dict)  # any unknown frontmatter fields


class Concept(BaseModel):
    """A single OKF concept file, fully parsed."""
    id: str                   # relative path from bundle root, no .md suffix
                              # e.g. "concepts/orders"
    path: str                 # absolute path on disk
    frontmatter: ConceptFrontmatter
    body: str                 # raw markdown body (excludes frontmatter)
    outbound_links: list[str] = Field(default_factory=list)
                              # concept IDs this concept links to
    raw: str                  # full file contents


class LinkEdge(BaseModel):
    """A directed link between two concepts."""
    source_id: str
    target_id: str
    anchor_text: str          # the [text] part of the markdown link


class BundleIndex(BaseModel):
    """Parsed index.md at bundle root."""
    title: Optional[str] = None
    description: Optional[str] = None
    body: str
    concept_ids: list[str]    # concept IDs listed in the index
```

---

## 5. Bundle Core (`bundle.py`)

### Class: `OKFBundle`

```python
class OKFBundle:
    """
    Loads and provides access to an OKF bundle directory.

    Responsibilities:
    - Discover and parse all concept .md files
    - Build an in-memory directed link graph
    - Provide concept lookup, search, and graph traversal
    """

    @classmethod
    def load(cls, path: str | Path) -> "OKFBundle":
        """
        Load a bundle from a directory path.

        Raises:
            BundleNotFoundError: if path does not exist or has no index.md
            BundleValidationError: if any concept file is missing required `type` field
        """

    def get(self, concept_id: str) -> Concept:
        """
        Retrieve a concept by its ID.

        Args:
            concept_id: relative path from bundle root, without .md suffix

        Raises:
            ConceptNotFoundError: if concept_id is not in the bundle
        """

    def search(self, query: str, top_k: int = 5) -> list[Concept]:
        """
        Full-text search across concept titles, descriptions, tags, and body.
        Uses simple TF-IDF or keyword matching. No embeddings required.

        Returns concepts sorted by relevance score descending.
        """

    def links_from(self, concept_id: str) -> list[LinkEdge]:
        """Return all outbound links from a concept."""

    def backlinks(self, concept_id: str) -> list[LinkEdge]:
        """Return all concepts that link to this concept."""

    def neighbors(
        self,
        concept_id: str,
        hops: int = 1,
        direction: Literal["out", "in", "both"] = "out"
    ) -> list[Concept]:
        """
        Return concepts reachable within `hops` from concept_id.
        BFS traversal, deduped, excludes the root concept itself.
        """

    def index(self) -> BundleIndex:
        """Return the parsed index.md."""

    def all_concepts(self) -> list[Concept]:
        """Return all concepts in the bundle."""

    @property
    def concept_count(self) -> int: ...

    @property
    def root(self) -> Path: ...
```

### Behavior Contracts

- `load()` is eager: parses all files on init. Bundles are expected to be small (hundreds of concepts, not millions).
- Link resolution is best-effort: a link to a non-existent concept is recorded as an edge with an unresolved flag, not a hard error.
- `search()` is case-insensitive, matches partial words, scores by title > tags > description > body.
- The bundle is immutable after loading. No write methods exist in v0.

---

## 6. Agent Tools (`tools.py`)

### `create_okf_tools(bundle: OKFBundle) -> list[BaseTool]`

Returns four LangChain tools ready to be passed to any LangGraph agent.

#### Tool 1: `read_concept`

```
Name: read_concept
Description: Read the full content of a specific OKF concept by its ID.
             Use this when you know the exact concept you need.
Input: { "concept_id": str }
Output: str — formatted markdown with frontmatter metadata header
Raises: ConceptNotFoundError (caught, returned as error string to agent)
```

#### Tool 2: `search_concepts`

```
Name: search_concepts
Description: Full-text search across all concepts in the knowledge bundle.
             Use this when you need to find relevant concepts by topic.
Input: { "query": str, "top_k": int = 5 }
Output: str — numbered list of matching concepts with ID, title, description snippet
```

#### Tool 3: `list_links`

```
Name: list_links
Description: List all concepts linked to or from a given concept.
             Use this to explore the knowledge graph around a concept.
Input: { "concept_id": str, "direction": "out" | "in" | "both" = "out" }
Output: str — list of linked concept IDs with titles and link direction
```

#### Tool 4: `read_index`

```
Name: read_index
Description: Read the bundle's root index to understand what knowledge is available
             and how it is organized. Always start here if you are unsure where to look.
Input: {} (no args)
Output: str — index.md contents
```

### Tool Output Format

All tool outputs are plain strings formatted for LLM consumption. Metadata is embedded as structured text, not JSON, to avoid confusing smaller models.

Example `read_concept` output:

```
# Patient Admission Policy
ID: concepts/admission-policy
Type: policy
Tags: admissions, patient-care
Updated: 2026-06-15
Related: concepts/triage-procedure, concepts/emergency-dept

---

Patients arriving through the Emergency Department must complete...
```

---

## 7. Navigator Subgraph (`navigator.py`)

The navigator is a self-contained LangGraph `CompiledGraph` that autonomously traverses a bundle to answer a question. It can be embedded as a node or subgraph inside any parent graph.

### `create_okf_navigator(bundle, model, **config) -> CompiledGraph`

```python
def create_okf_navigator(
    bundle: OKFBundle,
    model: BaseLanguageModel,
    max_hops: int = 3,
    max_concepts: int = 10,
    token_budget: int = 8000,
    strategy: Literal["progressive", "exhaustive"] = "progressive",
) -> CompiledGraph:
    ...
```

### Navigator State

```python
class NavigatorState(TypedDict):
    question: str
    visited: list[str]          # concept IDs already read
    context: list[str]          # accumulated concept content
    tokens_used: int
    hops: int
    answer: Optional[str]
    citations: list[str]        # concept IDs cited in final answer
    traversal_path: list[str]   # ordered list of concept IDs visited
```

### Navigator Graph

```
START
  │
  ▼
[read_index]          # always starts with index.md
  │
  ▼
[plan]                # LLM: which concepts to read first given the question + index
  │
  ▼
[read_concepts]       # read planned concepts, accumulate context
  │
  ▼
[decide]              # LLM: enough context? → answer | need more? → expand
  │           │
  │       [expand]    # pick 1-2 outbound links from visited concepts, read them
  │           │
  │           └──────► [decide]   (loop, bounded by max_hops + max_concepts)
  │
  ▼
[generate]            # LLM: produce final answer + cite concept IDs
  │
  ▼
END → NavigatorState with answer + citations + traversal_path
```

### Strategy Modes

- `progressive`: stops as soon as the decide node is confident. Optimizes for speed and cost.
- `exhaustive`: reads all reachable concepts within budget. Optimizes for completeness.

### Usage

```python
navigator = create_okf_navigator(bundle, model)

# Standalone invocation
result = navigator.invoke({"question": "What is the patient admission process?"})
print(result["answer"])
print(result["citations"])      # ["concepts/admission-policy", "concepts/triage"]
print(result["traversal_path"]) # ["index", "concepts/admission-policy", ...]

# As a subgraph node inside a parent graph
parent = StateGraph(ParentState)
parent.add_node("navigate_knowledge", navigator)
```

---

## 8. Hybrid Retriever (`retriever.py`)

### Class: `OKFRetriever`

Simple LangChain `BaseRetriever` over a bundle. No vector store required.

```python
class OKFRetriever(BaseRetriever):
    """
    Retrieves LangChain Documents from an OKF bundle via full-text search.
    Each Document's page_content is the concept body.
    Each Document's metadata includes: concept_id, title, type, tags, path.
    """
    bundle: OKFBundle
    top_k: int = 5

    def _get_relevant_documents(self, query: str) -> list[Document]: ...
```

### Class: `OKFGraphRetriever`

Graph-aware retriever. Uses vector search as entry point, then expands via link graph.

```python
class OKFGraphRetriever(BaseRetriever):
    """
    Two-phase retrieval:
    1. Vector search over embedded concepts finds semantically relevant entry points
    2. Link graph expands the neighborhood around each hit

    Requires a vector store populated via sync_bundle_to_vector_store().
    """
    bundle: OKFBundle
    vector_store: VectorStore     # any LangChain VectorStore
    top_k: int = 5
    expand_hops: int = 1
    expand_direction: Literal["out", "in", "both"] = "out"

    def _get_relevant_documents(self, query: str) -> list[Document]:
        # 1. vector search → concept IDs (via metadata)
        # 2. map concept IDs → graph nodes
        # 3. expand neighborhoods
        # 4. dedupe, sort by relevance, return as Documents
        ...
```

### Document Metadata Schema

Every `Document` returned by either retriever must include:

```python
{
    "concept_id": str,
    "title": str,
    "type": str,
    "tags": list[str],
    "path": str,
    "source": "okf_bundle",
    "bundle_root": str,
}
```

---

## 9. Router Node (`router.py`)

### `create_okf_router(bundle, vector_store=None) -> Callable`

Returns a LangGraph node function that classifies an incoming query and routes it.

```python
def create_okf_router(
    bundle: OKFBundle,
    vector_store: Optional[VectorStore] = None,
    classifier: Optional[BaseLanguageModel] = None,
) -> Callable[[RouterState], RouterState]:
    ...
```

### RouterState

```python
class RouterState(TypedDict):
    query: str
    route: Optional[Literal["bundle", "vector", "both"]]
    retriever_result: Optional[list[Document]]
```

### Routing Logic

- If `classifier` is provided: LLM classifies "known fact" vs "fuzzy/semantic" query and sets `route`.
- If no classifier: heuristic routing — if query terms match bundle concept titles/tags exactly, route to `bundle`; otherwise route to `vector` (or `bundle` if no vector store).
- The router node sets `route` in state; conditional edges downstream handle actual branching.

```python
# Usage in parent graph
router_node = create_okf_router(bundle, vector_store=qdrant)

graph = StateGraph(RouterState)
graph.add_node("router", router_node)
graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {"bundle": "bundle_node", "vector": "vector_node", "both": "merge_node"}
)
```

---

## 10. Bundle Indexing (`indexing.py`)

### `sync_bundle_to_vector_store(bundle, vector_store, embeddings, **kwargs)`

```python
def sync_bundle_to_vector_store(
    bundle: OKFBundle,
    vector_store: VectorStore,
    embeddings: Embeddings,
    batch_size: int = 50,
    overwrite: bool = False,
) -> SyncResult:
    """
    Embeds all concepts and upserts them into the vector store.
    Each concept becomes one Document. Metadata schema matches retriever output.
    Uses concept_id as the stable document ID for idempotent upserts.

    Returns SyncResult with counts: added, updated, skipped, failed.
    """
```

```python
class SyncResult(BaseModel):
    added: int
    updated: int
    skipped: int
    failed: int
    errors: list[str]
```

---

## 11. Exceptions (`exceptions.py`)

```python
class OKFError(Exception):
    """Base exception for all langgraph-okf errors."""

class BundleNotFoundError(OKFError):
    """Raised when the bundle directory does not exist or has no index.md."""

class BundleValidationError(OKFError):
    """Raised when one or more concept files fail OKF spec validation."""
    def __init__(self, message: str, failed_files: list[str]): ...

class ConceptNotFoundError(OKFError):
    """Raised when a concept_id does not exist in the bundle."""
    def __init__(self, concept_id: str): ...

class LinkResolutionError(OKFError):
    """Raised when a concept links to a non-existent concept ID."""
    # Non-fatal by default; surfaced only in strict mode
```

---

## 12. Test Specification

### Fixtures (`tests/conftest.py`)

```python
@pytest.fixture(scope="session")
def sample_bundle_path() -> Path:
    """Path to tests/fixtures/sample_bundle/"""

@pytest.fixture(scope="session")
def bundle(sample_bundle_path) -> OKFBundle:
    return OKFBundle.load(sample_bundle_path)

@pytest.fixture
def mock_llm() -> FakeListChatModel:
    """LangChain FakeListChatModel for unit tests. No API calls."""
```

### Sample Bundle (`tests/fixtures/sample_bundle/`)

The fixture bundle must contain:
- `index.md` with title, description, and links to at least 3 concepts
- `log.md`
- `concepts/orders.md` — type: table, links to customers and payments
- `concepts/customers.md` — type: table, links back to orders
- `concepts/payments.md` — type: table, no outbound links

This gives us a graph with a cycle (orders → customers → orders) and a leaf node (payments), covering the main traversal edge cases.

---

### Unit Tests: `test_bundle.py`

```
GIVEN a valid bundle directory
WHEN OKFBundle.load() is called
THEN all concepts are parsed without error
AND concept_count matches the number of .md files (excluding index.md and log.md)

GIVEN a directory with no index.md
WHEN OKFBundle.load() is called
THEN BundleNotFoundError is raised

GIVEN a concept file missing the `type` frontmatter field
WHEN OKFBundle.load() is called in strict mode
THEN BundleValidationError is raised listing the failing file

GIVEN a loaded bundle
WHEN bundle.get("concepts/orders") is called
THEN the returned Concept has correct id, frontmatter, body, and outbound_links

GIVEN a loaded bundle
WHEN bundle.get("concepts/nonexistent") is called
THEN ConceptNotFoundError is raised

GIVEN a loaded bundle
WHEN bundle.search("customer") is called
THEN concepts/customers is in the results
AND results are sorted by relevance

GIVEN the orders concept links to customers
WHEN bundle.links_from("concepts/orders") is called
THEN a LinkEdge with target_id="concepts/customers" is returned

WHEN bundle.backlinks("concepts/customers") is called
THEN a LinkEdge with source_id="concepts/orders" is returned

GIVEN orders → customers → orders (cycle)
WHEN bundle.neighbors("concepts/orders", hops=5) is called
THEN the result is finite (cycle guard works)
AND concepts/customers is in the result

WHEN bundle.neighbors("concepts/orders", hops=1, direction="out") is called
THEN only direct outbound neighbors are returned

WHEN bundle.all_concepts() is called
THEN every concept in the bundle is returned exactly once
```

---

### Unit Tests: `test_parser.py`

```
GIVEN a markdown string with YAML frontmatter
WHEN parsed
THEN ConceptFrontmatter fields are populated correctly
AND unknown frontmatter fields land in ConceptFrontmatter.extra

GIVEN a markdown body with [Link Text](../other/concept.md)
WHEN links are extracted
THEN the outbound concept ID resolves to "other/concept"

GIVEN a concept file with no frontmatter
WHEN parsed
THEN BundleValidationError is raised (type field missing)

GIVEN a concept file with valid type but no other frontmatter
WHEN parsed
THEN optional fields default correctly (empty lists, None)
```

---

### Unit Tests: `test_tools.py`

```
GIVEN a bundle and create_okf_tools(bundle)
THEN four tools are returned with names:
     read_concept, search_concepts, list_links, read_index

GIVEN read_concept tool
WHEN invoked with a valid concept_id
THEN output string contains the concept title and body
AND output string contains the concept's tags

WHEN invoked with an invalid concept_id
THEN output is a descriptive error string (not a raised exception)

GIVEN search_concepts tool
WHEN invoked with query="customer"
THEN output lists the customers concept
AND output includes concept IDs in a parseable format

GIVEN list_links tool
WHEN invoked with direction="out" on orders
THEN output lists concepts/customers and concepts/payments

WHEN invoked with direction="in" on customers
THEN output lists concepts/orders

GIVEN read_index tool
WHEN invoked
THEN output contains the index.md title
```

---

### Unit Tests: `test_retriever.py`

```
GIVEN OKFRetriever(bundle, top_k=3)
WHEN get_relevant_documents("customer orders") is called
THEN returns at most 3 Documents
AND each Document has metadata with "concept_id" and "source" == "okf_bundle"
AND each Document's page_content is non-empty

GIVEN OKFRetriever with top_k=1
WHEN queried for "payments"
THEN the top result is concepts/payments

GIVEN OKFGraphRetriever with a mock vector store returning concepts/orders
WHEN get_relevant_documents is called with expand_hops=1
THEN the result includes concepts/orders AND its neighbors (customers, payments)
AND all returned Documents have valid concept_id metadata
```

---

### Unit Tests: `test_router.py`

```
GIVEN create_okf_router(bundle) with no classifier
WHEN invoked with a query matching an exact concept title
THEN state["route"] == "bundle"

WHEN invoked with a vague query with no title match
AND no vector store is configured
THEN state["route"] == "bundle"

WHEN invoked with a vague query
AND a vector store is configured
THEN state["route"] == "vector"

GIVEN create_okf_router with a classifier mock that returns "both"
WHEN invoked with any query
THEN state["route"] == "both"
```

---

### Integration Tests: `test_navigator.py`

Gated by `RUN_INTEGRATION_TESTS=1` env var. Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

```
GIVEN a real LLM and the sample bundle
WHEN navigator.invoke({"question": "What tables are in this bundle?"}) is called
THEN result["answer"] is a non-empty string
AND result["citations"] is a non-empty list of valid concept IDs
AND result["traversal_path"] starts with the index node
AND result["hops"] <= max_hops config value

GIVEN max_hops=0
WHEN navigator is invoked
THEN navigation terminates after reading the index
AND an answer is still produced

GIVEN a question about payments
WHEN navigator is invoked
THEN concepts/payments appears in citations or traversal_path

GIVEN a question with no answer in the bundle
WHEN navigator is invoked
THEN result["answer"] contains a hedged or "not found" response
AND citations is empty or minimal
```

---

### Integration Tests: `test_hybrid_retriever.py`

Gated by `RUN_INTEGRATION_TESTS=1`. Requires a live or in-process vector store (use Chroma in-memory for CI).

```
GIVEN bundle synced to Chroma in-memory
WHEN OKFGraphRetriever.get_relevant_documents("customer") is called
THEN concepts/customers is in results
AND its neighbor concepts/orders is also in results (expand_hops=1)

GIVEN expand_hops=0
WHEN OKFGraphRetriever is called
THEN only vector search results are returned (no graph expansion)

GIVEN sync_bundle_to_vector_store called twice (idempotent test)
WHEN called the second time with overwrite=False
THEN no duplicate documents exist in the store
AND SyncResult.skipped > 0
```

---

### E2E Test: `test_full_workflow.py`

Gated by `RUN_E2E_TESTS=1`.

```
GIVEN a LangGraph agent with create_okf_tools(bundle) as its toolset
AND a real LLM
WHEN the agent is asked "Explain the relationship between orders and customers"
THEN the agent invokes read_concept or search_concepts at least once
AND the final response mentions both concepts
AND no exceptions are raised

GIVEN a parent graph with create_okf_router + navigator as a subgraph
WHEN invoked end-to-end
THEN routing occurs before navigation
AND the final answer is grounded in bundle content
```

---

## 13. Dependencies

### Required

```toml
[tool.poetry.dependencies]
python = "^3.11"
langgraph = ">=0.2"
langchain-core = ">=0.3"
pydantic = "^2.0"
```

### Optional extras

```toml
[tool.poetry.extras]
graph = ["networkx"]          # for advanced graph traversal
vector = ["langchain-community"]   # for OKFGraphRetriever
chroma = ["chromadb"]         # for Chroma vector store in tests
```

### Dev

```toml
[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-cov = "*"
ruff = "*"
mypy = "*"
langchain-community = "*"     # for testing
chromadb = "*"                # in-process vector store for integration tests
```

---

## 14. Project Configuration

### `pyproject.toml` (non-dep sections)

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "UP", "B"]

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: fast, no external dependencies",
    "integration: requires API keys or live services",
    "e2e: full end-to-end, slow",
]
addopts = "-m unit"           # default: only unit tests in CI

[tool.coverage.run]
source = ["langgraph_okf"]
branch = true

[tool.coverage.report]
fail_under = 85
```

### `.github/workflows/ci.yml` (outline)

```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [ruff check, mypy]

  unit-tests:
    runs-on: ubuntu-latest
    steps: [pytest -m unit --cov]

  integration-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    env:
      RUN_INTEGRATION_TESTS: "1"
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    steps: [pytest -m integration]
```

---

## 15. Public API Surface (`__init__.py`)

```python
from langgraph_okf.bundle import OKFBundle
from langgraph_okf.models import Concept, ConceptFrontmatter, LinkEdge, BundleIndex
from langgraph_okf.tools import create_okf_tools
from langgraph_okf.navigator import create_okf_navigator
from langgraph_okf.retriever import OKFRetriever, OKFGraphRetriever
from langgraph_okf.router import create_okf_router
from langgraph_okf.indexing import sync_bundle_to_vector_store
from langgraph_okf.exceptions import (
    OKFError,
    BundleNotFoundError,
    BundleValidationError,
    ConceptNotFoundError,
)

__version__ = "0.1.0"
__all__ = [
    "OKFBundle",
    "Concept",
    "ConceptFrontmatter",
    "LinkEdge",
    "BundleIndex",
    "create_okf_tools",
    "create_okf_navigator",
    "OKFRetriever",
    "OKFGraphRetriever",
    "create_okf_router",
    "sync_bundle_to_vector_store",
    "OKFError",
    "BundleNotFoundError",
    "BundleValidationError",
    "ConceptNotFoundError",
]
```

---

## 16. Implementation Order

Build in this order so each layer is independently shippable and testable.

1. `models.py` + `_internal/parser.py` — no deps, fully unit testable
2. `bundle.py` + `_internal/graph_utils.py` — depends only on layer 1
3. `tests/unit/` for layers 1 and 2 — reach 85% coverage before moving on
4. `tools.py` — depends on bundle, no LLM needed for unit tests
5. `retriever.py` (OKFRetriever only) — no vector store needed
6. `exceptions.py` — can be done alongside any layer
7. `navigator.py` — first layer requiring a real LLM
8. `router.py` — depends on bundle, optional LLM
9. `retriever.py` (OKFGraphRetriever) + `indexing.py` — requires vector store
10. Integration + e2e tests

**Ship to PyPI after step 6.** Claim the name early with a working `OKFBundle` + tools.

---

## 17. Open Questions for v0.2

- Async support (`aget_relevant_documents`, async navigator)
- Bundle composition: mounting multiple bundles as one logical namespace
- Caching layer for concept reads inside long-running agents
- Write support: `bundle.create_concept()`, `bundle.update_concept()`
- Streaming navigator output for real-time UX

---

## 18. GitHub SEO & Discoverability

This section is as important as the code itself. A library nobody finds is a library nobody uses.

### Repository Name

The repo name is your title tag for GitHub search. `langgraph-okf` is correct — it names the framework and the format, keyword-first. Do not rename it to something clever. Keyword-first beats brand-first at this stage.

### About Section

The About and Topics sections are the highest-leverage ranking factors you can directly control — more impactful than the README for GitHub's internal search algorithm. Keep it ~10 words, keyword-rich:

```
LangGraph and LangChain retriever, tools, and navigator subgraph for OKF (Open Knowledge Format) bundles.
```

That hits: `langgraph`, `langchain`, `retriever`, `OKF`, `Open Knowledge Format` — all terms people will search.

### Topics

Pick topics in the middle ground: popular enough that people search for them, specific enough that you can realistically rank. Avoid irrelevant tags like `beta-feature`, `new-version`, or dates — they hurt more than help. Aim for 10-12 topics:

```
langgraph  langchain  okf  open-knowledge-format  rag
retriever  knowledge-graph  llm  ai-agents  python
langchain-integration  knowledge-management
```

### README Structure

GitHub has high domain authority with Google — a well-optimized README can outrank competitor blogs for specific technical keywords faster than a standalone site. The first 3 lines must answer: what it is, who it's for, how to install it. Put a working code snippet above the fold.

Required README elements:
- Badges: PyPI version, Python version, CI status, coverage, license
- One-line install: `pip install langgraph-okf`
- 15-line working code example in the first scroll
- Comparison table: "with langgraph-okf" vs "without" (tables get shared and linked)
- Star CTA at the bottom: `"If this saved you time, a ⭐ helps others find it"`
- Headers must use exact search keywords (`LangGraph OKF integration`, `OKF retriever`, etc.)

### Activity Signals

Developers and search engines both penalize repos that look abandoned. Ship small releases every 2-3 weeks even when the delta is small — a changelog entry keeps the repo looking alive.

### PyPI Keywords

Mirror your GitHub topics in `pyproject.toml` — PyPI has its own search index:

```toml
[tool.poetry]
keywords = [
  "langgraph", "langchain", "okf",
  "open-knowledge-format", "retriever",
  "knowledge-graph", "rag", "llm", "agents"
]
```

### llms.txt

Add an `llms.txt` file to the repo root. This is an emerging standard for making your project legible to AI assistants, which increasingly influence what developers discover. As search shifts toward AI tools, LLMs like ChatGPT rely on Google results — ranking in traditional search also means ranking in AI-assisted discovery.

```
# langgraph-okf

A Python library that integrates OKF (Open Knowledge Format) bundles with LangGraph and LangChain.
Provides a bundle loader, agent tools, a prebuilt navigator subgraph, and a hybrid retriever.

## Install
pip install langgraph-okf

## Docs
See README.md and /docs
```

### Launch Distribution Checklist

Stars drive ranking and stars come from distribution. Execute in this order on launch day:

1. **Claim PyPI name immediately** — publish a v0.1.0 stub before writing the full library
2. **Hacker News Show HN** — `Show HN: LangGraph integration for OKF bundles (link-graph-aware retrieval)`
3. **r/LocalLLaMA and r/LangChain** — post the benchmark showing navigator vs naive RAG
4. **LangChain Discord `#show-and-tell`** — drop the repo link with a one-paragraph summary
5. **Dev.to / Medium blog post** — `"OKF vs RAG: I built a navigator to test which one wins"` — this is your organic Google SEO play and the piece that earns backlinks
6. **X/Twitter** — tag @LangChainAI and use `#OKF #LangGraph #buildinpublic`
