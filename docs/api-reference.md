# API reference

Everything below is importable from the top-level `okf_agents` package
(`from okf_agents import ...`) unless noted otherwise. This is a
reference for behavior and contracts; see [concepts.md](concepts.md),
[navigator-and-budgets.md](navigator-and-budgets.md), and
[vector-stores.md](vector-stores.md) for the reasoning behind them.

## `okf_agents.bundle`

### `OKFBundle`

The entry point for everything else in the library. Immutable once
loaded; every collection-returning method returns a new list, so callers
cannot mutate bundle internals.

- `OKFBundle.load(path: str | Path, *, on_error: Literal["raise", "skip"] = "raise") -> OKFBundle` —
  eagerly parse every concept file under `path`. With the default
  `on_error="raise"`, all validation failures aggregate into one
  `BundleValidationError` and nothing loads. With `on_error="skip"`,
  invalid files (including an invalid root `index.md`) are excluded
  instead — the bundle loads from whatever is valid, and excluded paths
  are reported via `.skipped_files`. Raises `BundleNotFoundError` if
  `path` is not a readable directory, and `ValueError` for an unknown
  `on_error`. Emits a `UserWarning` if the loaded bundle ends up with zero
  concepts.
- `.root -> Path` — the resolved bundle root.
- `.concept_count -> int`
- `.skipped_files -> dict[str, str]` — root-relative path -> reason for
  every file excluded by `on_error="skip"`; always empty under the
  default `on_error="raise"`.
- `.get(concept_id: str) -> Concept` — raises `ConceptNotFoundError` if
  absent.
- `.index() -> BundleIndex` — the parsed or synthesized root index.
- `.all_concepts() -> list[Concept]` — every concept, sorted by ID.
- `.search(query: str, top_k: int = 5) -> list[Concept]` — weighted
  lexical search; raises `ValueError` if `top_k < 1`.
- `.links_from(concept_id: str) -> list[LinkEdge]` — outbound edges,
  document order, including unresolved (broken) links.
- `.backlinks(concept_id: str) -> list[LinkEdge]` — inbound edges,
  ordered by source concept ID then document order.
- `.neighbors(concept_id: str, hops: int = 1, direction: Literal["out", "in", "both"] = "out") -> list[Concept]` —
  breadth-first over resolved edges only; raises `ValueError` for a
  negative `hops` or unknown `direction`.

## `okf_agents.models`

Pure Pydantic data contracts; validate structure only.

- `ConceptFrontmatter` — `type: str` (required, non-empty), `title`,
  `description`, `resource`, `tags: list[str]`, `aliases: list[str]`
  (Obsidian's alias convention; participates in wikilink resolution),
  `timestamp: datetime | None`, `extra: dict[str, Any]` for unrecognized
  keys.
- `Concept` — `id`, `path` (absolute), `frontmatter`, `body` (Markdown
  without the frontmatter block), `outbound_links: list[str]`, `raw`
  (exact file contents).
- `LinkEdge` — `source_id`, `target_id`, `anchor_text`, `resolved: bool`,
  `link_kind: Literal["markdown", "wiki"]`, `ambiguous: bool` (set only
  when a wikilink's lookup key matches more than one concept).
- `BundleIndex` — `title`, `description`, `body`, `concept_ids: list[str]`.
- `SyncResult` — `added`, `updated`, `skipped`, `failed` (all `int`),
  `errors: list[str]`.

## `okf_agents.exceptions`

All derive from `OKFError`, so callers can catch one base type.

- `OKFError` — base class.
- `BundleNotFoundError` — bundle root missing, not a directory, or
  unreadable. Carries `.path` and `.reason` (`"missing"` or
  `"not_a_directory"`), each with a distinct message.
- `BundleValidationError` — one or more concept files are invalid.
  Carries `.failed_files: dict[str, str]` (root-relative path -> reason).
- `ConceptNotFoundError` — a concept ID is not in the bundle. Carries
  `.concept_id`.
- `LinkResolutionError` — an internal link could not be resolved on
  demand. Carries `.source_id` and `.target`.

## `okf_agents.tools`

- `create_okf_tools(bundle: OKFBundle) -> list[BaseTool]` — returns
  `read_concept`, `search_concepts`, `list_links`, `read_index`, in that
  order, as LangChain `StructuredTool`s with typed Pydantic input
  schemas. All four are synchronous, deterministic, produce stable plain
  text (never JSON, never an absolute filesystem path), and require no
  model to construct or call directly. Expected failures (unknown
  concept ID, invalid input) return a string starting with `Error:`
  rather than raising.

## `okf_agents.retriever`

- `concept_to_document(concept: Concept, *, bundle_root: Path) -> Document` —
  the single shared conversion from a `Concept` to a LangChain `Document`;
  see [vector-stores.md](vector-stores.md#document-metadata-contract) for
  the metadata contract.
- `OKFRetriever(bundle: OKFBundle, top_k: int = 5)` — a `BaseRetriever`
  over `OKFBundle.search`. Use the inherited `.invoke(query)`. Requires no
  vector-store package.
- `OKFGraphRetriever(bundle: OKFBundle, vector_store: VectorStore, top_k: int = 5, expand_hops: int = 1, expand_direction: Literal["out", "in", "both"] = "out")` —
  a `BaseRetriever` that expands vector-store hits through the bundle's
  link graph; see [vector-stores.md](vector-stores.md).

## `okf_agents.router`

- `Route = Literal["bundle", "vector", "both"]`
- `RouterState` — `TypedDict` with `query: str`, `route: NotRequired[Route | None]`,
  `retriever_result: NotRequired[list[Document] | None]`.
- `create_okf_router(bundle: OKFBundle, vector_store: VectorStore | None = None, classifier: BaseLanguageModel | None = None) -> Callable[[RouterState], dict[str, Route]]` —
  builds a LangGraph node that classifies a query without performing any
  retrieval itself. Without a classifier: an exact, normalized match
  against a concept title or tag routes to `bundle`; anything else routes
  to `vector` if `vector_store` is given, else `bundle`. With a
  classifier: at most one model call per invocation, with a single
  heuristic fallback on malformed output. A route needing an absent
  vector store is coerced to `bundle`. Raises `ValueError` on an empty
  query.

## `okf_agents.navigator`

- `NavigatorState` — `TypedDict` with `question: str` plus
  `NotRequired` fields `visited`, `context`, `tokens_used`, `hops`,
  `answer`, `citations`, `traversal_path` (all initialized internally).
- `create_okf_navigator(bundle: OKFBundle, model: BaseLanguageModel, max_hops: int = 3, max_concepts: int = 10, token_budget: int = 8000, strategy: Literal["progressive", "exhaustive"] = "progressive", token_estimator: Callable[[str], int] | None = None) -> CompiledStateGraph` —
  see [navigator-and-budgets.md](navigator-and-budgets.md) for the full
  behavior contract.

## `okf_agents.indexing`

- `stable_document_id(bundle_root: str | Path, concept_id: str) -> str` —
  deterministic UUIDv5 vector-store ID for one concept.
- `sync_bundle_to_vector_store(bundle: OKFBundle, vector_store: VectorStore, batch_size: int = 50, overwrite: bool = False) -> SyncResult` —
  see [vector-stores.md](vector-stores.md). Raises `TypeError` if the
  store lacks `get_by_ids` or stable-ID writes, `ValueError` if
  `batch_size < 1`.
