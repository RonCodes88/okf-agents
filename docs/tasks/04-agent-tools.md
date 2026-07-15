# Task 04 — LangChain agent tools

## Goal

Expose a loaded bundle as four deterministic, LLM-friendly LangChain tools.

## Depends on

Task 03.

## Owned files

- `langgraph_okf/tools.py`
- `tests/unit/test_tools.py`

## API

Implement `create_okf_tools(bundle: OKFBundle) -> list[BaseTool]`, returning tools in this order:

1. `read_concept`
2. `search_concepts`
3. `list_links`
4. `read_index`

Use typed Pydantic input schemas so generated tool schemas constrain `top_k` and link direction.

## Output contracts

- `read_concept`: heading, ID, type, optional metadata, resolved and unresolved related IDs, separator, then body.
- `search_concepts`: numbered matches with ID, title fallback, type, and a bounded description/body snippet. Explicitly say when no concept matches.
- `list_links`: direction marker, ID, title fallback, and unresolved status. Support `out`, `in`, and deduplicated `both`.
- `read_index`: return the parsed or synthesized index body.

Formatting must be stable plain text, not JSON. Do not expose absolute filesystem paths in tool output.

## Error behavior

Convert `ConceptNotFoundError` and input-validation failures to concise strings beginning with `Error:`. Unexpected exceptions must still raise so defects are observable.

## Tests

Verify count/order/names/descriptions/schema, valid and invalid reads, every optional metadata field, no-match search, top-k bounds, all link directions, broken-link formatting, synthesized index output, and absence of absolute fixture paths. Invoke tools through the public LangChain tool interface.

## Acceptance criteria

- No model is required to create or invoke the tools.
- Outputs are deterministic and fit for direct insertion into an LLM context.
- `pytest -m unit tests/unit/test_tools.py`, Ruff, and mypy pass.

## Out of scope

Async tools, tool-call tracing, model binding, and navigator logic.
