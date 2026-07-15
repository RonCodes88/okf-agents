# Task 09 — Integration and end-to-end tests

## Goal

Prove that the completed components interoperate with a real vector store, LangGraph composition, and optionally a real chat model.

## Depends on

Tasks 04–08.

## Owned files

- `tests/integration/test_hybrid_retriever.py`
- `tests/integration/test_navigator.py`
- `tests/e2e/test_full_workflow.py`
- integration/e2e-only fixture helpers under `tests/`

Do not change production behavior merely to make a brittle integration assertion pass; report contract defects to the owning task.

## Test tiers

### Offline integration

Use an in-process vector store and deterministic local/fake embeddings to verify two syncs are idempotent, semantic entry results are expanded through links, `expand_hops=0` does not expand, and router conditional edges compose with navigator state.

### Provider integration

Gate real-model tests with `RUN_INTEGRATION_TESTS=1` and skip with a clear reason when no supported provider key exists. Parameterize provider/model selection through environment variables rather than hard-coding a paid model.

Verify a navigator answer is non-empty, citations are valid visited IDs, traversal starts at the index, all budgets hold, payment questions reach the payment concept, and unsupported questions produce a grounded “not found” answer.

### End to end

Gate with `RUN_E2E_TESTS=1`. Build one tool-calling agent and one parent graph containing router plus navigator. Assert observable outcomes and captured tool calls, not exact prose.

## Reliability rules

- Mark every test correctly and set explicit timeouts.
- Never print API keys, prompts containing secrets, or full provider responses on failure.
- Avoid tests whose only assertion depends on a model choosing one exact valid route.
- Ensure all resources are closed and vector-store data is isolated per test.

## Acceptance criteria

- Default `pytest` remains offline and does not collect a provider SDK error.
- Offline integration tests pass in CI without secrets.
- Provider and e2e suites skip cleanly when disabled.
- Enabled suites exercise real public APIs, not internal node functions.

## Out of scope

Benchmarks, load tests, multiple vector-store compatibility, and fixing external provider outages.
