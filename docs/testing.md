# Testing

`okf-agents` has three test tiers, matching `pytest` markers
`unit`, `integration`, and `e2e`. Only the first two run by default;
provider- and network-dependent tests are opt-in.

## Unit tests — `tests/unit/`, marker `unit`

Fully offline and deterministic: no network access, no real model, no
live vector store. Bundle fixtures live under `tests/fixtures/`. Model
and vector-store behavior is exercised with small, scripted fakes (a
`ScriptedChatModel` that replays canned JSON responses and fails if
called more times than scripted, and an in-memory fake `VectorStore`)
defined alongside each test module.

```bash
pytest -m unit
```

## Offline integration tests — `tests/integration/test_hybrid_retriever.py`

Also run by default, with no environment flags and no secrets: they use
an in-process, dependency-free vector store paired with deterministic
"fake embeddings" (`tests/integration/conftest.py`) — a hashed
bag-of-words representation, not a real embedding model — to prove that
`sync_bundle_to_vector_store`, `OKFGraphRetriever`, and the router/
navigator compose correctly end to end without needing any external
service.

## Provider-integration tests — `tests/integration/test_navigator.py`

Exercise the navigator against a **real** chat model. Gated behind two
things, both required:

- The environment variable `RUN_INTEGRATION_TESTS=1`.
- A supported provider key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (or
  an explicit `OKF_TEST_PROVIDER=anthropic|openai`), plus that provider's
  LangChain integration package (`langchain-anthropic` or
  `langchain-openai`) installed.

```bash
RUN_INTEGRATION_TESTS=1 ANTHROPIC_API_KEY=sk-... pytest -m integration
```

Without `RUN_INTEGRATION_TESTS=1`, or without a usable provider, these
tests **skip with a clear reason** — they never error, and they never run
by accident in an environment without secrets (such as a fork's pull
request CI). Model and model name are configurable via
`OKF_TEST_PROVIDER` / `OKF_TEST_MODEL`, defaulting to a small, low-cost
model per provider rather than hard-coding an expensive one.

## End-to-end tests — `tests/e2e/test_full_workflow.py`

Build a real tool-calling agent over `create_okf_tools` and a parent
graph combining the router with the navigator, both against a real
model. Gated behind `RUN_E2E_TESTS=1` plus the same provider
configuration as above:

```bash
RUN_E2E_TESTS=1 ANTHROPIC_API_KEY=sk-... pytest -m e2e
```

These tests assert **observable outcomes** — a tool was called, a
concept ID shows up in citations, routing happened before navigation —
never exact prose, since a real model's wording is not something a test
suite should pin down.

## Coverage

```bash
pytest --cov
```

Coverage is measured with branch coverage over `okf_agents/`; CI and
local runs both fail under 85% total coverage (`[tool.coverage.report]`
in `pyproject.toml`).
