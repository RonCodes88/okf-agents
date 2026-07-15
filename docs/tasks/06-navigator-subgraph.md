# Task 06 — Navigator subgraph

## Goal

Build a bounded LangGraph subgraph that reads an OKF index, plans traversal, expands linked concepts, and returns a grounded answer with concept citations.

## Depends on

Task 03. It may use Task 04 formatting helpers only after Task 04 is merged; do not make agent tools part of the graph's runtime.

## Owned files

- `langgraph_okf/navigator.py`
- `tests/unit/test_navigator.py`

Real-provider tests belong to Task 09.

## API and state

Implement `create_okf_navigator(bundle, model, max_hops=3, max_concepts=10, token_budget=8000, strategy="progressive", token_estimator=None) -> CompiledGraph`.

Expose a documented `NavigatorState` containing `question`, `visited`, `context`, `tokens_used`, `hops`, `answer`, `citations`, and `traversal_path`. Invocation requires only `question`; initialize all other fields internally.

## Graph behavior

1. Read the original or synthesized index and record `"index"` first in `traversal_path`.
2. Ask the model for initial candidate concept IDs using validated structured output.
3. Read only known, unvisited concepts within concept and token budgets.
4. In progressive mode, ask once per round whether context is sufficient; otherwise choose linked candidates and loop.
5. In exhaustive mode, traverse reachable links breadth-first without confidence-based early stopping.
6. Generate a final answer even when no concept can be read.
7. Validate final citations against visited concept IDs; discard invented citations.

Terminate before a read that would exceed a hard budget. `max_hops=0` permits the index but no concept expansion. Empty questions and invalid configuration raise `ValueError`.

## Failure behavior

Malformed planning/decision output falls back to deterministic lexical candidates or termination. A model exception propagates; do not silently produce a fabricated answer. Never retry a model call without an explicit bounded retry setting.

## Tests

Use deterministic fake chat models. Cover state initialization, plan/read/decide/generate flow, progressive early stop, exhaustive traversal, cycles, invalid model output fallback, unknown candidate IDs, citation filtering, every budget boundary, no-answer behavior, and graph embedding as a parent node. Assert model call counts to catch accidental loops.

## Acceptance criteria

- Every graph route has a provable bound from `max_hops`, `max_concepts`, and `token_budget`.
- Unit tests make no API calls and need no provider package.
- `pytest -m unit tests/unit/test_navigator.py`, Ruff, and mypy pass.

## Out of scope

Streaming, async invocation, persistent checkpoints, human approval, and retrieval via vector stores.
