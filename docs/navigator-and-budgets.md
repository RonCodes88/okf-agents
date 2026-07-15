# The navigator subgraph and its budgets

`create_okf_navigator(bundle, model, ...)` compiles a bounded LangGraph
state graph that answers a question by reading the bundle's index,
choosing concepts to read, optionally following links, and generating a
final answer with citations validated against what it actually read.

```python
from okf_agents import OKFBundle, create_okf_navigator

bundle = OKFBundle.load("./my_bundle")
navigator = create_okf_navigator(bundle, model, max_hops=3, max_concepts=10)
result = navigator.invoke({"question": "How are payments refunded?"})
print(result["answer"])
print(result["citations"])       # concept IDs the answer actually cites
print(result["traversal_path"])  # ["index", "concepts/payments", ...]
```

Invoke it with only `question`; every other `NavigatorState` field
(`visited`, `context`, `tokens_used`, `hops`, `answer`, `citations`,
`traversal_path`) is initialized internally and present when the graph
finishes. An empty question, or a negative/invalid configuration value,
raises `ValueError` rather than silently doing nothing.

## The read loop

1. **`read_index`** reads the bundle's original or synthesized root
   index and records `"index"` as the first entry in `traversal_path`.
2. **`plan`** asks `model` for initial candidate concept IDs, using
   validated structured output (a JSON object naming concept IDs).
3. Candidates are read in order, skipping anything already visited or
   unknown to the bundle, until a budget would be exceeded.
4. In **`progressive`** strategy (the default), the graph asks the model
   once per round whether the accumulated context is sufficient; if not,
   it reads a bounded number of the concepts linked from what it has
   already read, and asks again.
5. In **`exhaustive`** strategy, the graph instead traverses every
   resolved link reachable from what it has read, breadth-first, with no
   confidence-based early stopping — it simply expands until a budget is
   hit or there is nothing left to reach.
6. **`generate`** always runs, even if nothing could be read (for
   example, an empty bundle, or `max_hops=0`), and always produces an
   answer.

## Budgets are hard limits, not suggestions

Three independent budgets bound every run, and the graph is built so no
route through it can exceed them:

| Budget         | Meaning                                                              |
| -------------- | --------------------------------------------------------------------- |
| `max_hops`     | Maximum read *rounds* (the initial planned read counts as one).       |
| `max_concepts` | Maximum concepts read in total, across the whole run.                 |
| `token_budget` | Hard cap on estimated tokens accumulated in `context`.                |

A read that would push `tokens_used` past `token_budget` never happens —
the graph terminates traversal and moves straight to `generate` instead.
`max_hops=0` is a valid, meaningful configuration: the index is still
read and an answer is still generated, but no concept is ever read.
Token estimation defaults to `max(1, len(text) // 4)` and is injectable
via `token_estimator=` if you have a real tokenizer and want tighter
budgeting — no tokenizer dependency is required by default.

Model calls are similarly bounded: at most one `plan` call, one
`generate` call, and — in `progressive` mode only — one `decide` call per
additional read round. `exhaustive` mode never calls the model to decide
whether to keep going. No model call is ever retried; a model exception
propagates to the caller instead of being swallowed.

## Malformed model output never breaks traversal

If the model's `plan` or `decide` response is not valid JSON, or names no
concept IDs the bundle actually has, the navigator falls back to
deterministic behavior instead of guessing or crashing:

- A malformed or empty **plan** falls back to the bundle's lexical
  `search()` over the question, capped at `max_concepts`.
- A malformed **decide** response terminates traversal and moves to
  `generate`, using whatever context has been read so far.
- Citations the model invents that were never actually read are
  silently dropped: `citations` is always a subset of `visited`.

This means a flaky or unhelpful model degrades traversal quality but
never produces an exception, an infinite loop, or a citation pointing at
a concept the navigator never read.

## Embedding the navigator in a larger graph

`create_okf_navigator` returns an ordinary compiled `StateGraph`, so it
can be used as a node inside a parent graph — for example, downstream of
[the router](../README.md#router) — as long as the parent
state's `question`/`answer`/`citations` keys line up:

```python
from langgraph.graph import END, START, StateGraph

parent = StateGraph(ParentState)
parent.add_node("navigate", navigator)
parent.add_edge(START, "navigate")
parent.add_edge("navigate", END)
app = parent.compile()
```

See [tests/integration/test_hybrid_retriever.py](../tests/integration/test_hybrid_retriever.py)
for a worked example that bridges the router's `query` field into the
navigator's `question` field with conditional edges.
