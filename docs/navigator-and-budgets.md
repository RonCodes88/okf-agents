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
print(result["degraded"])        # True if any step's output failed to parse
```

Invoke it with only `question`; every other `NavigatorState` field
(`visited`, `context`, `tokens_used`, `hops`, `answer`, `citations`,
`traversal_path`, `degraded`, `degraded_steps`) is initialized internally
and present when the graph finishes. An empty question, or a
negative/invalid configuration value, raises `ValueError` rather than
silently doing nothing.

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

### The index has a fixed, unavoidable token cost

The bundle's index is read unconditionally before anything else — there
is no way to answer without it — so its estimated token cost is charged
against `token_budget` before a single concept is ever considered.
`create_okf_navigator` computes this cost eagerly (using the same
`token_estimator` you pass in, or the default) and raises `ValueError` at
construction time if `token_budget` is smaller than it, rather than
silently letting `tokens_used` exceed the configured cap once the graph
runs. This keeps the "hard cap, never exceeded" guarantee true for the
whole run, index included, instead of only for the concept-read loop.

Use `index_token_cost(bundle, token_estimator=None)` to compute this
fixed cost up front, before choosing a `token_budget`:

```python
from okf_agents.navigator import index_token_cost

floor = index_token_cost(bundle)          # e.g. 173
navigator = create_okf_navigator(bundle, model, token_budget=floor + 2000)
```

A `token_budget` set exactly to `index_token_cost(bundle)` is valid and
meaningful: the index is read, but no concept read can ever fit, so the
graph answers from the index alone (`plan` is skipped entirely, matching
`max_hops=0` and `max_concepts=0`'s "index-only" behavior). `OKFBundle.index()`
is a cheap deep-copy of a value computed once at bundle load time, so
calling `index_token_cost` costs nothing extra beyond the estimator call.

Model calls are similarly bounded: at most one `plan` call, one
`generate` call, and — in `progressive` mode only — one `decide` call per
additional read round. `exhaustive` mode never calls the model to decide
whether to keep going. No model call is ever retried; a model exception
propagates to the caller instead of being swallowed.

## Malformed model output never breaks traversal — and never fails open

If the model's `plan`, `decide`, or `generate` response doesn't parse
into its expected schema, the navigator fails **closed**, not open: it
never guesses its way into reading more than the model actually
justified, and it never leaves callers unable to tell that something
went wrong.

- A **plan** response that doesn't parse, doesn't match the expected
  shape (e.g. `concept_ids` isn't a list), or names no concept ID the
  bundle actually has (empty list, or every ID unknown) reads nothing
  that round. The graph proceeds straight to `generate` with just the
  index, the same way it would if `max_hops=0`. It does **not** fall
  back to a lexical search over the whole bundle — a single bad `plan`
  response can no longer cause the navigator to read all the way up to
  `max_concepts` on faith.
- A malformed **decide** response terminates traversal the same way it
  always has: it moves straight to `generate`, using whatever context
  has been read so far.
- Citations the model invents that were never actually read are
  silently dropped: `citations` is always a subset of `visited`. This one
  case is deliberately silent, not flagged — a citation being filtered
  out is a normal, expected outcome of validating against `visited`, not
  evidence the model's output was malformed.
- Whenever `plan`, `decide`, or `generate`'s response actually fails
  schema validation (as opposed to parsing fine but naming nothing
  useful), the returned state records it:
  - `degraded: bool` — `True` if any step failed validation during the
    run, `False` otherwise.
  - `degraded_steps: list[str]` — the ordered names of the steps that
    failed (`"plan"`, `"decide"`, `"generate"`), empty when nothing did.

  In particular, when `generate`'s response fails to parse, `answer`
  still contains the model's raw text (unchanged from before) and
  `citations` is `[]`, but now `degraded` is `True` and
  `degraded_steps` includes `"generate"` — so a caller can tell "this is
  a real grounded answer" from "the model returned garbage and this is
  that garbage verbatim" without guessing from `citations == []` alone
  (an empty-but-valid answer also has no citations).

```python
result = navigator.invoke({"question": "..."})
if result["degraded"]:
    # result["answer"] may be unparsed model text rather than a
    # validated, cited answer — decide how to surface that to your
    # caller/UI; result["degraded_steps"] says which step(s) failed.
    ...
```

This means a flaky or unhelpful model degrades traversal quality but
never produces an exception, an infinite loop, a citation pointing at a
concept the navigator never read, or a silent, indistinguishable
garbage-in-garbage-out answer.

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
