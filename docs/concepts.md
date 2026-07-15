# Core concepts

This page explains the data model `okf-agents` builds from an
[OKF](https://okf.md) bundle: what counts as a concept, how concept IDs are
derived, how the root index works, how links resolve, and how lexical
search ranks results. It documents the implementation contracts this
library follows, which clarify a few points the draft OKF v0.1
specification leaves ambiguous — see
[docs/tasks/00-shared-contracts.md](tasks/00-shared-contracts.md) for the
full list.

## What is a concept?

A concept is any UTF-8 `.md` file in the bundle, at any directory depth,
**except** a file literally named `index.md` or `log.md`. Those two names
are reserved: `index.md` at the bundle root becomes the root index (see
below), and both names are skipped everywhere else so a bundle can use them
for navigation pages or changelogs without breaking concept discovery.

Every concept file must start with YAML frontmatter containing a
non-empty `type` field:

```markdown
---
type: table
title: Orders
description: Order lifecycle and fulfilment records.
tags: [sales, commerce]
timestamp: 2026-07-01T09:30:00
---

# Orders

Each order belongs to a [customer](customers.md).
```

Recognized frontmatter fields are `type`, `title`, `description`,
`resource`, `tags`, and `timestamp` (an ISO 8601 datetime). Any other keys
are preserved in `Concept.frontmatter.extra` rather than being silently
dropped. A missing or empty `type` makes the file invalid; there is no
"strict mode" toggle in v0.1, and this is the one validation rule that
always applies.

Loading a bundle is eager and all-or-nothing: `OKFBundle.load()` parses
every concept file up front and aggregates every validation failure (bad
YAML, empty `type`, unreadable files, paths escaping the bundle root) into
one `BundleValidationError` keyed by stable, bundle-root-relative paths,
rather than failing on the first bad file.

## Concept IDs

A concept's ID is its path relative to the bundle root, using forward
slashes, with the trailing `.md` removed:

| File on disk (relative to bundle root) | Concept ID          |
| --------------------------------------- | -------------------- |
| `concepts/orders.md`                    | `concepts/orders`     |
| `guides/getting-started.md`             | `guides/getting-started` |
| `alpha.md`                               | `alpha`                |

IDs are what you pass to `bundle.get(concept_id)`, what tool calls like
`read_concept` expect, and what shows up in `Document` metadata as
`concept_id`. They are stable across machines because they never include
the bundle's absolute filesystem path — only `Concept.path` and
`Document.metadata["path"]`/`["bundle_root"]` carry absolute, resolved
paths.

## The root index

A root `index.md` is optional. When present, it is parsed like any other
Markdown file (frontmatter is not required for the index) into a
`BundleIndex`: `title` is its first H1 heading if present, `description`
is its first non-heading paragraph if present, `body` is the file's
Markdown, and `concept_ids` lists every internal concept link the index
references, in first-seen order.

When no root `index.md` exists, `OKFBundle.load()` **synthesizes** one in
memory from the sorted list of loaded concepts — no file is ever written
to disk. Code that calls `bundle.index()` does not need to know which case
it is in; both return the same `BundleIndex` shape. This is why the
navigator subgraph can always read "the index" as its first traversal
step, whether or not the bundle author wrote one.

## Links and resolution

Internal links are standard inline Markdown links, `[text](target)`,
where `target` is either **bundle-relative** (starts with `/`, resolved
against the bundle root) or **file-relative** (resolved against the
linking file's directory, e.g. `../file.md`). Images, reference-style
links, autolinks, links inside fenced code blocks, fragment-only links
(`#section`), and external URLs are not treated as graph edges in v0.1.

Every internal link becomes a `LinkEdge` with `source_id`, `target_id`,
`anchor_text`, and a `resolved` flag. A link to a concept that does not
exist in the bundle is kept as an **unresolved** edge (`resolved=False`)
rather than being dropped or making the bundle invalid — broken links are
a normal, tolerated part of a bundle, and tools like `list_links` surface
them explicitly rather than hiding them.

`bundle.links_from(concept_id)` and `bundle.backlinks(concept_id)` return
copies of the outbound/inbound edges for one concept. `bundle.neighbors()`
performs a breadth-first walk over **resolved** edges only, in a chosen
direction (`"out"`, `"in"`, or `"both"`), excluding the start node and
handling cycles safely; results are ordered by distance and then concept
ID so traversal order is always deterministic.

## Lexical search

`bundle.search(query, top_k=5)` is a dependency-free, weighted lexical
search — not TF-IDF and not an embedding model. The query is split into
case-folded word tokens, and each token is matched as a **substring**
against four fields, each with its own weight:

| Field         | Weight |
| ------------- | ------ |
| `title`       | 4      |
| `tags`        | 3      |
| `description` | 2      |
| `body`        | 1      |

A concept must match at least one query token to appear in results at
all. Matches are ranked by descending total score, then by concept ID, so
ties are always broken the same way. This makes `search()` fully
deterministic and safe to unit test without a model — it is what
`OKFRetriever` and the `search_concepts` tool are built on, and what the
navigator falls back to when a model's suggested concept IDs are missing
or malformed.
