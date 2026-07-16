# Core concepts

This page explains the data model `okf-agents` builds from an
[OKF](https://okf.md) bundle: what counts as a concept, how concept IDs are
derived, how the root index works, how links resolve, and how lexical
search ranks results. It documents the implementation contracts this
library follows, which clarify a few points the draft OKF v0.1
specification leaves ambiguous.

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
`resource`, `tags`, `aliases`, and `timestamp` (an ISO 8601 datetime). Any
other keys are preserved in `Concept.frontmatter.extra` rather than being
silently dropped. A missing or empty `type` makes the file invalid; there
is no "strict mode" toggle in v0.1, and this is the one validation rule
that always applies. `aliases` follows Obsidian's own frontmatter
convention (a list of alternate names for the note) and participates in
wikilink resolution — see "Links and resolution" below.

Loading a bundle is eager: `OKFBundle.load()` parses every concept file up
front, and by default (`on_error="raise"`, the default) it is
all-or-nothing — every validation failure (bad YAML, empty `type`,
unreadable files, paths escaping the bundle root) is aggregated into one
`BundleValidationError` keyed by stable, bundle-root-relative paths,
rather than failing on the first bad file.

Pass `on_error="skip"` for partial loading of a messy, organically-grown
bundle: invalid files (concept files and an invalid root `index.md`
alike) are excluded from the loaded bundle instead of blocking it
entirely. The bundle loads normally from whatever files are valid; a link
to an excluded concept simply becomes an unresolved edge, exactly like a
link to a concept that never existed. The excluded paths and their
reasons are available afterwards on `bundle.skipped_files` (same shape as
`BundleValidationError.failed_files`) so they can be surfaced and fixed
later without blocking on them up front. This does not relax the
validation rule itself — a missing or empty `type` still makes a file
invalid, there is still no "strict mode" toggle for *that* rule — it only
changes what happens to files that fail it.

If a bundle ends up with zero concepts (a typo'd path with no matching
`.md` files, or every file skipped), `OKFBundle.load()` emits a
`UserWarning` rather than succeeding silently, since a `concept_count` of
zero is far more often a mistake than an intentionally empty bundle.

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

Two internal link syntaxes are recognized, because the two knowledge-base
tools this library implicitly targets — Obsidian and plain Markdown wikis
— default to different ones.

**Standard inline Markdown links**, `[text](target)`, where `target` is
either **bundle-relative** (starts with `/`, resolved against the bundle
root) or **file-relative** (resolved against the linking file's
directory, e.g. `../file.md`) — resolution is pure path arithmetic, with
no knowledge of the rest of the bundle required. Images, reference-style
links, autolinks, links inside fenced code blocks, fragment-only links
(`#section`), and external URLs are not treated as graph edges in v0.1.

**Obsidian-style wikilinks**, `[[target]]`, `[[target|Display text]]`,
`[[target#Heading]]`, and `[[target^blockid]]` — resolution is by
case-insensitive **filename or title match**, not by path, matching how
Obsidian itself resolves them: the `target` is looked up against every
loaded concept's bare filename, its frontmatter `title`, its frontmatter
`aliases`, and its full concept ID (in that the full ID is indexed too, so
a path-qualified wikilink like `[[folder/Note]]` — the same style of link
Obsidian itself writes to disambiguate a collision — always resolves
deterministically even when the bare title or filename is ambiguous). A
`#Heading` or `^blockid` suffix and a trailing `.md` are stripped before
lookup and are not resolved to a location inside the target file: both
`[[Note]]` and `[[Note#Heading]]` point at the whole `Note` concept, since
sub-file anchors are outside this library's concept-level granularity.
When the `target` contains a `|`, the text after it is the display text
(`anchor_text`); without one, `anchor_text` is the raw target text as
written, anchor included, matching what Obsidian renders. File/block
embeds (`![[target]]`) are not treated as links, mirroring how `![...]()`
image syntax is excluded from Markdown links. `![[embed]]` and Obsidian's
`[[Note]]` are otherwise unrelated to Notion, whose own Markdown export
already produces standard `[Page](https://notion.so/Page-<uuid>)` links —
wikilink-shaped text only shows up in bundles produced by a third-party
Notion-to-Obsidian converter.

Because wikilink resolution needs to see every concept's filename, title,
and aliases at once, it only happens once the whole bundle is loaded
(inside `OKFBundle.load()`); a single concept parsed in isolation (e.g.
`Concept.outbound_links`) records a wikilink's raw casefolded lookup key,
not yet a real concept ID.

**Ambiguity is reported, never silently guessed.** If a wikilink's lookup
key matches more than one loaded concept — e.g. two files both titled
"Orders" in different folders — the edge is left unresolved and marked
`ambiguous=True` rather than the library picking one candidate. This
mirrors that Obsidian's own "shortest path when possible" behavior is
applied at link-*creation* time inside the app, using vault settings this
library does not have access to when reading files after the fact;
picking a candidate here could silently point a reader at the wrong
concept, which is worse than a link that is visibly unresolved. Bundle
authors can disambiguate with a path-qualified `[[folder/Note]]` link or a
unique `aliases:` entry.

Every internal link becomes a `LinkEdge` with `source_id`, `target_id`,
`anchor_text`, a `resolved` flag, a `link_kind` (`"markdown"` or `"wiki"`),
and an `ambiguous` flag (always `False` for Markdown links, and for
wikilinks except the multiple-match case above). A link to a concept that
does not exist in the bundle — including a wikilink whose lookup key
matches nothing — is kept as an **unresolved** edge (`resolved=False`)
rather than being dropped or making the bundle invalid — broken links are
a normal, tolerated part of a bundle, and tools like `list_links` surface
them explicitly rather than hiding them.

`BundleIndex.concept_ids` (the root index's link targets, see above) is
extracted the same way but is never re-resolved against the whole bundle:
a wikilink in `index.md` is recorded under its raw casefolded lookup key,
even when the real link graph (`links_from`/`backlinks`/`neighbors` below)
would resolve or flag it as ambiguous. Consumers that walk the root index
already tolerate IDs that don't match a loaded concept (a pre-existing
possibility for broken Markdown links in `index.md`), so this is a
deliberate, narrow scope boundary rather than an oversight.

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
