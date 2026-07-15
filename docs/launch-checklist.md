# Maintainer launch checklist

This checklist is **prepared material only**. Nothing on this page changes
a repository setting, publishes a package, creates a release, or posts an
announcement — every item here requires a maintainer to act manually and
separately.

## 1. PyPI name

- [x] Verified `okf-agents` is unclaimed on PyPI (checked
      `https://pypi.org/pypi/okf-agents/json` on 2026-07-14 → 404, not
      registered).
- [x] Maintainer: claim the name by publishing `0.1.0` through the
      [release workflow](../.github/workflows/release.yml) once the
      `release` GitHub Environment (see below) is configured.

## 2. GitHub repository settings (manual)

- [ ] Set the repository **About** description to exactly:

  > LangGraph and LangChain tools, navigator, and graph-aware retrieval for Open Knowledge Format bundles.

- [ ] Set repository **Topics** to:

  `langgraph` `langchain` `okf` `open-knowledge-format` `rag`
  `retriever` `knowledge-graph` `ai-agents` `python` `knowledge-management`

- [ ] Add a homepage URL (docs site or this repository) in the About panel.
- [ ] Set a social-preview image (Settings → General → Social preview),
      1280×640px, once a repository-owned asset exists — see
      [.github/assets/README.md](../.github/assets/README.md). Do not use
      a generated or third-party image.
- [ ] Configure a `release` GitHub Environment (Settings → Environments)
      with at least one required reviewer, and add it as a
      [PyPI trusted publisher](https://docs.pypi.org/trusted-publishers/)
      for this repository/workflow, so `.github/workflows/release.yml` can
      publish without a stored API token.

## 3. Consistency check

Confirm these all describe the project the same way before launch:

- [ ] README title + opening sentence
- [ ] `pyproject.toml` `description` and `keywords`
- [ ] `llms.txt` summary
- [ ] GitHub About text and Topics (above)
- [ ] PyPI project description (mirrors `pyproject.toml` at publish time)

## 4. Launch posts (drafted, not sent)

Send only after `0.1.0` is actually published to PyPI and CI is green on
`main`.

- [ ] **Hacker News (Show HN)** — title:
      `Show HN: LangGraph integration for OKF bundles (link-graph-aware retrieval)`
- [ ] **r/LangChain** and **r/LocalLLaMA** — short post linking the repo,
      focused on graph-aware retrieval vs. plain vector search; do not
      claim benchmark numbers that have not actually been measured.
- [ ] **LangChain Discord `#show-and-tell`** — one-paragraph summary plus
      the repo link.
- [ ] **X/Twitter** — one post, plain description, no unverified claims.
- [ ] **A written article** (Dev.to / personal blog) walking through the
      navigator subgraph and its budgets — the durable, linkable piece
      that the other posts point back to.

## 5. Ongoing

- [ ] Ship changelog entries for real, user-visible changes on a regular
      cadence; do not fabricate release activity.
- [ ] Keep `docs/api-reference.md` and the README's public API map in
      sync with `okf_agents/__init__.py` on every release.
