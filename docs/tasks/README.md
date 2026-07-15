# langgraph-okf implementation tasks

This directory turns `langgraph-okf-spec.md` into bounded work packages for coding agents. The original specification remains the product vision; [00-shared-contracts.md](00-shared-contracts.md) records clarified implementation contracts and takes precedence where the draft conflicts with the official OKF v0.1 specification.

## Agent workflow

For every task:

1. Read this file, `00-shared-contracts.md`, and the assigned task.
2. Work only in the task's **Owned files** unless a prerequisite defect makes a small cross-task edit unavoidable.
3. Implement production code and the named tests together.
4. Run the task's verification commands and report any command that could not run.
5. Do not publish packages, create releases, change repository settings, or post launch announcements unless separately authorized.

## Git conventions

Create one branch per task using `<type>/<short-kebab-case-description>`. Use the standard types `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, and `ci`. Do not include an AI agent or model name in branch names.

Recommended task branches:

```text
chore/project-foundation
feat/okf-models-parser
feat/bundle-graph-search
feat/langchain-agent-tools
feat/keyword-retriever
feat/navigator-subgraph
feat/router-node
feat/vector-graph-retriever
test/integration-e2e
docs/release-discoverability
```

Commit messages must be clear, high-level one-line summaries using Conventional Commits, for example `feat: add deterministic OKF bundle search`. Use imperative wording, keep the subject at 72 characters or fewer, and omit bodies unless a maintainer explicitly requests one. Never include an AI agent/model name, AI attribution, or AI-generated co-author trailer.

## Dependency graph

```text
01 project foundation
  └─ 02 models and parser
       └─ 03 bundle, graph, and search
            ├─ 04 agent tools
            ├─ 05 keyword retriever
            ├─ 06 navigator subgraph
            ├─ 07 router node
            └─ 08 vector indexing and graph retriever
                 └─ 09 integration and end-to-end tests

01–09 ── 10 docs, CI, packaging, and launch assets
```

Tasks 04–08 may run in parallel after Task 03, except Task 08 also builds on the document conversion contract introduced by Task 05. Task 09 starts only after Tasks 04–08 are merged. Task 10 should be finalized last, although its documentation can be drafted earlier.

## Task index

- [01 — Project foundation](01-project-foundation.md)
- [02 — Models and parser](02-models-and-parser.md)
- [03 — Bundle, graph, and search](03-bundle-graph-search.md)
- [04 — LangChain agent tools](04-agent-tools.md)
- [05 — Keyword retriever](05-keyword-retriever.md)
- [06 — Navigator subgraph](06-navigator-subgraph.md)
- [07 — Router node](07-router-node.md)
- [08 — Vector indexing and graph retriever](08-vector-indexing-graph-retriever.md)
- [09 — Integration and end-to-end tests](09-integration-e2e.md)
- [10 — Docs, CI, packaging, and launch assets](10-release-and-discoverability.md)

## Definition of done

A task is complete only when its acceptance criteria pass, public APIs are typed, errors are covered, and no new Ruff or mypy failures are introduced. Unit tests must not call an external model or service. Changes should preserve the minimum Python version, Python 3.11.
