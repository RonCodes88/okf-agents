# Task 10 — Docs, CI, packaging, and launch assets

## Goal

Finish the public API, automate quality checks and packaging, and prepare accurate discoverability and release materials.

## Depends on

Tasks 01–09. Drafting may start earlier, but examples and API exports must be verified against merged code.

## Owned files

- `langgraph_okf/__init__.py`
- `pyproject.toml` metadata not owned by feature tasks
- `README.md`
- `CHANGELOG.md`
- `LICENSE`
- `llms.txt`
- `docs/**` except `docs/tasks/**`
- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- `.github/ISSUE_TEMPLATE/**`
- `.github/pull_request_template.md`

## Work

1. Export the source specification's public API plus `SyncResult`; keep `__all__` and version metadata consistent.
2. Write a README with what/who/install in the first lines, a tested quick start, bundle-tools example, navigator example, retriever example, compatibility notes, limitations, badges, and links to official OKF.
3. Add API and conceptual documentation explaining concept IDs, optional indexes, broken links, lexical weighting, vector-store requirements, budgets, and sync idempotency.
4. Add package classifiers, project URLs, license, keywords, optional dependency groups, typed-package marker if applicable, and source distribution inclusion rules.
5. Add CI jobs for Ruff, strict mypy, Python-version matrix unit tests with coverage, offline integration tests, package build, and artifact validation. Provider tests must not run on untrusted pull requests.
6. Add a release workflow triggered by an explicit version tag, using PyPI trusted publishing and an environment approval gate. Never embed tokens.
7. Add `llms.txt`, changelog, issue/PR templates, and a launch checklist covering PyPI name verification, GitHub About/topics, Show HN, relevant communities, an article, and social posts.
8. Document the proposed GitHub About text and topics in a maintainer checklist; repository settings and external posts remain manual.

## Verification

- Execute every README code sample as a test or script.
- Run the full offline test, lint, type, coverage, build, and package-inspection commands.
- Install the built wheel in a clean environment and smoke-test every public import.
- Validate workflow YAML and ensure release jobs have least-privilege permissions.
- Search docs for stale `updated`, mandatory-index claims, separate `embeddings` sync arguments, and unimplemented APIs.

## Acceptance criteria

- Documentation matches actual signatures and behavior.
- Default installation excludes provider and vector-store implementations.
- CI can run on forks without secrets.
- Release automation cannot publish from an ordinary branch push.
- Launch materials are prepared but no package, release, repository setting, or announcement is changed without separate user authorization.
