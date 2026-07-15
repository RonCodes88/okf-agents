# Task 01 — Project foundation

## Goal

Create an installable Python 3.11 package with quality tooling and the shared exception hierarchy. Do not implement bundle behavior yet.

## Depends on

None.

## Owned files

- `pyproject.toml`
- `langgraph_okf/__init__.py`
- `langgraph_okf/exceptions.py`
- `langgraph_okf/_internal/__init__.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `.gitignore`

## Work

1. Configure a PEP 517 build and package metadata for `langgraph-okf` version `0.1.0`.
2. Require Python 3.11+, `langgraph`, `langchain-core`, Pydantic v2, and PyYAML. Add test, lint, type-check, vector-test, and release tooling as development/optional groups without forcing vector-store implementations on base users.
3. Configure Ruff, strict mypy, pytest markers (`unit`, `integration`, `e2e`), and branch coverage with an 85% threshold.
4. Implement `OKFError`, `BundleNotFoundError`, `BundleValidationError`, `ConceptNotFoundError`, and `LinkResolutionError`.
5. Give validation and lookup exceptions stable human-readable messages and preserve structured attributes such as `failed_files` and `concept_id`.
6. Keep root `__init__.py` limited to version and currently available exception exports. Later tasks add their own public exports.

## Tests

Add exception tests only if behavior is more than a trivial subclass. Verify structured attributes, deterministic message formatting, and inheritance from `OKFError`.

## Acceptance criteria

- `python -m pip install -e .` succeeds in a clean Python 3.11+ environment.
- Importing `langgraph_okf` and all exceptions succeeds.
- `pytest -m unit`, `ruff check .`, and `mypy langgraph_okf` are wired and runnable.
- Base installation does not install Chroma or a model-provider SDK.

## Out of scope

Models, Markdown parsing, bundle loading, GitHub Actions, README content, publishing, and repository settings.
