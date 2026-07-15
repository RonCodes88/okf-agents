# Contributing to okf-agents

Thanks for your interest in improving `okf-agents`. This project is a
young, alpha-stage library, so please open an issue to discuss non-trivial
changes before investing time in a pull request.

## Development setup

```bash
git clone https://github.com/RonCodes88/okf-agents.git
cd okf-agents
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer.

## Running checks locally

```bash
ruff check .                      # lint
mypy okf_agents tests          # strict type checking
pytest                            # unit tests + offline integration tests
pytest --cov                      # with coverage (>= 85% required)
```

`pytest` alone never contacts a network or model provider: opt-in
provider-integration and end-to-end tests are gated behind
`RUN_INTEGRATION_TESTS=1` / `RUN_E2E_TESTS=1` plus a provider API key, and
skip cleanly when those are unset. See [docs/testing.md](docs/testing.md).

## Making changes

1. Create a branch named `<type>/<short-kebab-case-description>`, where
   `<type>` is one of `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, or
   `ci` — for example `fix/broken-link-edge-case`.
2. Keep changes focused; add or update tests alongside any behavior change.
3. Write commit messages as clear, imperative
   [Conventional Commits](https://www.conventionalcommits.org/) subjects
   (72 characters or fewer), for example `fix: resolve nested reserved
   filenames correctly`. Omit bodies unless a maintainer asks for one.
4. Make sure `ruff`, `mypy`, and `pytest` all pass before opening a pull
   request.
5. Open a pull request against `main` describing the change and how you
   tested it. CI must pass before merge.

## Design constraints to respect

- Public APIs are fully typed; `mypy --strict` must stay clean.
- Unit tests are offline and deterministic — never call a real model or
  network service from `tests/unit`.
- Keep the two required runtime dependencies (`pydantic`, `pyyaml`) plus
  `langgraph`/`langchain-core` as the only hard dependencies; provider SDKs
  and vector-store packages stay optional.
- See [docs/tasks/00-shared-contracts.md](docs/tasks/00-shared-contracts.md)
  for the implementation contracts this project follows where the original
  specification is ambiguous.

## Reporting bugs and requesting features

Use the issue templates in `.github/ISSUE_TEMPLATE/`. Include a minimal
reproduction (a small bundle plus the code that triggers the problem) for
bug reports whenever possible.

## Security issues

Please do not file public issues for security vulnerabilities — see
[SECURITY.md](SECURITY.md).

## Code of conduct

Participation in this project is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md).
