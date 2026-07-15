## Summary

<!-- What does this change do, and why? -->

## Test plan

<!-- How did you verify this change? Check what applies. -->

- [ ] `ruff check .`
- [ ] `mypy langgraph_okf tests`
- [ ] `pytest` (offline unit + integration tests)
- [ ] `pytest --cov` (coverage still >= 85%)
- [ ] Added or updated tests for the behavior change

## Checklist

- [ ] Public API changes are typed and documented (README/`docs/`)
- [ ] No new hard runtime dependency was added without discussion
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
