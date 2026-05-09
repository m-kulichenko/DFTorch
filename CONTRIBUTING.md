# Contributing to DFTorch

Thank you for your interest in DFTorch! This guide covers everything you need
to contribute effectively.

## Development setup

DFTorch uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
git clone https://github.com/m-kulichenko/DFTorch.git
cd DFTorch
uv venv --python 3.11
uv pip install -e ".[dev]"
pre-commit install
```

## Code style

DFTorch enforces consistent code style using [Ruff](https://docs.astral.sh/ruff/).

Run locally before committing:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Pre-commit hooks run these automatically on every `git commit`. Contributions
that have not been formatted with Ruff will not be merged.

## Pre-commit hooks

After installing dev dependencies, activate pre-commit:

```bash
pre-commit install
```

This installs:

- **ruff** — linting and auto-fix
- **ruff-format** — code formatting
- **pytest** (quick) — runs the test suite on every commit

## Tests

Tests live in `tests/`. Run the full suite with:

```bash
uv run pytest
```

Mark slow or GPU-only tests with the provided markers:

```python
import pytest

@pytest.mark.slow
def test_long_md_run(): ...

@pytest.mark.gpu
def test_cuda_kernel(): ...
```

Run tests excluding slow ones:

```bash
uv run pytest -m "not slow"
```

Aim for >80 % test coverage on any new module you add.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(md): add Langevin thermostat
fix(scf): correct Pulay mixing for spin-polarized systems
docs(tutorial): add open-shell O2 example
test(scf): add convergence regression tests
```

Common scopes: `md`, `scf`, `io`, `forces`, `pme`, `dftb3`, `spin`, `sedacs`.

## Submitting changes

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature
   ```
2. Make your changes and ensure all tests pass and pre-commit hooks are green.
3. Open a pull request with a clear description of what was changed and why.

## Reporting issues

Please include:

- DFTorch version (`python -c "import dftorch; print(dftorch.__version__)"`)
- Python version and OS
- Minimal reproducible example
- Full traceback
