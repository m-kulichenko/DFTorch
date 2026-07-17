# Technology Stack

**Analysis Date:** 2026-07-17

## Languages

**Primary:**
- Python >=3.11 - Library source in `src/dftorch/`, tests in `tests/`, examples in `experiments/`.

**Secondary:**
- TOML - Packaging, dependency, pytest, and Ruff configuration in `pyproject.toml`.
- YAML - GitHub Actions workflows in `.github/workflows/*.yml` and pre-commit hooks in `.pre-commit-config.yaml`.
- Dockerfile - Containerized test/runtime image in `Dockerfile`.
- Jupyter notebooks - Example workflows in `experiments/*.ipynb` and `Untitled.ipynb`.
- Markdown - User and release documentation in `README.md`, `CONTRIBUTING.md`, and `CHANGELOG.md`.

## Runtime

**Environment:**
- CPython 3.11 or newer. `pyproject.toml` sets `requires-python = ">=3.11"`.
- CI pins Python 3.11 through `actions/setup-python@v5` in `.github/workflows/tests.yml` and `.github/workflows/release.yml`.
- Container runtime uses `python:3.11-slim` in `Dockerfile`.
- PyTorch 2.10.0 is locked in `uv.lock` and powers tensor math, GPU execution, autograd, `torch.compile`, and `torch.distributed`.
- CUDA/GPU execution is optional. Code selects CUDA when available in examples (`README.md`) and GPU-specific paths appear in `src/dftorch/MD.py`, `src/dftorch/ewald_pme/__init__.py`, and `src/dftorch/sedacs/sedacs_interface.py`.
- Triton 3.6.0 is present as a Linux x86_64 PyTorch dependency in `uv.lock` and is used by `src/dftorch/ewald_pme/ewald_triton.py` when importable on CUDA.

**Package Manager:**
- uv - Recommended package/environment manager in `README.md`, `CONTRIBUTING.md`, `Dockerfile`, and CI workflows.
- Lockfile: present at `uv.lock`.
- pip - Supported install fallback through `pip install .` in `README.md`.

## Frameworks

**Core:**
- PyTorch 2.10.0 - Core numerical framework for SCC-DFTB, molecular dynamics, autograd, GPU execution, and neural network models. Used throughout `src/dftorch/*.py`.
- setuptools >=68 with wheel - Build backend configured in `pyproject.toml` under `[build-system]`.
- NumPy 2.4.4 - Array handling, element data, parameter files, and I/O. Used in `src/dftorch/_elements.py`, `src/dftorch/_io.py`, and `src/dftorch/_dftd3.py`.
- SciPy 1.17.1 - Scientific computing dependency declared in `pyproject.toml` and locked in `uv.lock`.
- pandas 3.0.2 - Data handling dependency declared in `pyproject.toml` and locked in `uv.lock`.

**Testing:**
- pytest 9.0.3 - Test runner configured in `pyproject.toml` and used by `tests/*.py`, `Dockerfile`, `.github/workflows/tests.yml`, and `.pre-commit-config.yaml`.
- pytest markers - `slow` and `gpu` markers are configured in `pyproject.toml`.

**Build/Dev:**
- Ruff 0.15.12 - Linting and formatting. Configured in `pyproject.toml`, enforced by `.github/workflows/tests.yml`, and installed as a pre-commit hook in `.pre-commit-config.yaml`.
- mypy 2.0.0 - Optional dev type checker declared in `pyproject.toml` and locked in `uv.lock`; no dedicated mypy config detected.
- pre-commit 4.6.0 - Local hooks configured in `.pre-commit-config.yaml`.
- Docker - Containerized test image defined by `Dockerfile` and exercised by `.github/workflows/container-tests.yml` and `.github/workflows/container-release.yml`.
- GitHub Actions - CI, release build, TestPyPI publishing, and GHCR container publishing in `.github/workflows/`.

## Key Dependencies

**Critical:**
- `torch` 2.10.0 - Required for all core computation, tensor storage, device movement, autograd, `torch.nn.Module` constants, distributed SEDACS paths, and optional compile paths.
- `numpy` 2.4.4 - Required for array conversion, local data loading, structure I/O, and reference parameter files.
- `scipy` 1.17.1 - Required scientific computing dependency for numerical operations.
- `pandas` 3.0.2 - Required dependency for tabular scientific data handling.
- `nvalchemi-toolkit-ops` 0.3.1 - Required dependency that exposes the optional NVIDIA ALCHEMI neighbor-list backend imported from `nvalchemiops.torch.neighbors` in `src/dftorch/_nearestneighborlist.py`.

**Infrastructure:**
- `warp-lang` 1.13.0 - Transitive dependency of `nvalchemi-toolkit-ops` in `uv.lock`.
- `triton` 3.6.0 - Transitive PyTorch dependency and direct optional backend import in `src/dftorch/ewald_pme/ewald_triton.py`.
- `mpi4py` 4.1.1 - Optional `sedacs` extra dependency for MPI-compatible SEDACS workflows.
- `numba` 0.65.1 - Optional `sedacs` extra dependency.
- `toml` 0.10.2 - Optional `sedacs` extra dependency.
- `scikit-learn` 1.8.0 - Optional `sedacs` extra dependency.
- `setuptools` 82.0.1 - Locked build tooling in `uv.lock`.

## Configuration

**Environment:**
- Use `uv venv --python 3.11`, `uv sync`, and `uv run pytest` for local development as documented in `README.md` and `CONTRIBUTING.md`.
- Install optional SEDACS support with `uv pip install -e ".[sedacs]"` as documented in `README.md`.
- `DFTORCH_ENABLE_COMPILE` controls whether `_maybe_compile()` wraps selected functions in `torch.compile` in `src/dftorch/_tools.py`.
- PyTorch compile/runtime behavior may also be controlled through `TORCHDYNAMO_DISABLE`, `TORCH_COMPILE_DISABLE`, `TORCHINDUCTOR_DISABLE`, `TORCH_LOGS`, `TORCHINDUCTOR_VERBOSE`, and `TORCHDYNAMO_VERBOSE`, as shown in `README.md`, `tests/test_*.py`, and `experiments/*.py`.
- Distributed SEDACS examples expect `LOCAL_RANK`, `WORLD_SIZE`, and `RANK` from `torch.distributed.launch` or an equivalent launcher in `experiments/DDP_2.py`.
- Runtime simulation parameters are passed through dictionaries with keys such as `FILENAME`, `SKFPATH`, `T_ELECTRONIC`, `RCUT_ELECTRONIC`, `RCUT_REPULSIVE`, and `COUL_METHOD`, as shown in `README.md` and consumed by `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, and `src/dftorch/ESDriver.py`.
- `.env` files: Not detected.

**Build:**
- `pyproject.toml` is the source of package metadata, dependencies, optional extras, pytest settings, and Ruff settings.
- `uv.lock` pins resolved dependency versions.
- `Dockerfile` builds a Python 3.11 slim image, installs uv, copies `pyproject.toml`, `uv.lock`, `README.md`, `src/`, and `tests/`, runs `uv sync --extra dev`, and defaults to `uv run pytest`.
- `.pre-commit-config.yaml` runs Ruff with `--fix`, Ruff format, and quick pytest.
- `.github/workflows/tests.yml` runs Ruff lint, Ruff format check, and pytest.
- `.github/workflows/release.yml` builds distributions and publishes to TestPyPI via trusted publishing.
- `.github/workflows/container-release.yml` builds and pushes container images to GHCR.

## Platform Requirements

**Development:**
- Python >=3.11.
- uv installed locally for the documented workflow.
- PyTorch-compatible CPU environment for standard tests and calculations.
- CUDA-compatible GPU environment for GPU acceleration, Triton Ewald backend, and large MD workloads.
- Optional MPI/distributed environment for SEDACS workflows using `torch.distributed` and optional `mpi4py`.
- Local Slater-Koster parameter files (`.skf`) are required at runtime through the `SKFPATH` parameter. Examples and tests include data under `experiments/sk_orig/` and `tests/data_skf_mio-1-1/`.

**Production:**
- Library/package distribution through Python packaging from `pyproject.toml`.
- Container image deployment through GHCR using `.github/workflows/container-release.yml`.
- TestPyPI publishing through `.github/workflows/release.yml`; PyPI production publishing is not configured.

---

*Stack analysis: 2026-07-17*
