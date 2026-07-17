# External Integrations

**Analysis Date:** 2026-07-17

## APIs & External Services

**Scientific compute libraries:**
- PyTorch - Core tensor, autograd, GPU, compile, neural network, and distributed execution backend.
  - SDK/Client: `torch`
  - Auth: Not applicable
  - Code paths: `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/_tools.py`, `src/dftorch/_ml_sk.py`, `src/dftorch/ewald_pme/`, `src/dftorch/sedacs/`
- NVIDIA ALCHEMI / nvalchemiops - Optional accelerated neighbor-list backend selected by `use_alchemi=True` or module global `USE_ALCHEMI`.
  - SDK/Client: `nvalchemi-toolkit-ops` provides `nvalchemiops.torch.neighbors.cell_list`
  - Auth: Not applicable
  - Code paths: `src/dftorch/_nearestneighborlist.py`
- Triton - Optional CUDA Ewald backend when CUDA is available and `src/dftorch/ewald_pme/ewald_triton.py` imports successfully.
  - SDK/Client: `triton`, `triton.language`
  - Auth: Not applicable
  - Code paths: `src/dftorch/ewald_pme/__init__.py`, `src/dftorch/ewald_pme/ewald_triton.py`
- SEDACS - Optional large-scale simulation integration for graph partitioning and distributed workflows.
  - SDK/Client: External `sedacs` Python package/modules, plus `torch.distributed`
  - Auth: Not applicable
  - Code paths: `src/dftorch/sedacs/__init__.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`, `src/dftorch/sedacs/sedacs_interface.py`, `experiments/DDP_2.py`

**Package and release services:**
- PyPI package index - Dependency resolution and package downloads through uv/pip.
  - SDK/Client: `uv`, `pip`
  - Auth: Not detected in repository files
  - Code paths: `pyproject.toml`, `uv.lock`, `Dockerfile`, `.github/workflows/*.yml`
- TestPyPI - Release workflow publishes built distributions to TestPyPI using trusted publishing.
  - SDK/Client: `pypa/gh-action-pypi-publish@release/v1`
  - Auth: GitHub OIDC trusted publishing through workflow `id-token: write`; no repository secret value is stored in code.
  - Code paths: `.github/workflows/release.yml`
- GitHub Container Registry - Container release workflow pushes Docker images to GHCR.
  - SDK/Client: Docker CLI, `docker/login-action@v3`
  - Auth: `secrets.GITHUB_TOKEN`
  - Code paths: `.github/workflows/container-release.yml`
- GitHub Actions - CI/CD runner for linting, tests, release builds, and container builds.
  - SDK/Client: GitHub-hosted actions (`actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v3`, `actions/upload-artifact@v4`, `actions/download-artifact@v4`)
  - Auth: `secrets.GITHUB_TOKEN` for GHCR publishing; OIDC permission for TestPyPI publishing
  - Code paths: `.github/workflows/tests.yml`, `.github/workflows/release.yml`, `.github/workflows/container-tests.yml`, `.github/workflows/container-release.yml`

**Application APIs:**
- HTTP clients, REST APIs, GraphQL APIs, cloud SDKs, payment APIs, and email/SMS APIs: Not detected.
  - SDK/Client: Not detected
  - Auth: Not detected

## Data Storage

**Databases:**
- Not detected.
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only.
  - Input structures are read from `FILENAME` paths by `src/dftorch/Structure.py` and `src/dftorch/Constants.py`.
  - XYZ and PDB files are parsed by `read_xyz()` and `read_pdb()` in `src/dftorch/_io.py`.
  - Slater-Koster parameter files are loaded from `SKFPATH` by `src/dftorch/Constants.py` and `src/dftorch/_bond_integral.py`.
  - GBSA parameter files are read by `read_param_file()` in `src/dftorch/_gbsa.py`.
  - D3 reference data is loaded from `src/dftorch/params/dftd3_reference.npz` by `src/dftorch/_dftd3.py`.
  - ML-SK checkpoints are loaded with `torch.load()` by `load_ml_sk_model()` in `src/dftorch/_ml_sk.py`.
  - Trajectory and structure outputs are written locally by `src/dftorch/_io.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`, and `src/dftorch/sedacs/MD.py`.
  - Bundled parameter/reference assets live in `src/dftorch/params/`, `tests/data_skf_mio-1-1/`, `tests/f_orbital_data/`, `experiments/sk/`, and `experiments/sk_orig/`.

**Caching:**
- uv dependency cache is enabled in GitHub Actions through `astral-sh/setup-uv@v3` in `.github/workflows/release.yml` and `.github/workflows/tests.yml`.
- Docker layer caching is supported by copying `pyproject.toml`, `uv.lock`, and `README.md` before source files in `Dockerfile`.
- No application-level Redis, Memcached, database cache, or persistent model cache detected.

## Authentication & Identity

**Auth Provider:**
- Not detected for the Python library itself.
  - Implementation: No user identity, login, token validation, OAuth, session storage, or authorization middleware detected.

**CI/CD Auth:**
- GitHub Actions uses platform credentials for release automation.
  - GHCR publishing uses `secrets.GITHUB_TOKEN` in `.github/workflows/container-release.yml`.
  - TestPyPI publishing uses trusted publishing with `id-token: write` in `.github/workflows/release.yml`.

## Monitoring & Observability

**Error Tracking:**
- None detected.

**Logs:**
- Standard Python logging is used for optional Ewald backend selection in `src/dftorch/ewald_pme/__init__.py`.
- Some experiment and long-running workflow scripts use `logging` or `print`, such as `experiments/DDP_2.py` and `experiments/deltascf_acetone_dtscaling.py`.
- No centralized log service, metrics backend, tracing SDK, or error-reporting SaaS integration detected.

## CI/CD & Deployment

**Hosting:**
- Python package artifacts are built in `.github/workflows/release.yml`.
- TestPyPI is configured as the current Python package publishing target in `.github/workflows/release.yml`.
- GHCR is configured as the container image registry in `.github/workflows/container-release.yml`.
- No application hosting platform, web server deployment, Kubernetes manifests, Terraform, or cloud runtime configuration detected.

**CI Pipeline:**
- GitHub Actions.
  - `.github/workflows/tests.yml` runs Ruff lint, Ruff format check, and pytest on pushes and pull requests.
  - `.github/workflows/container-tests.yml` builds the Docker image and runs containerized tests on pushes and pull requests.
  - `.github/workflows/release.yml` runs tests, builds distributions, uploads artifacts, and publishes to TestPyPI on `v*` tags.
  - `.github/workflows/container-release.yml` builds, tests, saves, tags, and pushes GHCR images on `v*` tags or manual dispatch.

## Environment Configuration

**Required env vars:**
- Not detected for normal library import or standard test execution.
- `DFTORCH_ENABLE_COMPILE` is optional and enables `torch.compile` wrapping in `src/dftorch/_tools.py`.
- `TORCHDYNAMO_DISABLE`, `TORCH_COMPILE_DISABLE`, and `TORCHINDUCTOR_DISABLE` are optional PyTorch runtime controls used in tests and examples.
- `TORCH_LOGS`, `TORCHINDUCTOR_VERBOSE`, and `TORCHDYNAMO_VERBOSE` are optional PyTorch logging controls used by experiments.
- `LOCAL_RANK`, `WORLD_SIZE`, and `RANK` are required by distributed experiment code in `experiments/DDP_2.py` when launched with `torch.distributed`.
- `GITHUB_TOKEN` is provided by GitHub Actions as `secrets.GITHUB_TOKEN` for GHCR publishing in `.github/workflows/container-release.yml`.

**Secrets location:**
- No local `.env` file detected.
- No checked-in secret values detected during integration scan.
- CI secrets are referenced only by name through GitHub Actions (`secrets.GITHUB_TOKEN`), not stored in repository files.

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- None detected.

---

*Integration audit: 2026-07-17*
