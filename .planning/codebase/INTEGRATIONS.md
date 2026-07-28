# External Integrations

**Analysis Date:** 2026-07-20
**Last Mapped Commit:** `e824543a0b411dcf52462ee55db5362c360e7780`
**Scope:** `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`, `tests/f_orbital_data`

## APIs & External Services

**Scientific compute libraries:**
- PyTorch - Local tensor, autograd, module, device, and linear algebra API used for all scoped f-orbital runtime and validation code.
  - SDK/Client: `torch`
  - Auth: Not applicable
  - Code paths: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`
- NumPy - Local array API used for constants/debug data and D3 handoff.
  - SDK/Client: `numpy`
  - Auth: Not applicable
  - Code paths: `src/dftorch/Constants.py`, `src/dftorch/ESDriver.py`

**Application APIs:**
- HTTP clients, REST APIs, GraphQL APIs, cloud SDKs, payment APIs, and email/SMS APIs: Not detected in scoped paths.
  - SDK/Client: Not detected
  - Auth: Not detected

## Data Storage

**Databases:**
- Not detected in scoped paths.
  - Connection: Not applicable
  - Client: Not applicable

**File Storage:**
- Local filesystem only.
  - Slater-Koster `.skf` files are loaded from the `SKFPATH` directory by `src/dftorch/Constants.py` and `src/dftorch/_bond_integral.py`.
  - `src/dftorch/_bond_integral.py` resolves both dashed filenames like `Eu-N.skf` and compact filenames like `EuN.skf`.
  - `src/dftorch/_bond_integral.py` reads `.skf` files with `Path(path).read_text(errors="ignore")`, parses electronic tables, normalizes 20-column and 40-column rows, parses repulsive spline blocks, and returns tensor data.
  - Homonuclear `.skf` files provide element-level metadata consumed by `src/dftorch/_bond_integral.py`: `N_ORB`, `MAX_ANG`, `MAX_ANG_OCC`, `TORE`, `N_S`, `N_P`, `N_D`, `N_F`, `ES`, `EP`, `ED`, `EF`, `US`, `UP`, `UD`, `UF`, and `SHELL_PRESENT`.
  - `tests/f_orbital_data/` provides scoped f-orbital fixture files for Eu/Ga/N pairs: `Eu-Eu.skf`, `Eu-Ga.skf`, `Eu-N.skf`, `Ga-Eu.skf`, `Ga-Ga.skf`, `Ga-N.skf`, `N-Eu.skf`, `N-Ga.skf`, and `N-N.skf`.
  - Input coordinate files are read through local `.xyz`/`.pdb` parsers called by `src/dftorch/Constants.py` and `src/dftorch/Structure.py` from `FILENAME`.
  - `src/dftorch/script.py` writes synthetic `.xyz` files with `Path.write_text()` during validation, then uses those files to instantiate `Constants`, `Structure`, and `StructureBatch`.
  - Optional `spinw.txt` in `SKFPATH` is loaded by `src/dftorch/Constants.py` through `load_spinw_to_matrix()`.
  - Optional `hubbard_derivative.txt` in `SKFPATH` is loaded by `src/dftorch/Constants.py` through `load_hubbard_derivs()` when `DFTB3` is enabled.
  - Optional `wfc.hsd` support exists in `src/dftorch/_bond_integral.py` through `read_wfc_hsd()`, which reads text with `Path(path).read_text(errors="ignore")` and can override shell metadata.
  - Optional GBSA parameter file integration is exposed by `src/dftorch/ESDriver.py` through `SOLVENT_PARAM_FILE`, passed to `create_gbsa()`.

**Caching:**
- No database cache, Redis, Memcached, or persistent application cache detected in scoped paths.
- `src/dftorch/ESDriver.py` supports reuse of a precomputed `GBSABatch` via `gbsa_batch_precomputed` in `ESDriverBatch.forward()`.

## Authentication & Identity

**Auth Provider:**
- Not detected in scoped paths.
  - Implementation: No login, token validation, OAuth, session storage, or authorization middleware appears in the scoped files.

## Monitoring & Observability

**Error Tracking:**
- None detected in scoped paths.

**Logs:**
- `src/dftorch/Constants.py` prints a warning when `spinw.txt` cannot be loaded and prints final DFTB3 status.
- `src/dftorch/ESDriver.py` prints timing information for selected GBSA initialization and gradient paths when those branches execute.
- `src/dftorch/script.py` prints validation summaries and failure details for f-orbital checks.
- No centralized log service, metrics backend, tracing SDK, or error-reporting SaaS integration is detected in scoped paths.

## CI/CD & Deployment

**Hosting:**
- Not detected in scoped paths.

**CI Pipeline:**
- Not detected in scoped paths.

## Environment Configuration

**Required env vars:**
- None detected in scoped paths.

**Runtime parameters:**
- `SKFPATH` is required by `src/dftorch/Constants.py` and points to local Slater-Koster data.
- `FILENAME` is required for file-backed constants/structure construction in `src/dftorch/Constants.py` and `src/dftorch/Structure.py`.
- `DFTB3` enables optional `hubbard_derivative.txt` loading in `src/dftorch/Constants.py`.
- `MAGNETIC_HUBBARD_LDEP` selects shell-dependent magnetic Hubbard behavior in `src/dftorch/Constants.py`.
- `GRAD_PARAM`, `GRAD_XYZ`, and `GRAD_CELL` control differentiable parameter/coordinate/cell tensors in `src/dftorch/Constants.py` and `src/dftorch/Structure.py`.
- `COUL_METHOD`, `COULOMB_CUTOFF`, `COULOMB_ACC`, `SCF_ALPHA`, `RCUT_ELECTRONIC`, and `RCUT_REPULSIVE` configure driver paths in `src/dftorch/ESDriver.py`.
- `SOLVENT_PARAM_FILE`, `SOLVATION_MODEL`, and `GBSA_DIFFERENTIABLE` configure optional local GBSA solvation integration in `src/dftorch/ESDriver.py`.
- `D3_PARAMS` configures optional D3(BJ) dispersion correction in `src/dftorch/ESDriver.py`.

**Secrets location:**
- Not applicable. No secret-bearing files were read, and no scoped code references secret environment variables.

## Webhooks & Callbacks

**Incoming:**
- None detected in scoped paths.

**Outgoing:**
- None detected in scoped paths.

---

*Integration audit: 2026-07-20*
