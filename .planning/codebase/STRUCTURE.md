# Codebase Structure

**Analysis Date:** 2026-07-17

## Directory Layout

```
DFTorch/
├── src/
│   ├── __init__.py                 # Marks `src` as importable in this repo
│   └── dftorch/                    # Main Python package
│       ├── __init__.py             # Public API exports
│       ├── Constants.py            # SKF/element parameter database
│       ├── Structure.py            # Single and batched geometry containers
│       ├── ESDriver.py             # Electronic-structure drivers
│       ├── MD.py                   # XL-BOMD drivers and velocity helpers
│       ├── Optimizer.py            # Geometry/cell optimization
│       ├── _*.py                   # Private numerical kernels and utilities
│       ├── ewald_pme/              # PME/Ewald PyTorch/Triton implementation
│       ├── sedacs/                 # SEDACS distributed integration
│       ├── _legacy/                # Legacy Hamiltonian implementation
│       └── params/                 # Packaged CSV/NPZ parameter assets
├── tests/                          # Pytest tests and small SKF/geometry fixtures
│   ├── data_skf_mio-1-1/           # Test SKF parameter set
│   └── f_orbital_data/             # Test/experimental f-orbital SKF fixtures
├── experiments/                    # Notebooks, scripts, sample inputs, generated outputs
├── docs/assets/                    # README/demo images
├── .github/workflows/              # CI, release, and container workflows
├── .planning/codebase/             # GSD codebase maps
├── pyproject.toml                  # Packaging, dependencies, pytest, ruff config
├── uv.lock                         # uv lockfile
├── Dockerfile                      # Container image definition
├── README.md                       # User-facing setup and usage docs
├── CONTRIBUTING.md                 # Contribution guidance
└── CHANGELOG.md                    # Release notes
```

## Directory Purposes

**`src/dftorch/`:**
- Purpose: Main installable package for DFTB calculations in PyTorch.
- Contains: Public facade, state containers, drivers, private numerical kernels, PME/SEDACS subpackages, packaged parameters.
- Key files: `src/dftorch/__init__.py`, `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`.

**`src/dftorch/ewald_pme/`:**
- Purpose: Particle Mesh Ewald and Ewald electrostatics implementation.
- Contains: Backend selection, PyTorch Ewald kernels, Triton kernels, PME charge-grid helpers, neighbor-list helpers, numerical utilities.
- Key files: `src/dftorch/ewald_pme/__init__.py`, `src/dftorch/ewald_pme/ewald_torch.py`, `src/dftorch/ewald_pme/ewald_triton.py`, `src/dftorch/ewald_pme/PME_torch.py`, `src/dftorch/ewald_pme/neighbor_list.py`, `src/dftorch/ewald_pme/util.py`.

**`src/dftorch/sedacs/`:**
- Purpose: Optional SEDACS integration for distributed graph-partitioned simulations.
- Contains: SEDACS SCF, SEDACS MD, and interface helpers that prepare DFTorch structures and graph data.
- Key files: `src/dftorch/sedacs/sedacs_interface.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`, `src/dftorch/sedacs/__init__.py`.

**`src/dftorch/_legacy/`:**
- Purpose: Legacy Hamiltonian/overlap implementation retained separately from active kernels.
- Contains: Older `H0andS` implementation.
- Key files: `src/dftorch/_legacy/H0andS.py`.

**`src/dftorch/params/`:**
- Purpose: Packaged built-in parameter assets.
- Contains: CSV Slater-Koster-style parameter tables and D3 reference data.
- Key files: `src/dftorch/params/*.csv`, `src/dftorch/params/dftd3_reference.npz`.

**`tests/`:**
- Purpose: Pytest coverage for import contracts, IO, neighbor lists, and CPU SCF smoke behavior.
- Contains: Test modules, small molecule geometry, SKF fixtures, f-orbital fixture data.
- Key files: `tests/test_public_api_contract.py`, `tests/test_public_api.py`, `tests/test_scf.py`, `tests/test_io.py`, `tests/test_nearestneighborlist.py`, `tests/ch4.xyz`, `tests/data_skf_mio-1-1/`.

**`experiments/`:**
- Purpose: Example notebooks, exploratory scripts, sample structures, SKF sets, and generated profiling/trajectory artifacts.
- Contains: Tutorials, delta-SCF examples, XYZ inputs, SKF parameter directories, Nsight reports, trajectory outputs.
- Key files: `experiments/1_tutorial.ipynb`, `experiments/2_tutorial_freq.ipynb`, `experiments/3_tutorial_deltaSCF.ipynb`, `experiments/DDP_2.py`, `experiments/sk_orig/`.

**`docs/assets/`:**
- Purpose: Static visual assets used by `README.md`.
- Contains: Demo GIF and PNG images.
- Key files: `docs/assets/comb_cell.gif`, `docs/assets/meth_comb.png`.

**`.github/workflows/`:**
- Purpose: GitHub Actions workflows for tests, releases, and container builds.
- Contains: YAML workflow definitions.
- Key files: `.github/workflows/tests.yml`, `.github/workflows/release.yml`, `.github/workflows/container-tests.yml`, `.github/workflows/container-release.yml`.

**`.planning/codebase/`:**
- Purpose: GSD-generated codebase reference documents.
- Contains: Architecture, structure, stack, conventions, testing, integrations, concerns maps as available.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

## Key File Locations

**Entry Points:**
- `src/dftorch/__init__.py`: Public package import surface and `__all__` contract.
- `src/dftorch/ESDriver.py`: Main energy, SCF, force, stress, and Hessian driver entry points.
- `src/dftorch/MD.py`: MD driver entry points, including `MDXL.run` and `MDXLBatch.run`.
- `src/dftorch/Optimizer.py`: Geometry optimization entry point via `GeoOpt.run`.
- `src/dftorch/sedacs/sedacs_interface.py`: SEDACS structure preparation and distributed bridge functions.

**Configuration:**
- `pyproject.toml`: Build system, package metadata, dependencies, optional dependencies, pytest config, ruff lint config.
- `uv.lock`: Locked dependency graph for uv-based development.
- `.pre-commit-config.yaml`: Pre-commit hooks.
- `.markdownlint.yaml`: Markdown lint settings.
- `.github/workflows/*.yml`: CI/release/container automation.
- `Dockerfile`: Container build instructions.

**Core Logic:**
- `src/dftorch/Constants.py`: Parameter loading and `torch.nn.Parameter` registration.
- `src/dftorch/Structure.py`: Single/batch geometry and basis-index construction.
- `src/dftorch/ESDriver.py`: Orchestration for electronic-structure calculations.
- `src/dftorch/_scf.py`: Closed-shell, open-shell, delta-SCF, and batched SCF.
- `src/dftorch/_h0ands.py`: Hamiltonian and overlap matrix assembly.
- `src/dftorch/_slater_koster_pair.py`: Slater-Koster pair interpolation.
- `src/dftorch/_coulomb_matrix.py`: Direct and Ewald-style Coulomb matrix construction.
- `src/dftorch/_coulomb_matrix_batch.py`: Batched Coulomb helpers.
- `src/dftorch/_forces.py`: Single-structure force kernels.
- `src/dftorch/_forces_batch.py`: Batched force kernels.
- `src/dftorch/_energy.py`: Electronic and shadow-energy calculations.
- `src/dftorch/_stress.py`: Analytical stress components.
- `src/dftorch/_nearestneighborlist.py`: Neighbor-list generation around DFTorch structures.
- `src/dftorch/_bond_integral.py`: SKF parsing and spline/tensor preparation.
- `src/dftorch/_io.py`: XYZ/PDB read and trajectory write helpers.
- `src/dftorch/_tools.py`: Shared math, compile, Coulomb-setting, spin/Hubbard file utilities.

**Optional Physics and ML:**
- `src/dftorch/_gbsa.py`: GBSA/ALPB implicit solvation.
- `src/dftorch/_dftd3.py`: D3(BJ) dispersion.
- `src/dftorch/_thirdorder.py`: DFTB3 third-order correction.
- `src/dftorch/_spin.py`: Spin Hamiltonian and spin energy terms.
- `src/dftorch/_ml_sk.py`: Graph neural network Slater-Koster model helpers.

**Testing:**
- `tests/test_public_api_contract.py`: Supported public symbol contract.
- `tests/test_public_api.py`: Basic public import coverage.
- `tests/test_scf.py`: CPU SCF and force smoke test using `tests/ch4.xyz` and `tests/data_skf_mio-1-1/`.
- `tests/test_io.py`: IO behavior.
- `tests/test_nearestneighborlist.py`: Neighbor-list behavior.
- `tests/data_skf_mio-1-1/`: SKF fixture data for tests.
- `tests/f_orbital_data/`: f-orbital SKF fixture data.

**Documentation and Examples:**
- `README.md`: Install, run, capabilities, public API, and minimal example.
- `CONTRIBUTING.md`: Contributor instructions.
- `CHANGELOG.md`: Release history.
- `experiments/*.ipynb`: Notebook examples and tutorials.

## Naming Conventions

**Files:**
- User-facing class modules use `PascalCase.py`: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`.
- Private implementation modules use leading-underscore snake case: `src/dftorch/_scf.py`, `src/dftorch/_h0ands.py`, `src/dftorch/_coulomb_matrix.py`, `src/dftorch/_nearestneighborlist.py`.
- Batched variants use `_batch` suffix when separated from single-system kernels: `src/dftorch/_coulomb_matrix_batch.py`, `src/dftorch/_forces_batch.py`.
- Test files use `test_*.py`: `tests/test_scf.py`, `tests/test_public_api_contract.py`.
- Parameter data uses domain-specific names and extensions: `.skf` under `tests/data_skf_mio-1-1/`, `.csv`/`.npz` under `src/dftorch/params/`.

**Directories:**
- Main package code lives under `src/dftorch/`.
- Specialized implementation subpackages use lowercase names: `src/dftorch/ewald_pme/`, `src/dftorch/sedacs/`.
- Legacy code is isolated under `src/dftorch/_legacy/`.
- Test fixtures live beside tests under `tests/data_skf_mio-1-1/` and `tests/f_orbital_data/`.
- Exploratory assets and generated research artifacts live under `experiments/`.

## Where to Add New Code

**New Public Feature:**
- Primary code: Add a user-facing class or function to an existing public module such as `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`, `src/dftorch/Structure.py`, or `src/dftorch/Constants.py`.
- Public export: Add to `src/dftorch/__init__.py` only when it is intended to be a stable supported import.
- Tests: Add import-contract coverage to `tests/test_public_api_contract.py` when adding a public export, and add behavior coverage under `tests/test_*.py`.

**New Numerical Kernel:**
- Implementation: Add a private underscore module under `src/dftorch/`, for example `src/dftorch/_new_kernel.py`, or extend the most relevant existing kernel module.
- Driver integration: Call it from `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, or `src/dftorch/Optimizer.py`, then assign returned tensors onto `structure` only at the orchestration layer.
- Tests: Add focused tests under `tests/`, using small fixtures from `tests/ch4.xyz` or `tests/data_skf_mio-1-1/` where possible.

**New Batch Support:**
- Implementation: Mirror the single-system path with `_batch` naming or an existing batch class.
- Primary code: `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_scf.py`, and any relevant `src/dftorch/*_batch.py`.
- Tests: Add or extend tests to cover `StructureBatch`/`ESDriverBatch` behavior.

**New Coulomb, PME, or Neighbor-List Work:**
- Direct Coulomb: `src/dftorch/_coulomb_matrix.py` and `src/dftorch/_coulomb_matrix_batch.py`.
- PME/Ewald: `src/dftorch/ewald_pme/`.
- Neighbor lists: `src/dftorch/_nearestneighborlist.py` for DFTorch-native lists, `src/dftorch/ewald_pme/neighbor_list.py` for PME neighbor state.
- Driver wiring: `src/dftorch/ESDriver.py` and `src/dftorch/_scf.py`.

**New Optional Physics Correction:**
- Implementation: Add a private module under `src/dftorch/_<feature>.py`.
- Driver wiring: Integrate energy setup in `src/dftorch/ESDriver.py` near existing GBSA/D3/third-order sections, and integrate force/stress contributions in `calc_forces`/`calc_stress`.
- Batch wiring: Add a batch class or batch helper when the correction applies to `StructureBatch`, following `GBSABatch` and `ThirdOrderBatch` patterns.

**New SEDACS Integration Code:**
- Implementation: `src/dftorch/sedacs/`.
- Shared DFTorch kernel usage: Import from `dftorch` public objects or specific internal kernels only when SEDACS needs a low-level tensor operation.
- Tests: Prefer isolated helper tests unless a distributed runtime is required.

**New IO Format:**
- Implementation: `src/dftorch/_io.py`.
- Structure integration: Call the reader from `src/dftorch/Structure.py` and `src/dftorch/Constants.py` if the format provides geometry/species.
- Tests: Add `tests/test_io.py` coverage with a minimal fixture under `tests/`.

**New Documentation or Examples:**
- Docs: `README.md`, `CONTRIBUTING.md`, or files under `docs/`.
- Examples: `experiments/` for notebooks, exploratory scripts, and non-test research assets.
- Testable examples: Prefer adding minimal deterministic coverage under `tests/` rather than relying on notebooks.

**Utilities:**
- Shared math/device/config helpers: `src/dftorch/_tools.py`.
- Cell geometry helpers: `src/dftorch/_cell.py`.
- Element lookup constants: `src/dftorch/_elements.py`.
- Keep utility functions private unless they are intentionally part of the stable API.

## Special Directories

**`src/dftorch/params/`:**
- Purpose: Packaged model/reference data used by the library.
- Generated: No.
- Committed: Yes.

**`tests/data_skf_mio-1-1/`:**
- Purpose: Small SKF parameter set for tests.
- Generated: No.
- Committed: Yes.

**`tests/f_orbital_data/`:**
- Purpose: f-orbital SKF fixtures.
- Generated: No.
- Committed: Yes.

**`experiments/`:**
- Purpose: Notebooks, sample systems, SKF parameter directories, profiling reports, and trajectory outputs.
- Generated: Mixed; contains both hand-authored examples and generated outputs such as trajectory/profiling files.
- Committed: Yes.

**`build/`:**
- Purpose: Local setuptools build output.
- Generated: Yes.
- Committed: No expected source role; do not add new source files here.

**`.venv/`:**
- Purpose: Local Python virtual environment.
- Generated: Yes.
- Committed: No.

**`src/dftorch.egg-info/`:**
- Purpose: Local package metadata generated by editable/build operations.
- Generated: Yes.
- Committed: No expected source role.

**`__pycache__/`:**
- Purpose: Python bytecode cache.
- Generated: Yes.
- Committed: No.

---

*Structure analysis: 2026-07-17*
