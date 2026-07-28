# Technology Stack

**Analysis Date:** 2026-07-20
**Last Mapped Commit:** `e824543a0b411dcf52462ee55db5362c360e7780`
**Scope:** `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`, `tests/f_orbital_data`

## Languages

**Primary:**
- Python - f-orbital loader, constants, structure indexing, electronic-structure driver, and validation script in `src/dftorch/_bond_integral.py`, `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, and `src/dftorch/script.py`.

**Secondary:**
- DFTB Slater-Koster text data (`.skf`) - f-orbital parameter fixtures in `tests/f_orbital_data/Eu-Eu.skf`, `tests/f_orbital_data/Eu-Ga.skf`, `tests/f_orbital_data/Eu-N.skf`, `tests/f_orbital_data/Ga-Eu.skf`, `tests/f_orbital_data/Ga-Ga.skf`, `tests/f_orbital_data/Ga-N.skf`, `tests/f_orbital_data/N-Eu.skf`, `tests/f_orbital_data/N-Ga.skf`, and `tests/f_orbital_data/N-N.skf`.

## Runtime

**Environment:**
- CPython runtime. The scoped files use standard Python modules `os`, `re`, `math`, `time`, `sys`, `types`, `pathlib.Path`, `importlib.util`, and `typing`.
- PyTorch runtime. The scoped implementation stores constants and structure metadata as tensors and `torch.nn.Module` subclasses in `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, and `src/dftorch/ESDriver.py`.
- Device-aware tensor execution. `src/dftorch/_bond_integral.py` accepts `device` and `dtype` for SKF parsing and spline construction, `src/dftorch/Structure.py` moves species/coordinates to a requested device, and `src/dftorch/ESDriver.py` runs single and batched calculations on `self.device`.

**Package Manager:**
- Not detected in scoped paths.
- Lockfile: Not applicable to scoped incremental scan.

## Frameworks

**Core:**
- PyTorch - Core tensor, autograd, linear algebra, module, and device framework used by all scoped runtime modules.
  - `src/dftorch/Constants.py` registers Slater-Koster tables, shell metadata, onsite energies, Hubbard values, and pair lookup tables as `torch.nn.Parameter`.
  - `src/dftorch/Structure.py` builds atom-major AO indices, shell starts/ends, density vectors, periodic cell data, and batched structure tensors.
  - `src/dftorch/_bond_integral.py` parses SKF files into tensors and builds cubic spline coefficients with `torch.linalg.solve`.
  - `src/dftorch/ESDriver.py` assembles Hamiltonian/overlap matrices, Coulomb matrices, SCC state, corrections, energies, and forces.
- NumPy - Used in `src/dftorch/Constants.py` for the debugging `ConstantsTest` element table and in `src/dftorch/ESDriver.py` indirectly through D3 setup via `.cpu().numpy()` before `create_dftd3()`.

**Testing:**
- Standalone Python validation script - `src/dftorch/script.py` validates f-orbital SKF parsing, metadata propagation through `Constants`, and AO layout in `Structure`/`StructureBatch`.
- PyTorch assertions - `src/dftorch/script.py` uses tensor comparisons and explicit `AssertionError` failures rather than a test-runner-specific API.

**Build/Dev:**
- Not detected in scoped paths.
- `src/dftorch/script.py` can be run directly from the project root with `python src/dftorch/script.py tests/f_orbital_data` or without an argument to default to `tests/f_orbital_data`.

## Key Dependencies

**Critical:**
- `torch` - Required for all f-orbital tensor data structures, spline construction, device movement, `torch.nn.Module` classes, SCC driver execution, and validation checks in `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, and `src/dftorch/script.py`.
- `numpy` - Required by the scoped constants/debug path in `src/dftorch/Constants.py`; `src/dftorch/ESDriver.py` also converts batched atomic numbers to NumPy for `create_dftd3()`.

**Infrastructure:**
- Local DFTorch modules provide the scoped physics stack:
  - `src/dftorch/_bond_integral.py` depends on `src/dftorch/_tools.py` for `ordered_pairs_from_TYPE`.
  - `src/dftorch/Constants.py` depends on `src/dftorch/_elements.py`, `src/dftorch/_io.py`, `src/dftorch/_tools.py`, and `src/dftorch/_bond_integral.py`.
  - `src/dftorch/Structure.py` depends on `src/dftorch/_cell.py` and `src/dftorch/_io.py`.
  - `src/dftorch/ESDriver.py` depends on local modules for Coulomb matrices, DFT-D3 correction, energies, forces, GBSA solvation, Hamiltonian/overlap assembly, neighbor lists, repulsive splines, SCF, stress, third-order DFTB, spin, and tensor utilities.

## Configuration

**Environment:**
- Runtime simulation configuration is dictionary-based.
- `SKFPATH` is required by `src/dftorch/Constants.py` and points to a directory containing `.skf` Slater-Koster files.
- `FILENAME` is required by `src/dftorch/Constants.py` and used by `src/dftorch/Structure.py`; it may be a single `.xyz`/`.pdb` path or a list of paths.
- `DFTB3`, `MAGNETIC_HUBBARD_LDEP`, and `GRAD_PARAM` configure constants loading in `src/dftorch/Constants.py`.
- `GRAD_XYZ` and `GRAD_CELL` configure differentiability of coordinates/cell state in `src/dftorch/Structure.py`.
- `RCUT_ELECTRONIC`, `RCUT_REPULSIVE`, `COUL_METHOD`, `COULOMB_CUTOFF`, `COULOMB_ACC`, `SCF_ALPHA`, `H_DAMP_EXP`, `H5_PARAMS`, `dftb3_diagonal_only`, `SOLVENT_PARAM_FILE`, `SOLVATION_MODEL`, `GBSA_DIFFERENTIABLE`, and `D3_PARAMS` configure the single and batched drivers in `src/dftorch/ESDriver.py`.

**Build:**
- Not detected in scoped paths.
- `src/dftorch/script.py` dynamically loads `src/dftorch/_bond_integral.py` through `importlib.util.spec_from_file_location()` and installs a fake `dftorch` package in `sys.modules` so validation avoids importing `src/dftorch/__init__.py`.

## Platform Requirements

**Development:**
- Python interpreter capable of running the scoped type-hint syntax, including `from __future__ import annotations`, `dict[str, Any]`, and `torch.device | None`.
- PyTorch installed and importable.
- Local f-orbital Slater-Koster fixtures under `tests/f_orbital_data/` for the validation script.
- Input structures must be readable by local `read_xyz()` or `read_pdb()` calls invoked from `src/dftorch/Constants.py` and `src/dftorch/Structure.py`.

**Production:**
- Library/runtime deployment is local Python package execution; no web server or hosted runtime is detected in the scoped paths.
- Production f-orbital calculations require a complete local `SKFPATH` directory containing pairwise `.skf` files for every ordered species pair used by `src/dftorch/_bond_integral.py`.
- Optional local companion files in `SKFPATH` are supported by `src/dftorch/Constants.py`: `spinw.txt` for spin-orbit/magnetic Hubbard data and `hubbard_derivative.txt` for DFTB3 Hubbard derivatives.

---

*Stack analysis: 2026-07-20*
