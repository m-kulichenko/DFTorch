# Changelog

All notable changes to DFTorch are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Periodic MD tutorial with PME electrostatics
- Batched MD tutorial
- Open-shell (spin-polarized) SCC-DFTB support
- DFTB3 third-order corrections
- Backpropagation through SCF for machine-learning workflows
- Optional ALCHEMI neighbor-list backend via `nvalchemi-toolkit-ops`

## [0.1.5] - 2025

### Added

- Ewald/PME electrostatics for periodic boundary conditions
- GBSA implicit solvation
- D3 dispersion corrections
- Spin-polarized SCF (`_spin.py`)
- DFTB3 (`_thirdorder.py`)
- Batched Coulomb matrix for GPU-parallel simulations
- XL-BOMD extended Lagrangian Born-Oppenheimer MD
- SEDACS interface (`src/dftorch/sedacs/`)
- Docker support (`Dockerfile`)
- GitHub Actions CI (`tests.yml`, `container-tests.yml`, `release.yml`)
- Pre-commit hooks (ruff format + ruff check + pytest)

## [0.1.0] - initial release

### Added

- Core SCC-DFTB implementation in PyTorch
- Slater-Koster integrals (`_slater_koster_pair.py`, `_bond_integral.py`)
- SCF solver (`_scf.py`)
- Forces and stress tensors (`_forces.py`, `_stress.py`)
- Nearest-neighbor list (`_nearestneighborlist.py`)
- Structure I/O (`_io.py`, `Structure.py`)
- MD integrator (`MD.py`)
- Geometry optimizer (`Optimizer.py`)
- MIO-1-1 parameters bundled in `params/`

[Unreleased]: https://github.com/m-kulichenko/DFTorch/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/m-kulichenko/DFTorch/compare/v0.1.0...v0.1.5
[0.1.0]: https://github.com/m-kulichenko/DFTorch/releases/tag/v0.1.0
