# DFTorch

DFTorch is a Density Functional Tight Binding (DFTB) implementation in PyTorch.

## Installation

### uv (recommended)

```bash
cd DFTorch
uv venv --python 3.11
uv pip install .
```

or for compatibility with sedacs

```bash
uv pip install -e ".[sedacs]"
```

Run tests:
```bash
uv run pytest
```

### pip
To install DFTorch, run:

```bash
pip install .
```


## Requirements
- torch
- numpy
- scipy
- pandas

## Usage

See `experiments/1_tutorial.ipynb` for examples.

## Main Capabilities

- SCC-DFTB calculations in PyTorch for single structures and batched multi-structure workloads.
- DFTB3 support, including diagonal-only and full third-order charge corrections.
- Restricted and unrestricted/open-shell electronic structure calculations with finite-temperature occupations.
- Non-periodic and periodic simulations with full Coulomb summation and Particle Mesh Ewald (PME) electrostatics.
- [NVIDIA ALCHEMI](https://github.com/NVIDIA/nvalchemi-toolkit-ops) neighbor-list generation backend for accelerated neighbor-list construction.
- Analytical and autograd forces and stress tensors.
- Delta-SCF excited-state calculations for targeted non-Aufbau electronic excitations.
- Extended-Lagrangian Born-Oppenheimer molecular dynamics ([XL-BOMD](https://link.springer.com/article/10.1140/epjb/s10051-021-00151-6)), including batched MD drivers.
- NVT and NPT molecular dynamics with Langevin thermostat and Berendsen barostat.
- Geometry optimization for atomic positions and periodic cells.
- Implicit solvation ([GBSA/ALPB](https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00471)).
- D3(BJ) dispersion.
- Hydrogen bond damping ([$γ^h$](https://pubs.acs.org/doi/10.1021/ct100684s) and [H5](https://pubs.acs.org/doi/10.1021/acs.jctc.7b00629)).
- Automatic differentiation with respect to coordinates and selected model parameters for backpropagation workflows.
- GPU acceleration through PyTorch, with optional compile-time optimization and differentiable tensor workflows.

### Public API

Supported public imports:

```python
from dftorch import (
    Constants,
    Structure,
    StructureBatch,
    ESDriver,
    ESDriverBatch,
    MDXL,
    MDXLBatch,
)
```

All other modules are internal implementation details and may change.

### Methane Combustion Demo
- DFTB2, mio-1-1
- 100 $CH_4$ + 200 $O_2$.
- Langevin thermostat at T = 3200 K.
- 0.05 ns, 200,000 step, Δt=0.25 fs.
- 0.3 s wall time per MD step on  NVIDIA A100 GPU.

<p align="left">
  <img src="docs/assets/comb_cell.gif" alt="DFTorch demo" width="400">
</p>

<p align="left">
  <img src="docs/assets/meth_comb.png" alt="DFTorch demo" width="400">
</p>

## Authors
 M. Kulichenko, A.P. Baldo, A.M.N. Niklasson
