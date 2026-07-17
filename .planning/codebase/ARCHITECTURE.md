<!-- refreshed: 2026-07-17 -->
# Architecture

**Analysis Date:** 2026-07-17

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                       Public Python API                      │
│                    `src/dftorch/__init__.py`                 │
├──────────────────┬──────────────────┬───────────────────────┤
│  Parameter Data  │  Geometry State  │ Simulation Drivers     │
│ `Constants.py`   │ `Structure.py`   │ `ESDriver.py`/`MD.py`  │
└────────┬─────────┴────────┬─────────┴──────────┬────────────┘
         │                  │                     │
         ▼                  ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vectorized Numerical Kernels                │
│ `_h0ands.py`, `_scf.py`, `_forces.py`, `_energy.py`,         │
│ `_coulomb_matrix.py`, `_nearestneighborlist.py`, `_stress.py`│
└────────┬────────────────────────────┬───────────────────────┘
         │                            │
         ▼                            ▼
┌──────────────────────────────┐ ┌─────────────────────────────┐
│ Optional Physics Corrections │ │ Parallel / PME Specialization│
│ `_gbsa.py`, `_dftd3.py`,     │ │ `ewald_pme/`, `sedacs/`      │
│ `_thirdorder.py`, `_spin.py` │ │                             │
└──────────────┬───────────────┘ └──────────────┬──────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│ Input/Output, SKF Parameters, Test Data, Experiment Assets    │
│ `_io.py`, `_bond_integral.py`, `tests/`, `experiments/`       │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| Public API | Defines supported imports and treats non-exported modules as internal implementation details. | `src/dftorch/__init__.py` |
| Constants | Loads element data, Slater-Koster SKF tensors, repulsive splines, Hubbard parameters, spin data, and optional DFTB3 derivatives. | `src/dftorch/Constants.py` |
| Structure | Owns single-system atom types, coordinates, periodic cell, orbital indexing, shell indexing, charge state, and differentiable coordinate/cell tensors. | `src/dftorch/Structure.py` |
| StructureBatch | Owns batched geometry, per-structure cells, padded orbital state, and batch metadata for multi-structure workflows. | `src/dftorch/Structure.py` |
| ESDriver | Orchestrates single-structure neighbor lists, Hamiltonian/overlap assembly, Coulomb treatment, SCF, corrections, forces, stress, and Hessian helpers. | `src/dftorch/ESDriver.py` |
| ESDriverBatch | Orchestrates the same electronic-structure flow for `StructureBatch`, except batched PME raises as unimplemented. | `src/dftorch/ESDriver.py` |
| MDXL / MDXLBatch | Propagates extended-Lagrangian Born-Oppenheimer MD, velocities, thermostats/barostats, charge extrapolation, and trajectory output. | `src/dftorch/MD.py` |
| GeoOpt | Runs geometry and optional cell optimization using `ESDriver` energy, force, and stress evaluations. | `src/dftorch/Optimizer.py` |
| SCF kernels | Implement closed-shell, open-shell, delta-SCF, and batched self-consistent charge loops with Anderson/DIIS style mixing. | `src/dftorch/_scf.py` |
| Hamiltonian kernels | Build `H0`, overlap `S`, derivatives, and Slater-Koster pair interpolation from neighbor-list geometry and SKF tensors. | `src/dftorch/_h0ands.py`, `src/dftorch/_slater_koster_pair.py` |
| Coulomb/PME kernels | Provide direct Coulomb matrices, Ewald/PME energy, k-space/real-space operations, and optional Triton acceleration. | `src/dftorch/_coulomb_matrix.py`, `src/dftorch/ewald_pme/` |
| Optional corrections | Add GBSA/ALPB solvation, D3(BJ), spin, and DFTB3 third-order contributions. | `src/dftorch/_gbsa.py`, `src/dftorch/_dftd3.py`, `src/dftorch/_spin.py`, `src/dftorch/_thirdorder.py` |
| SEDACS bridge | Provides distributed graph partition, global kernels, and structure preparation for large-scale SEDACS workflows. | `src/dftorch/sedacs/` |

## Pattern Overview

**Overall:** Import-first scientific Python package with public facade, stateful PyTorch containers, and private vectorized tensor kernels.

**Key Characteristics:**
- Use `src/dftorch/__init__.py` as the public API boundary; add exports there only for supported user-facing objects.
- Keep simulation state on `Structure`, `StructureBatch`, `ESDriver`, and MD/optimizer objects; kernel modules accept tensors and mutate or return tensors through the driver.
- Add numerical work as private underscore modules unless it is a user-facing container or driver.
- Preserve paired single/batch implementations when touching core physics: examples include `Structure`/`StructureBatch`, `ESDriver`/`ESDriverBatch`, `SCFx`/`SCFx_batch`, and `*_batch.py` kernels.

## Layers

**Public Facade:**
- Purpose: Stable import surface for users and tests.
- Location: `src/dftorch/__init__.py`
- Contains: Re-exports of `Constants`, `Structure`, `StructureBatch`, `ESDriver`, `ESDriverBatch`, `MDXL`, `MDXLBatch`, `MDXLOS`, `GeoOpt`, GBSA/DFTB3/D3/stress/ML helpers.
- Depends on: User-facing modules and selected helper classes/functions.
- Used by: README examples, tests, notebooks, downstream imports.

**Domain State Containers:**
- Purpose: Convert calculation dictionaries, coordinate files, SKF parameters, and device choices into tensor state.
- Location: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`
- Contains: `torch.nn.Module` containers with `torch.nn.Parameter` buffers, atom/orbital/shell index maps, cell normalization, charge/spin state, and initial density data.
- Depends on: `_io.py`, `_cell.py`, `_atomic_density_matrix.py`, `_bond_integral.py`, `_tools.py`, `_elements.py`.
- Used by: `ESDriver`, `ESDriverBatch`, `MDXL`, `MDXLBatch`, `GeoOpt`, SEDACS helpers.

**Orchestration Drivers:**
- Purpose: Sequence kernel calls into full electronic-structure, MD, and optimization workflows.
- Location: `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`
- Contains: `forward`, `calc_forces`, `calc_stress`, `calc_hessian`, `run` methods, thermostat/barostat state, history arrays, and trajectory writes.
- Depends on: Private kernel modules and domain state containers.
- Used by: User scripts, notebooks in `experiments/`, and smoke tests in `tests/test_scf.py`.

**Numerical Kernel Modules:**
- Purpose: Implement vectorized tensor algorithms for DFTB physics.
- Location: `src/dftorch/_*.py`
- Contains: Hamiltonian/overlap assembly, neighbor lists, Coulomb matrices, density matrices, SCF, energy decomposition, forces, stress, spin, third-order, GBSA, D3, and utility helpers.
- Depends on: PyTorch, NumPy/SciPy where needed, and local tensor shape conventions from `Structure`/`Constants`.
- Used by: `ESDriver.py`, `MD.py`, `Optimizer.py`, and SEDACS modules.

**Specialized Subpackages:**
- Purpose: Isolate optional or domain-specific implementations.
- Location: `src/dftorch/ewald_pme/`, `src/dftorch/sedacs/`, `src/dftorch/_legacy/`, `src/dftorch/params/`
- Contains: PME/Ewald backend selection, Triton kernels, distributed graph partitioning, legacy Hamiltonian code, and packaged parameter tables.
- Depends on: PyTorch, optional Triton/CUDA, optional `sedacs`, optional `mpi4py`.
- Used by: SCF Coulomb paths, SEDACS integration, and fallback/compatibility code.

## Data Flow

### Primary Single-Structure Energy/Force Path

1. User imports public symbols from `dftorch` (`src/dftorch/__init__.py:22`).
2. User builds `Constants(dftorch_params)`; it reads geometry to discover element pairs, loads SKF tensors with `get_skf_tensors`, and registers tensors as parameters (`src/dftorch/Constants.py:43`, `src/dftorch/Constants.py:69`, `src/dftorch/Constants.py:99`, `src/dftorch/Constants.py:142`).
3. User builds `Structure(dftorch_params, const)`; it reads XYZ/PDB input when tensors are not provided and builds atom/orbital/shell state (`src/dftorch/Structure.py:62`, `src/dftorch/Structure.py:99`).
4. User calls `ESDriver.forward(structure, const)`; the driver normalizes Coulomb settings and builds an electronic neighbor list (`src/dftorch/ESDriver.py:51`, `src/dftorch/ESDriver.py:77`, `src/dftorch/ESDriver.py:100`).
5. The driver assembles `H0`, `S`, derivatives, orthogonalizer, and repulsion terms (`src/dftorch/ESDriver.py:114`, `src/dftorch/ESDriver.py:147`, `src/dftorch/ESDriver.py:150`).
6. The driver chooses PME or direct Coulomb and builds optional DFTB3 third-order state (`src/dftorch/ESDriver.py:167`, `src/dftorch/ESDriver.py:219`, `src/dftorch/ESDriver.py:262`).
7. The driver creates optional GBSA/D3 correction objects and dispatches closed-shell or open-shell SCF (`src/dftorch/ESDriver.py:294`, `src/dftorch/ESDriver.py:297`, `src/dftorch/ESDriver.py:309`, `src/dftorch/ESDriver.py:340`).
8. SCF returns Hamiltonian, density, charges, occupations, Coulomb intermediates, and stress intermediates (`src/dftorch/_scf.py:154`, `src/dftorch/_scf.py:543`).
9. The driver computes decomposed electronic energy and adds repulsion, spin, solvation, and D3 terms to `structure.e_tot` (`src/dftorch/ESDriver.py:294`, `src/dftorch/_energy.py`).
10. User calls `ESDriver.calc_forces(structure, const)`; it chooses PME or direct force kernels and adds spin, GBSA, third-order, and D3 force contributions to `structure.f_tot` (`src/dftorch/ESDriver.py:581`, `src/dftorch/ESDriver.py:594`, `src/dftorch/ESDriver.py:630`, `src/dftorch/ESDriver.py:663`, `src/dftorch/ESDriver.py:678`, `src/dftorch/ESDriver.py:689`, `src/dftorch/ESDriver.py:696`).

### Batched Electronic-Structure Path

1. Use `StructureBatch` when `dftorch_params["FILENAME"]` is a list of files (`src/dftorch/Structure.py:350`, `src/dftorch/Structure.py:375`).
2. Use `ESDriverBatch.forward` to build batched neighbor lists, batched `H0`/`S`, batched repulsion, batched direct Coulomb, optional per-structure GBSA, and `SCFx_batch` (`src/dftorch/ESDriver.py:1150`, `src/dftorch/ESDriver.py:1198`, `src/dftorch/_h0ands.py:325`, `src/dftorch/_scf.py:954`).
3. Batched PME is not supported; the driver raises `ValueError("Batched PME Coulomb not implemented.")` in forward and force paths (`src/dftorch/ESDriver.py`, `src/dftorch/ESDriver.py:1568`).
4. Use `ESDriverBatch.calc_forces` for vectorized force assembly and batched correction gradients (`src/dftorch/ESDriver.py:1554`).

### Molecular Dynamics Flow

1. Build an already-SCF-initialized `Structure` or `StructureBatch` with corresponding driver.
2. Use `MDXL.run` for single systems or `MDXLBatch.run` for batches (`src/dftorch/MD.py:323`, `src/dftorch/MD.py:1474`).
3. The MD driver normalizes Coulomb settings, initializes velocities when missing, stores propagated charge variables, and uses kernel update helpers from `_xl_tools.py` (`src/dftorch/MD.py:356`, `src/dftorch/MD.py:361`, `src/dftorch/MD.py:370`).
4. The MD loop writes XYZ/PDB/velocity outputs through `_io.py` (`src/dftorch/_io.py:27`, `src/dftorch/_io.py:71`, `src/dftorch/_io.py:272`).

### SEDACS / Distributed Flow

1. `prepare_structure` constructs `Constants` and `Structure`, sets `dftorch_params["CELL"]`, and tags the structure for DFTorch use (`src/dftorch/sedacs/sedacs_interface.py:1116`).
2. Neighbor state and graph data are prepared with `NeighborState`, `calculate_dist_dips`, and SEDACS graph functions (`src/dftorch/sedacs/sedacs_interface.py:29`, `src/dftorch/sedacs/sedacs_interface.py:96`).
3. Distributed Krylov/kernel work happens in `kernel_global` with `torch.distributed` and PME calls (`src/dftorch/sedacs/sedacs_interface.py:693`).

**State Management:**
- Simulation state is mutable and stored on `Structure`/`StructureBatch` instances (`H0`, `S`, `D`, `q`, `e_tot`, `f_tot`, correction objects).
- Driver state is mutable and stored on `ESDriver`, `ESDriverBatch`, `MDXL`, `MDXLBatch`, and `GeoOpt` instances.
- Configuration is a plain `dict` (`dftorch_params`) passed through drivers and kernels; several methods write defaults or normalized values back into that dictionary.

## Key Abstractions

**Calculation Parameter Dictionary:**
- Purpose: Single runtime configuration object containing input paths, SKF paths, cutoffs, Coulomb method, SCF controls, spin/charge flags, correction settings, and output settings.
- Examples: `tests/test_scf.py`, `README.md`, `src/dftorch/ESDriver.py`
- Pattern: Pass the same mutable `dftorch_params` into `Constants`, `Structure`, and driver methods; normalize defaults near the driver entry point.

**Constants Database:**
- Purpose: Device-aware parameter table for element data and SKF-derived tensors.
- Examples: `src/dftorch/Constants.py`
- Pattern: Register constant tensors as `torch.nn.Parameter(..., requires_grad=False)` except selected gradient-enabled model parameters.

**Structure Containers:**
- Purpose: Canonical shape and indexing contract for downstream kernels.
- Examples: `src/dftorch/Structure.py`
- Pattern: Store atom types as `TYPE`, coordinate components as `RX`/`RY`/`RZ`, cells as normalized 3x3 tensors, and atom-to-orbital ranges as `H_INDEX_START`/`H_INDEX_END`.

**Vectorized Kernel Functions:**
- Purpose: Pure-ish tensor operations called by drivers after structure state is prepared.
- Examples: `src/dftorch/_h0ands.py`, `src/dftorch/_scf.py`, `src/dftorch/_forces.py`, `src/dftorch/_coulomb_matrix.py`
- Pattern: Accept many explicit tensors rather than whole objects, return tuple outputs, and let drivers assign outputs back onto `structure`.

**Single/Batch Pairing:**
- Purpose: Support one-system and multi-system workloads with analogous APIs.
- Examples: `Structure`/`StructureBatch`, `ESDriver`/`ESDriverBatch`, `H0_and_S_vectorized`/`H0_and_S_vectorized_batch`, `SCFx`/`SCFx_batch`, `forces_shadow`/`forces_shadow_batch`.
- Pattern: Add batch support beside the single-system path and keep names aligned with `_batch` suffixes.

## Entry Points

**Package Import:**
- Location: `src/dftorch/__init__.py`
- Triggers: `from dftorch import Constants, Structure, ESDriver, MDXL`
- Responsibilities: Expose the supported user API and hide private modules by convention.

**Single Electronic Structure:**
- Location: `src/dftorch/ESDriver.py`
- Triggers: `es_driver(structure, const, do_scf=True)` and `es_driver.calc_forces(structure, const)`
- Responsibilities: Build matrices, solve SCF, compute energies, forces, stress, and Hessian helper evaluations.

**Batched Electronic Structure:**
- Location: `src/dftorch/ESDriver.py`
- Triggers: `es_driver_batch(structure_batch, const, do_scf=True)`
- Responsibilities: Vectorized multi-structure energy/force path for direct Coulomb workflows.

**Molecular Dynamics:**
- Location: `src/dftorch/MD.py`
- Triggers: `MDXL.run(...)`, `MDXLBatch.run(...)`
- Responsibilities: Propagate coordinates, velocities, charge variables, thermostat/barostat state, and trajectory output.

**Geometry Optimization:**
- Location: `src/dftorch/Optimizer.py`
- Triggers: `GeoOpt.run(...)`
- Responsibilities: Iterate energy/force/stress evaluations and update atom positions and optional cell variables.

**SEDACS Integration:**
- Location: `src/dftorch/sedacs/sedacs_interface.py`, `src/dftorch/sedacs/SCF.py`, `src/dftorch/sedacs/MD.py`
- Triggers: SEDACS distributed workflows importing the DFTorch bridge.
- Responsibilities: Prepare DFTorch structures, graph partition data, distributed SCF/MD kernels, and PME coupling.

## Architectural Constraints

- **Threading:** Core package execution is regular Python/PyTorch tensor execution; distributed large-scale paths use `torch.distributed` in `src/dftorch/sedacs/`.
- **Global state:** `torch.set_default_dtype` is expected in user scripts/tests before constructing tensors; `_tools._maybe_compile` reads environment controls such as `DFTORCH_ENABLE_COMPILE`; `src/dftorch/ewald_pme/__init__.py` selects its backend at import time based on CUDA/Triton availability.
- **Circular imports:** `src/dftorch/ewald_pme/__init__.py` imports `PME_torch` at the end because of a circular dependency comment; keep PME additions aware of package-import order.
- **Mutable input config:** `dftorch_params` is mutated by drivers, MD, and SEDACS helpers; treat it as shared runtime state, not an immutable value object.
- **Device/dtype consistency:** New tensors must be allocated on the same device and dtype as structure or constants tensors; follow existing `device=...` and `dtype=torch.get_default_dtype()` patterns.
- **Batch limitations:** Batched PME is explicitly not implemented in `ESDriverBatch`; do not route `COUL_METHOD == "PME"` through batched force or energy paths without implementing the missing kernels.

## Anti-Patterns

### Exporting Internal Kernels as Public API

**What happens:** A new helper is imported from `src/dftorch/__init__.py` even though it is a low-level tensor kernel.
**Why it's wrong:** The package docs and tests treat `__init__.py` as a stable public contract; exporting private kernels makes implementation details harder to change.
**Do this instead:** Keep new numerical helpers in underscore modules such as `src/dftorch/_energy.py` or `src/dftorch/_forces.py`, and export only user-facing containers or explicitly supported functions from `src/dftorch/__init__.py`.

### Adding Single-System Physics Without Batch Consideration

**What happens:** A correction is added only in `ESDriver.forward` or `ESDriver.calc_forces`.
**Why it's wrong:** The codebase maintains parallel single/batch flows for many operations, and new behavior can silently diverge for `StructureBatch`.
**Do this instead:** Add the single path in `src/dftorch/ESDriver.py`, then either add the matching batch path in `ESDriverBatch` and `_batch` kernels or raise a precise `ValueError` like the existing batched PME path.

### Reading Geometry or Parameters Inside Low-Level Kernels

**What happens:** A kernel opens files or reads `dftorch_params` paths directly.
**Why it's wrong:** File IO and parameter loading are centralized in `Constants`, `Structure`, and `_io.py`; kernels should work with tensors passed by drivers.
**Do this instead:** Load files in `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, or `src/dftorch/_io.py`, then pass tensors into kernel functions such as `H0_and_S_vectorized`.

## Error Handling

**Strategy:** Raise explicit exceptions for unsupported configurations and invalid physical inputs; otherwise many numerical paths use printed warnings/timing and tensor outputs stored on state objects.

**Patterns:**
- Use `ValueError` for invalid user parameters and unsupported modes, such as invalid spin/charge combinations and batched PME (`src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`).
- Use `NotImplementedError` for known physics gaps such as full off-diagonal DFTB3 with PME (`src/dftorch/ESDriver.py:172`).
- Catch missing optional parameter files where the workflow can degrade, such as `spinw.txt` loading in `src/dftorch/Constants.py:136`.
- Tests disable Torch compile paths for deterministic CPU smoke coverage (`tests/test_scf.py`).

## Cross-Cutting Concerns

**Logging:** Mostly `print` statements and `logging` in PME backend selection; timing and progress output appears in `src/dftorch/ESDriver.py`, `src/dftorch/MD.py`, `src/dftorch/Optimizer.py`, and `src/dftorch/ewald_pme/__init__.py`.
**Validation:** Configuration validation is distributed across constructors and driver entry points; examples include spin/electron checks in `Structure`, Coulomb normalization in `_tools.py`, and unsupported-mode raises in `ESDriver`.
**Authentication:** Not applicable; this is a local scientific-computing package with no built-in auth layer.

---

*Architecture analysis: 2026-07-17*
