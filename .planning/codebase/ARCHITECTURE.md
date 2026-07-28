<!-- refreshed: 2026-07-20 -->
<!-- last_mapped_commit: e824543a0b411dcf52462ee55db5362c360e7780 -->
# Architecture

**Analysis Date:** 2026-07-20

## System Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                 Scoped f-Orbital Validation Entry            │
│                  `src/dftorch/script.py`                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               SKF Parsing and Parameter Assembly             │
│                 `src/dftorch/_bond_integral.py`              │
├───────────────────────┬─────────────────────────────────────┤
│  40 SK channels       │  element shell metadata              │
│  `_CHANNELS`          │  `N_F`, `EF`, `UF`, `SHELL_PRESENT`  │
└───────────┬───────────┴──────────────────────┬──────────────┘
            │                                  │
            ▼                                  ▼
┌─────────────────────────────┐    ┌──────────────────────────┐
│ Constants Tensor Registry   │    │ f-Orbital SKF Fixtures    │
│ `src/dftorch/Constants.py`  │    │ `tests/f_orbital_data/`   │
└──────────────┬──────────────┘    └──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│           Structure / StructureBatch AO Bookkeeping          │
│                  `src/dftorch/Structure.py`                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Electronic Structure Driver Consumers           │
│                   `src/dftorch/ESDriver.py`                  │
└─────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| f-orbital SKF fixture set | Provides ordered Eu/Ga/N `.skf` pair files, including homonuclear headers used to derive element-level f-shell metadata. | `tests/f_orbital_data/` |
| SKF channel map | Defines the 40-channel extended Hamiltonian/overlap order with f-f, d-f, p-f, and s-f channels before legacy s/p/d channels. | `src/dftorch/_bond_integral.py:101` |
| SKF row normalizer | Accepts either 20-column legacy rows or 40-column extended rows and emits the canonical 40-column `_CHANNELS` order. | `src/dftorch/_bond_integral.py:397` |
| SKF path resolver | Allows pair labels such as `Eu-N` to load dashed files like `Eu-N.skf` or compact files like `EuN.skf`. | `src/dftorch/_bond_integral.py:433` |
| SKF metadata parser | Reads homonuclear headers, infers s/p/d/f shell presence, validates nested shells, fills `N_F`, `EF`, `UF`, and `SHELL_PRESENT`. | `src/dftorch/_bond_integral.py:547` |
| WFC override parser | Uses optional `wfc.hsd` shell declarations as authoritative basis-presence metadata while preserving occupation values from SKF headers. | `src/dftorch/_bond_integral.py:869` |
| SKF tensor assembler | Loads all ordered element-pair SKF files for active species, builds spline coefficients, repulsive splines, pair tensors, and element metadata tensors. | `src/dftorch/_bond_integral.py:947` |
| Constants | Registers f-shell SKF outputs as device-aware `torch.nn.Parameter` tables used by structures and drivers. | `src/dftorch/Constants.py:45` |
| Structure | Converts a single species/coordinate input into atom-major AO ranges, shell ranges, onsite energies, labels, Hubbard data, and `D0`. | `src/dftorch/Structure.py:214` |
| StructureBatch | Mirrors `Structure` for multiple structures with padded per-structure AO rows and global flattened AO/shell offsets. | `src/dftorch/Structure.py:461` |
| ESDriver | Consumes `Structure` f-orbital metadata through Hamiltonian assembly and SCF arguments. | `src/dftorch/ESDriver.py:27` |
| ESDriverBatch | Consumes `StructureBatch` f-orbital metadata through batched Hamiltonian assembly and batched SCF arguments. | `src/dftorch/ESDriver.py:1126` |
| Validation script | Verifies `_bond_integral.py`, `Constants.py`, and `Structure.py` against `tests/f_orbital_data` without importing the full public package. | `src/dftorch/script.py:1` |

## Pattern Overview

**Overall:** Data-driven PyTorch scientific pipeline where SKF fixture files define pair and element metadata, `Constants` registers those tensors, `Structure` materializes atom-major AO/shell indexing, and `ESDriver` consumes the resulting tensor contract.

**Key Characteristics:**
- Treat `_bond_integral.py` as the boundary between text SKF data and tensor-ready parameter tables.
- Preserve a single canonical extended 40-channel Slater-Koster order for all loaded SKF rows; legacy 20-column rows must be expanded before any spline or driver code sees them.
- Represent basis presence explicitly with `SHELL_PRESENT[..., 4]` instead of deriving all AO layout from `max_ang`.
- Keep AO ordering nested and atom-major: `s`, three `p`, five `d`, seven `f` positions.
- Add f-shell fields in parallel with existing s/p/d fields: `N_F`, `EF`, `UF`, `const.n_f`, `const.Ef`, `const.Uf`, `Structure.has_f`.
- Keep single and batched structure bookkeeping aligned; any single-structure shell-index change needs corresponding `StructureBatch` handling.

## Layers

**Fixture Data Layer:**
- Purpose: Provide DFTB+ SKF source files for f-orbital validation.
- Location: `tests/f_orbital_data/`
- Contains: `Eu-Eu.skf`, `Eu-Ga.skf`, `Eu-N.skf`, `Ga-Eu.skf`, `Ga-Ga.skf`, `Ga-N.skf`, `N-Eu.skf`, `N-Ga.skf`, `N-N.skf`.
- Depends on: DFTB+ SKF text conventions, including optional `@` extended headers.
- Used by: `src/dftorch/script.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/Constants.py`.

**SKF Parser Layer:**
- Purpose: Convert raw `.skf` files into normalized electronic channels, spline inputs, repulsive spline data, and atomic basis metadata.
- Location: `src/dftorch/_bond_integral.py`
- Contains: `_CHANNELS`, `_SIMPLE_CHANNELS`, `_normalize_skf_row`, `_split_skf_pair_name`, `_validate_nested_shells`, `read_skf_table`, `read_wfc_hsd`, `get_skf_tensors`.
- Depends on: `torch`, `pathlib.Path`, `ordered_pairs_from_TYPE`.
- Used by: `src/dftorch/Constants.py` and validation helpers in `src/dftorch/script.py`.

**Constants Registry Layer:**
- Purpose: Hold parsed SKF tensors and element metadata on the calculation device.
- Location: `src/dftorch/Constants.py`
- Contains: `Constants` and `ConstantsTest`; `Constants` registers `coeffs_tensor`, `R_tensor`, `pair_lookup`, `n_orb`, `max_ang`, `max_ang_occ`, `n_f`, `Ef`, `Uf`, and `shell_present`.
- Depends on: `_bond_integral.get_skf_tensors`, `_io.read_xyz`, `_io.read_pdb`, `_tools.ordered_pairs_from_TYPE`, element tables.
- Used by: `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, and `src/dftorch/script.py`.

**Structure Metadata Layer:**
- Purpose: Convert species and constants into the AO/shell shape contract consumed by matrix assembly and SCF.
- Location: `src/dftorch/Structure.py`
- Contains: `SHELL_DIMS`, `SHELL_LOCAL_STARTS`, `AO_LABEL_TEMPLATE`, `_ao_mask_from_shell_present`, `_shell_local_start`, `_global_shell_start`, `_atomic_density_matrix_from_shells`, `Structure`, `StructureBatch`.
- Depends on: `const.n_orb`, `const.shell_present`, `const.n_f`, `const.Ef`, `const.Uf`, `const.shell_dim`.
- Used by: `src/dftorch/ESDriver.py` and validation helpers in `src/dftorch/script.py`.

**Driver Consumer Layer:**
- Purpose: Consume the f-orbital-aware structure contract during Hamiltonian assembly and SCF.
- Location: `src/dftorch/ESDriver.py`
- Contains: `ESDriver.forward`, `ESDriverBatch.forward`, SCF dispatch calls that pass `n_orbitals_per_atom`, `D0`, `el_per_shell`, `shell_types`, and `n_shells_per_atom`.
- Depends on: `Structure`/`StructureBatch` attributes and lower-level Hamiltonian/SCF kernels outside this scoped remap.
- Used by: Runtime calculations once structures are built.

**Scoped Validation Layer:**
- Purpose: Independently check parser, constants, and structure assumptions using synthetic XYZ inputs and the f-orbital SKF directory.
- Location: `src/dftorch/script.py`
- Contains: project-root discovery, direct module loading, expected metadata parsing, spline reconstruction checks, constants checks, single/batch structure layout checks.
- Depends on: `tests/f_orbital_data/`, `_bond_integral.py`, `Constants.py`, `Structure.py`.
- Used by: Manual f-orbital validation from project root.

## Data Flow

### f-Orbital Parameter Loading Path

1. `Constants.__init__` reads species from `dftorch_params["FILENAME"]` using XYZ/PDB helpers (`src/dftorch/Constants.py:71`).
2. Species are flattened into `TYPE`; ordered element pairs and a dense atomic-number pair lookup are built (`src/dftorch/Constants.py:88`).
3. `Constants` calls `get_skf_tensors(TYPE, self.skfpath)` to load all ordered pair parameters from `SKFPATH` (`src/dftorch/Constants.py:101`).
4. `get_skf_tensors` derives the ordered label list, allocates pair tensors, allocates 120-element metadata tables, and loops over every label (`src/dftorch/_bond_integral.py:980`).
5. Each label resolves to a dashed or compact `.skf` path with `_resolve_skf_path` (`src/dftorch/_bond_integral.py:1021`).
6. `read_skf_table` parses the SKF file, normalizes electronic rows to 40 channels, builds channel tensors, reads repulsive spline data, and updates homonuclear metadata in-place (`src/dftorch/_bond_integral.py:547`).
7. `channels_to_matrix` and `cubic_spline_coeffs` convert channels into `(npts, 40)` values and `(npts - 1, 40, 4)` cubic spline coefficients (`src/dftorch/_bond_integral.py:771`, `src/dftorch/_bond_integral.py:779`).
8. If `wfc.hsd` exists under `SKFPATH`, `read_wfc_hsd` overrides shell-presence, `N_ORB`, `MAX_ANG`, and `MAX_ANG_OCC` (`src/dftorch/_bond_integral.py:1060`).
9. `get_skf_tensors` returns pair tensors plus element metadata through the extended tuple ending in `N_F`, `EF`, `UF`, and `SHELL_PRESENT` (`src/dftorch/_bond_integral.py:1083`).
10. `Constants` registers returned tensors as `torch.nn.Parameter` objects, including `n_f`, `Uf`, `Ef`, and `shell_present` (`src/dftorch/Constants.py:160`).

### Single-Structure AO and Shell Path

1. `Structure.__init__` reads or accepts species/coordinates and normalizes them to batched tensor shapes (`src/dftorch/Structure.py:242`, `src/dftorch/Structure.py:259`).
2. The single structure selects `species[0]` as `TYPE`, prepares coordinate tensors, wraps positions when a cell is present, and stores `Nats` (`src/dftorch/Structure.py:262`, `src/dftorch/Structure.py:285`, `src/dftorch/Structure.py:317`).
3. Atom AO ranges are computed from `const.n_orb[self.TYPE]` into `H_INDEX_START` and `H_INDEX_END` (`src/dftorch/Structure.py:334`).
4. `const.shell_present[self.TYPE]` becomes the per-atom four-shell boolean matrix; `has_f` is the fourth column (`src/dftorch/Structure.py:363`).
5. Local shell starts use fixed offsets `(0, 1, 4, 9)`, then global shell starts add each atom's AO start (`src/dftorch/Structure.py:369`).
6. Onsite AO energies are expanded into a fixed 16-position template and masked by shell presence to build `diagonal` and `HDIM` (`src/dftorch/Structure.py:378`).
7. AO shell type IDs and labels are flattened in atom-major order from the same mask (`src/dftorch/Structure.py:410`).
8. Shell-resolved Hubbard values, shell type IDs, electron counts, and Hubbard shell ranges are built from `Us/Up/Ud/Uf` and `n_s/n_p/n_d/n_f` (`src/dftorch/Structure.py:413`).
9. `D0` is filled per shell by distributing shell occupations across shell dimensions and multiplying by the closed-shell `0.5` factor (`src/dftorch/Structure.py:443`).

### Batched Structure AO and Shell Path

1. `StructureBatch.__init__` reads all files listed in `dftorch_params["FILENAME"]`, normalizes species to `(B, N)` and coordinates to `(B, N, 3)` (`src/dftorch/Structure.py:476`).
2. Per-atom AO starts and ends are computed per batch row from `const.n_orb[self.TYPE]` (`src/dftorch/Structure.py:532`).
3. Per-atom shell presence, `has_f`, shell starts, and shell ends mirror the single-structure path with an added batch dimension (`src/dftorch/Structure.py:551`).
4. `diagonal_flat` is created from the same 16-position AO template, then copied into padded `diagonal[batch_idx, :hdim_b]` rows (`src/dftorch/Structure.py:571`, `src/dftorch/Structure.py:602`).
5. Per-structure AO offsets produce `H_INDEX_START_GLOBAL`, `H_INDEX_END_GLOBAL`, `shell_ao_start_global`, and `shell_ao_end_global` for flattened global operations (`src/dftorch/Structure.py:620`).
6. Shell-resolved Hubbard values, shell type IDs, electron counts, and global shell offsets are built with batched templates (`src/dftorch/Structure.py:630`).
7. Batched `D0` is filled row-wise by shell occupation and padded to the maximum `HDIM` across the batch (`src/dftorch/Structure.py:667`).

### Driver Consumption Path

1. `ESDriver.forward` builds the electronic neighbor list, then passes `structure.diagonal`, `structure.H_INDEX_START`, `const.R_orb`, and `const.coeffs_tensor` into `H0_and_S_vectorized` (`src/dftorch/ESDriver.py:87`, `src/dftorch/ESDriver.py:114`).
2. Open-shell SCF receives shell-resolved arrays `el_per_shell`, `shell_types`, `n_shells_per_atom`, and `const.shell_dim` (`src/dftorch/ESDriver.py:340`).
3. Closed-shell SCF receives `n_orbitals_per_atom`, `Hubbard_U`, `dU_dq`, and shell-derived `D0` (`src/dftorch/ESDriver.py:483`).
4. Energy evaluation uses `D0` and SCF outputs without re-deriving shell layout (`src/dftorch/ESDriver.py:517`).
5. `ESDriverBatch.forward` mirrors matrix assembly with `H0_and_S_vectorized_batch`, `structure.diagonal`, and `structure.H_INDEX_START` (`src/dftorch/ESDriver.py:1227`).
6. Batched SCF consumes `structure.n_orbitals_per_atom`, `structure.Hubbard_U`, `structure.dU_dq`, and `structure.D0` (`src/dftorch/ESDriver.py:1454`).

### Validation Script Flow

1. Running `python src/dftorch/script.py tests/f_orbital_data` loads `_bond_integral.py` directly under a fake `dftorch` package to avoid unrelated package imports (`src/dftorch/script.py:72`).
2. `_bond_integral.py` checks read every `.skf`, verify 40 channels, reconstruct spline values at original grid points, and compare homonuclear metadata (`src/dftorch/script.py:398`).
3. `Constants.py` checks instantiate `Constants` from a synthetic XYZ file and assert f-shell attributes and pair lookup coverage (`src/dftorch/script.py:633`).
4. `Structure.py` checks instantiate `Structure` and `StructureBatch`, then verify AO starts/ends, shell starts/ends, labels, onsite diagonal, `D0`, shell Hubbard values, and padding (`src/dftorch/script.py:956`, `src/dftorch/script.py:1058`).

**State Management:**
- `_bond_integral.read_skf_table` mutates metadata tensors passed in by `get_skf_tensors`; those tensors become immutable-style `torch.nn.Parameter(..., requires_grad=False)` fields in `Constants` unless `GRAD_PARAM` enables selected energies/Hubbard tensors.
- `Structure` and `StructureBatch` own mutable runtime state; drivers add `H0`, `S`, SCF outputs, charges, energies, and forces to these objects.
- `dftorch_params` remains the shared mutable configuration object for input filenames, `SKFPATH`, electronic temperature, Coulomb settings, and optional physics flags.

## Key Abstractions

**Extended SKF Channel Order:**
- Purpose: One canonical 40-column Slater-Koster electronic channel order for Hamiltonian and overlap spline tensors.
- Examples: `_CHANNELS`, `_SIMPLE_CHANNELS`, `_SIMPLE_TO_EXTENDED` in `src/dftorch/_bond_integral.py:101`.
- Pattern: Normalize data at load time; downstream tensors use `len(_CHANNELS)` and never branch on 20 vs 40 columns.

**Shell Presence Matrix:**
- Purpose: Explicitly state which of s/p/d/f shells exist for each atomic number.
- Examples: `SHELL_PRESENT` in `src/dftorch/_bond_integral.py:1001`, `const.shell_present` in `src/dftorch/Constants.py:172`, `structure.shell_present` in `src/dftorch/Structure.py:363`.
- Pattern: Use shape `(120, 4)` for constants, `(Nats, 4)` for single structures, and `(B, Nats, 4)` for batches.

**Nested AO Template:**
- Purpose: Fixed local AO positions for all possible s/p/d/f shells.
- Examples: `SHELL_DIMS`, `SHELL_LOCAL_STARTS`, `AO_LABEL_TEMPLATE`, `AO_SHELL_TEMPLATE` in `src/dftorch/Structure.py:11`.
- Pattern: Build 16-position templates and use `_ao_mask_from_shell_present` to remove absent shell positions.

**Atom-Major AO Ranges:**
- Purpose: Map each atom to its contiguous AO segment in Hamiltonian/overlap matrices.
- Examples: `H_INDEX_START`, `H_INDEX_END`, `HDIM` in `src/dftorch/Structure.py:334` and `src/dftorch/Structure.py:532`.
- Pattern: Compute from `const.n_orb[self.TYPE]`; do not infer offsets by manually adding shell dimensions in driver code.

**Shell-Resolved Hubbard Ranges:**
- Purpose: Provide open-shell and spin-aware code with shell-major Hubbard and occupation vectors.
- Examples: `Hubbard_U_sr`, `shell_types`, `el_per_shell`, `H_INDEX_START_U`, `H_INDEX_END_U` in `src/dftorch/Structure.py:434`.
- Pattern: Stack s/p/d/f arrays from `Constants`, mask by `shell_present`, and preserve atom-major shell order.

**Validation Harness:**
- Purpose: Local, direct-module f-orbital checks independent of public package imports.
- Examples: `find_project_root`, `load_dftorch_module`, `run_bond_integral_tests`, `run_constants_tests`, `run_structure_tests` in `src/dftorch/script.py`.
- Pattern: Build expected values by parsing SKF fixtures independently, then compare against the production parser/constants/structure output.

## Entry Points

**SKF Tensor Loading:**
- Location: `src/dftorch/_bond_integral.py:947`
- Triggers: `Constants.__init__` calls `get_skf_tensors(TYPE, SKFPATH)`.
- Responsibilities: Load all ordered SKF pair files, normalize electronic channels, compute spline coefficients, parse repulsive splines, populate element metadata, apply optional `wfc.hsd` shell overrides.

**Constants Construction:**
- Location: `src/dftorch/Constants.py:45`
- Triggers: User or validation code instantiates `Constants(dftorch_params)`.
- Responsibilities: Discover active species from `FILENAME`, create `pair_lookup`, register pair tensors and f-shell metadata fields.

**Single Structure Construction:**
- Location: `src/dftorch/Structure.py:214`
- Triggers: User or validation code instantiates `Structure(dftorch_params, const, ...)`.
- Responsibilities: Build atom-major AO/shell layout, onsite diagonal, shell-resolved Hubbard data, and initial density.

**Batched Structure Construction:**
- Location: `src/dftorch/Structure.py:461`
- Triggers: User or validation code instantiates `StructureBatch(dftorch_params, const, ...)`.
- Responsibilities: Build padded per-structure AO rows, flattened/global AO offsets, shell-resolved metadata, and batched initial density.

**Single Electronic Structure Driver:**
- Location: `src/dftorch/ESDriver.py:51`
- Triggers: `ESDriver.forward(structure, const, do_scf=True)`.
- Responsibilities: Pass f-orbital-aware structure metadata into Hamiltonian/overlap assembly and SCF.

**Batched Electronic Structure Driver:**
- Location: `src/dftorch/ESDriver.py:1150`
- Triggers: `ESDriverBatch.forward(structure, const, do_scf=True)`.
- Responsibilities: Pass batched f-orbital-aware structure metadata into batched Hamiltonian/overlap assembly and SCF.

**Scoped f-Orbital Validation:**
- Location: `src/dftorch/script.py:1232`
- Triggers: `python src/dftorch/script.py [tests/f_orbital_data]`.
- Responsibilities: Validate SKF normalization, spline reconstruction, constants exposure, and single/batch structure AO bookkeeping.

## Architectural Constraints

- **Threading:** Scoped f-orbital code uses regular Python and PyTorch tensor execution; no scoped file creates threads or distributed workers.
- **Global state:** `torch.get_default_dtype()` controls parser and structure tensor dtypes in `_bond_integral.py`, `Constants.py`, and `Structure.py`; `src/dftorch/script.py:1233` sets `torch.float64` for validation.
- **Circular imports:** `src/dftorch/script.py` deliberately creates a fake `dftorch` package and loads modules by file path to avoid importing `dftorch/__init__.py` while validating scoped internals.
- **Nested shells:** `_validate_nested_shells` rejects f without s/p/d, d without s/p, and p without s (`src/dftorch/_bond_integral.py:496`).
- **Channel count:** Downstream SKF tensors assume exactly 40 electronic channels after normalization; `Constants` validation checks `const.coeffs_tensor.shape[2] == 40` (`src/dftorch/script.py:663`).
- **Fixed local AO positions:** f-shell starts at local AO index `9`; any non-nested basis would require changing `SHELL_LOCAL_STARTS`, mask logic, and validation expectations (`src/dftorch/Structure.py:11`).
- **Metadata source:** Homonuclear SKF headers define `N_F`, `EF`, `UF`, and `n_f`; optional `wfc.hsd` changes shell presence and angular metadata only, not occupation totals (`src/dftorch/_bond_integral.py:876`).
- **Batch padding:** `StructureBatch.diagonal` and `StructureBatch.D0` are padded to max `HDIM` per batch; downstream batch code must respect per-row `HDIM_struct` (`src/dftorch/Structure.py:595`).

## Anti-Patterns

### Inferring f-Orbital Layout From `max_ang` Alone

**What happens:** New code assumes all shells from `s` through `max_ang` exist.
**Why it's wrong:** The scoped implementation stores explicit `shell_present` metadata and supports optional `wfc.hsd` overrides; shell presence is the authoritative basis layout.
**Do this instead:** Use `const.shell_present`, `structure.shell_present`, `structure.has_f`, `_ao_mask_from_shell_present`, and shell range fields from `src/dftorch/Structure.py`.

### Passing Raw SKF Rows Downstream

**What happens:** New parser or driver code branches on 20-column vs 40-column SKF rows outside `_bond_integral.py`.
**Why it's wrong:** `Constants`, `Structure`, and drivers assume `coeffs_tensor` is already normalized to 40 channels.
**Do this instead:** Normalize in `_normalize_skf_row` and keep all downstream tensors in `_CHANNELS` order (`src/dftorch/_bond_integral.py:397`).

### Manually Recomputing AO Offsets in Drivers

**What happens:** Driver code recreates shell offsets by summing hard-coded shell dimensions.
**Why it's wrong:** `Structure` and `StructureBatch` already own atom-major AO and shell ranges; recomputing in drivers risks drifting from `shell_present` and batch padding.
**Do this instead:** Use `structure.H_INDEX_START`, `structure.H_INDEX_END`, `structure.shell_ao_start`, `structure.shell_ao_end`, and `structure.n_orbitals_per_atom` (`src/dftorch/Structure.py:334`).

### Updating Single-Structure Metadata Without Batch Parity

**What happens:** A new shell or AO metadata field is added to `Structure` only.
**Why it's wrong:** `ESDriverBatch` depends on matching batched fields such as `diagonal`, `D0`, `n_orbitals_per_atom`, and shell-derived arrays.
**Do this instead:** Add matching logic in `StructureBatch` and update validation checks in `src/dftorch/script.py:1058`.

## Error Handling

**Strategy:** Fail early while parsing malformed or unsupported SKF basis data; use validation script assertions for contract checks.

**Patterns:**
- `_normalize_skf_row` raises `ValueError` when an electronic row is neither 20 nor 40 values (`src/dftorch/_bond_integral.py:428`).
- `_split_skf_pair_name` raises `ValueError` when a compact or dashed SKF basename cannot be parsed into element symbols (`src/dftorch/_bond_integral.py:493`).
- `_validate_nested_shells` raises `ValueError` for unsupported non-nested shell layouts (`src/dftorch/_bond_integral.py:522`).
- `read_skf_table` raises `ValueError` for empty files, malformed grid/header data, missing spline blocks, short electronic tables, and malformed repulsive spline rows (`src/dftorch/_bond_integral.py:592`).
- `Constants` catches optional `spinw.txt` load failures and proceeds with `self.w = None` (`src/dftorch/Constants.py:127`).
- `Structure` and `StructureBatch` raise `ValueError` for invalid closed-shell electron parity unless `ignore_spin` is used (`src/dftorch/Structure.py:355`, `src/dftorch/Structure.py:545`).

## Cross-Cutting Concerns

**Logging:** Scoped code primarily uses `print` in `Constants` and validation output in `src/dftorch/script.py`; no scoped logging framework is used.
**Validation:** `src/dftorch/script.py` is the focused f-orbital validation harness and checks parser, constants, and structure contracts against `tests/f_orbital_data`.
**Authentication:** Not applicable in scoped files.

---

*Architecture analysis: 2026-07-20*
