# Codebase Structure

**Analysis Date:** 2026-07-20
**Last Mapped Commit:** `e824543a0b411dcf52462ee55db5362c360e7780`

## Directory Layout

```text
DFTorch/
├── src/
│   └── dftorch/
│       ├── Constants.py          # Registers SKF-derived element and pair tensors, including f-shell metadata
│       ├── Structure.py          # Builds single/batch atom-major AO and shell bookkeeping for s/p/d/f bases
│       ├── ESDriver.py           # Consumes structure metadata during Hamiltonian assembly and SCF
│       ├── _bond_integral.py     # Parses `.skf` files, normalizes 20/40 SK channels, builds spline tensors
│       └── script.py             # Manual scoped f-orbital validation harness
└── tests/
    └── f_orbital_data/
        ├── Eu-Eu.skf            # Extended f-electron homonuclear SKF fixture
        ├── Eu-Ga.skf            # Ordered Eu/Ga heteronuclear SKF fixture
        ├── Eu-N.skf             # Ordered Eu/N heteronuclear SKF fixture
        ├── Ga-Eu.skf            # Ordered Ga/Eu heteronuclear SKF fixture
        ├── Ga-Ga.skf            # Homonuclear SKF fixture
        ├── Ga-N.skf             # Ordered Ga/N heteronuclear SKF fixture
        ├── N-Eu.skf             # Ordered N/Eu heteronuclear SKF fixture
        ├── N-Ga.skf             # Ordered N/Ga heteronuclear SKF fixture
        └── N-N.skf              # Homonuclear SKF fixture
```

## Directory Purposes

**`src/dftorch/`:**
- Purpose: Scoped implementation files for f-orbital SKF loading, constants registration, AO/shell layout, and driver consumption.
- Contains: Public-style class modules and private parser/kernel-adjacent modules.
- Key files: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`, `src/dftorch/_bond_integral.py`, `src/dftorch/script.py`.

**`tests/f_orbital_data/`:**
- Purpose: Fixture directory for validating f-orbital SKF parser and structure metadata.
- Contains: Ordered pair SKF files for Eu, Ga, and N. Homonuclear files define element metadata; all ordered pair files provide electronic and repulsive pair tables.
- Key files: `tests/f_orbital_data/Eu-Eu.skf`, `tests/f_orbital_data/Ga-Ga.skf`, `tests/f_orbital_data/N-N.skf`, `tests/f_orbital_data/Eu-Ga.skf`, `tests/f_orbital_data/Ga-Eu.skf`, `tests/f_orbital_data/Eu-N.skf`, `tests/f_orbital_data/N-Eu.skf`, `tests/f_orbital_data/Ga-N.skf`, `tests/f_orbital_data/N-Ga.skf`.

## Key File Locations

**Entry Points:**
- `src/dftorch/_bond_integral.py:947`: `get_skf_tensors(TYPE, skfpath)` loads all ordered pair SKF tensors and f-shell metadata.
- `src/dftorch/Constants.py:45`: `Constants(dftorch_params)` discovers species from `FILENAME`, calls `get_skf_tensors`, and registers f-shell fields.
- `src/dftorch/Structure.py:214`: `Structure(dftorch_params, const, ...)` builds single-structure AO/shell metadata.
- `src/dftorch/Structure.py:461`: `StructureBatch(dftorch_params, const, ...)` builds batched AO/shell metadata.
- `src/dftorch/ESDriver.py:51`: `ESDriver.forward(structure, const, ...)` consumes f-aware single-structure metadata.
- `src/dftorch/ESDriver.py:1150`: `ESDriverBatch.forward(structure, const, ...)` consumes f-aware batched metadata.
- `src/dftorch/script.py:1232`: Manual validation script entry point.

**Configuration:**
- `src/dftorch/Constants.py:57`: `dftorch_params["SKFPATH"]` points to the SKF directory.
- `src/dftorch/Constants.py:71`: `dftorch_params["FILENAME"]` supplies the species source used to decide which ordered pair SKF files to load.
- `src/dftorch/Structure.py:239`: `GRAD_XYZ`, `GRAD_CELL`, `CELL`, `CHARGE`, `SPIN_POL`, and `T_ELECTRONIC` configure structure tensor state.
- `src/dftorch/script.py:1238`: Optional CLI argument selects the SKF fixture directory; default is `tests/f_orbital_data`.

**Core Logic:**
- `src/dftorch/_bond_integral.py:101`: `_CHANNELS` defines the canonical 40-channel extended SK order.
- `src/dftorch/_bond_integral.py:144`: `_SIMPLE_CHANNELS` defines the legacy 20-channel order.
- `src/dftorch/_bond_integral.py:167`: `_SIMPLE_TO_EXTENDED` maps legacy columns into extended columns.
- `src/dftorch/_bond_integral.py:397`: `_normalize_skf_row` returns each electronic row in canonical 40-channel order.
- `src/dftorch/_bond_integral.py:433`: `_resolve_skf_path` supports dashed and compact pair filenames.
- `src/dftorch/_bond_integral.py:465`: `_split_skf_pair_name` parses dashed or compact SKF basenames into element symbols.
- `src/dftorch/_bond_integral.py:496`: `_validate_nested_shells` enforces nested s/p/d/f basis support.
- `src/dftorch/_bond_integral.py:530`: `_shell_metadata_from_presence` computes `n_orb`, `max_ang`, and `max_ang_occ`.
- `src/dftorch/_bond_integral.py:547`: `read_skf_table` reads SKF electronic rows, repulsive splines, and homonuclear metadata.
- `src/dftorch/_bond_integral.py:779`: `cubic_spline_coeffs` builds cubic coefficients for all normalized channels.
- `src/dftorch/_bond_integral.py:869`: `read_wfc_hsd` optionally overrides shell-presence metadata from `wfc.hsd`.
- `src/dftorch/Constants.py:160`: `Constants` registers `n_orb`, `max_ang`, `n_f`, `Ef`, `Uf`, and `shell_present`.
- `src/dftorch/Structure.py:11`: Fixed shell dimensions and local AO shell starts.
- `src/dftorch/Structure.py:25`: `AO_LABEL_TEMPLATE` defines atom-local AO labels through seven f functions.
- `src/dftorch/Structure.py:90`: `_ao_mask_from_shell_present` expands shell flags to a 16-position AO mask.
- `src/dftorch/Structure.py:160`: `_atomic_density_matrix_from_shells` fills single-structure `D0`.
- `src/dftorch/Structure.py:186`: `_atomic_density_matrix_batch_from_shells` fills batched `D0`.

**Testing/Validation:**
- `src/dftorch/script.py:398`: `check_one_skf` verifies normalized channels and spline reconstruction for one SKF file.
- `src/dftorch/script.py:633`: `check_constants_against_expected` verifies `Constants` exposes f-shell metadata and 40-channel tensors.
- `src/dftorch/script.py:956`: `check_single_structure_layout` verifies single-structure AO/shell starts, diagonal, `D0`, labels, and shell Hubbard data.
- `src/dftorch/script.py:1058`: `check_batch_structure_layout` verifies batched AO/shell metadata and padding.
- `tests/f_orbital_data/`: Input fixture directory for the validation script.

## Naming Conventions

**Files:**
- User-facing class modules use `PascalCase.py`: `src/dftorch/Constants.py`, `src/dftorch/Structure.py`, `src/dftorch/ESDriver.py`.
- Private parser/kernel-adjacent modules use leading-underscore snake case: `src/dftorch/_bond_integral.py`.
- Utility validation script uses lowercase module naming: `src/dftorch/script.py`.
- SKF fixtures use dashed ordered-pair names: `tests/f_orbital_data/Eu-Ga.skf`, `tests/f_orbital_data/Ga-Eu.skf`, `tests/f_orbital_data/N-N.skf`.

**Directories:**
- Scoped implementation lives under `src/dftorch/`.
- f-orbital validation fixture data lives under `tests/f_orbital_data/`.

**Variables and Fields:**
- Atomic element metadata arrays use uppercase names while being assembled in `_bond_integral.py`: `N_ORB`, `MAX_ANG`, `N_F`, `EF`, `UF`, `SHELL_PRESENT`.
- Registered `Constants` attributes use lowercase or existing historical names: `n_orb`, `max_ang`, `n_f`, `Ef`, `Uf`, `shell_present`.
- Structure AO index ranges use `H_INDEX_*`: `H_INDEX_START`, `H_INDEX_END`, `H_INDEX_START_GLOBAL`, `H_INDEX_END_GLOBAL`.
- Structure shell index ranges use `shell_*` names for AO ranges and `H_INDEX_*_U` names for shell/Hubbard ranges.
- Shell type IDs follow one-based IDs: `1=s`, `2=p`, `3=d`, `4=f` in `src/dftorch/Structure.py:13`.

## Where to Add New Code

**New f-Orbital SKF Parsing Behavior:**
- Primary code: `src/dftorch/_bond_integral.py`.
- Add channel-layout changes near `_CHANNELS`, `_SIMPLE_CHANNELS`, and `_SIMPLE_TO_EXTENDED` (`src/dftorch/_bond_integral.py:101`).
- Add row-format behavior in `_normalize_skf_row` (`src/dftorch/_bond_integral.py:397`).
- Add filename parsing behavior in `_resolve_skf_path` or `_split_skf_pair_name` (`src/dftorch/_bond_integral.py:433`, `src/dftorch/_bond_integral.py:465`).
- Validation: Extend `src/dftorch/script.py:398` for parser-level checks and use fixtures under `tests/f_orbital_data/`.

**New Element Shell Metadata:**
- Primary code: `src/dftorch/_bond_integral.py` for parsing and tensor creation.
- Constants registration: Add returned tensors to the `get_skf_tensors` tuple and `Constants.__init__` unpack/register block in `src/dftorch/Constants.py:101`.
- Structure consumption: Add single and batched layout handling in `src/dftorch/Structure.py`.
- Validation: Add expected metadata assertions in `src/dftorch/script.py:633`.

**New AO Layout or Shell Type:**
- Primary code: `src/dftorch/Structure.py`.
- Update `SHELL_DIMS`, `SHELL_LOCAL_STARTS`, `AO_LABEL_TEMPLATE`, `AO_SHELL_TEMPLATE`, `_ao_mask_from_shell_present`, and both density builders.
- Update both `Structure` and `StructureBatch`; keep fields aligned.
- Parser support: Update `_bond_integral.py` `MAX_SHELLS`, channel mapping, shell validation, SKF header parsing, and return tensors.
- Validation: Update single and batch checks in `src/dftorch/script.py:956` and `src/dftorch/script.py:1058`.

**New Constants Field Exposed From SKF Data:**
- Primary code: `src/dftorch/Constants.py`.
- Upstream source: Add or fill the tensor in `src/dftorch/_bond_integral.py:get_skf_tensors`.
- Registration pattern: Use `torch.nn.Parameter(tensor, requires_grad=False)` for metadata, or `requires_grad=self.grad_param` only for tunable parameter tensors following `U/Up/Ud/Uf` and `Es/Ep/Ed/Ef`.
- Validation: Add `hasattr` and value checks in `src/dftorch/script.py:633`.

**New Driver Use of f-Orbital Metadata:**
- Primary code: `src/dftorch/ESDriver.py`.
- Single-structure path: Consume existing fields from `Structure`, especially `diagonal`, `H_INDEX_START`, `n_orbitals_per_atom`, `D0`, `el_per_shell`, `shell_types`, and `n_shells_per_atom`.
- Batched path: Add parallel use of `StructureBatch` fields in `ESDriverBatch`.
- Avoid recomputing shell offsets in `ESDriver.py`; extend `Structure.py` instead when a new metadata shape is needed.

**New f-Orbital Fixture Data:**
- Location: `tests/f_orbital_data/`.
- File naming: Add both ordered heteronuclear directions when pair lookup requires both, for example `A-B.skf` and `B-A.skf`.
- Homonuclear files: Add `A-A.skf` for each new element so `read_skf_table` can fill element-level onsite energies, Hubbard values, occupations, and `SHELL_PRESENT`.
- Validation: Run `python src/dftorch/script.py tests/f_orbital_data`.

**Utilities:**
- Parser helpers: Keep SKF text parsing helpers in `src/dftorch/_bond_integral.py`.
- Structure layout helpers: Keep AO/shell tensor shape helpers in `src/dftorch/Structure.py`.
- Validation-only helpers: Keep independent expected-value parsing in `src/dftorch/script.py`; do not import validation helpers into runtime modules.

## Special Directories

**`tests/f_orbital_data/`:**
- Purpose: Scoped f-orbital SKF fixture data.
- Generated: No.
- Committed: Yes.
- Notes: Contains a `.DS_Store` file in addition to `.skf` fixtures; parser and validation code should select `*.skf` rather than consuming every file in the directory.

**`.planning/codebase/`:**
- Purpose: GSD-generated codebase maps consumed by planning and execution commands.
- Generated: Yes.
- Committed: Project-dependent; planning artifacts are local-only for this remap.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`.

---

*Structure analysis: 2026-07-20*
