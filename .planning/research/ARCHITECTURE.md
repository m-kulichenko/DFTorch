# Architecture Research: f-Orbital DFTB/SKF Support

**Project:** DFTorch f-orbital support
**Dimension:** Architecture
**Researched:** 2026-07-17
**Confidence:** HIGH for local architecture and risk ordering; MEDIUM for final f-angular formula completeness until validated against reference data.

## Recommendation

Integrate f-orbital support by preserving the current `Constants -> Structure -> ESDriver -> _h0ands -> _slater_koster_pair` flow and moving all raw SKF format differences behind `_bond_integral.py`. Downstream components should consume one universal SKF tensor layout, one orbital-count contract, and one shell-count contract. Do not branch later code on "simple" versus "extended" SKF files.

The lowest-risk build order is:

1. Normalize SKF parsing and spline reconstruction.
2. Store f-shell constants and orbital metadata.
3. Extend `Structure` orbital/shell indexing and atomic reference density.
4. Extend H0/S pair routing and f Slater-Koster formulas.
5. Extend shell-resolved Coulomb/spin/stress paths only after H0/S is validated.
6. Run full SCF/simulation validation while preserving simple-format tutorial outputs.

## Component Boundaries

| Component | Files | Responsibility for f support | Boundary Rule |
|-----------|-------|------------------------------|---------------|
| SKF parser and spline builder | `src/dftorch/_bond_integral.py` | Parse simple 20-column and extended 40-column SKF tables, normalize to the 40-channel internal order, compute cubic spline coefficients, parse f onsite/Hubbard/occupation fields. | This is the only layer that should know raw SKF file format. |
| Constants storage | `src/dftorch/Constants.py` | Register `n_f`, `Ef`, `Uf`, expanded `shell_dim`, `n_orb`, `max_ang`, `max_ang_occ`, and 40-channel `coeffs_tensor`. | `Constants` exposes tensors, not parser-specific structs. |
| Geometry/orbital indexing | `src/dftorch/Structure.py` | Convert `const.n_orb[TYPE]` into AO ranges, diagonal onsite vectors, shell-resolved Hubbard vectors, `shell_types`, `el_per_shell`, `D0`. | `Structure` owns AO and shell indexing; kernels should not infer basis layout from element names. |
| Atomic density helper | `src/dftorch/_atomic_density_matrix.py` | Fill reference occupation for s/p/d/f AO offsets. | Must share the same AO order as `Structure.diagonal`. |
| H0/S assembly router | `src/dftorch/_h0ands.py` | Build pair masks for orbital counts `1`, `4`, `9`, `16`; pass masks, spline indices, pair types, and AO starts into the SK kernel. | It routes pairs and geometry; it should not implement angular formulas. |
| Angular SK kernel | `src/dftorch/_slater_koster_pair.py` | Add s-f, p-f, d-f, f-s, f-p, f-d, and f-f angular blocks and derivatives for both single and batch paths. | Owns all AO block writes and derivative formulas. |
| Stress replay | `src/dftorch/_stress.py` | Reconstruct the same f-aware pair masks from stored metadata before replaying the SK kernel. | Must stay in lockstep with `_h0ands.py` masks. |
| Shell Coulomb paths | `src/dftorch/_coulomb_matrix.py`, batch equivalents if used | Add fourth-shell interactions using `Uf` and `max_ang == 4` where shell-resolved Coulomb is active. | Shell-space dimensions are based on `Structure.H_INDEX_START_U`, not AO dimensions. |
| Driver orchestration | `src/dftorch/ESDriver.py`, `src/dftorch/MD.py` | Pass expanded tensors through existing calls and preserve feature ordering. | Avoid new public API unless a real user-facing option is needed. |
| Validation harness | `tests/`, `src/dftorch/script.py`, `experiments/1_tutorial.ipynb` | Assert spline reconstruction, shape invariants, H0/S sanity, SCF reference behavior, and simple-format regressions. | Tests should catch channel/order bugs before full SCF. |

## Data Flow

```text
.skf files
  |
  v
_bond_integral.read_skf_table()
  - expands Fortran repeated tokens
  - detects simple vs extended rows
  - maps simple 20 channels into universal 40-channel order
  - parses homonuclear onsite/Hubbard/occupation data
  |
  v
_bond_integral.get_skf_tensors(TYPE, SKFPATH)
  - loads ordered element pairs
  - returns R_orb, coeffs_tensor[pair, interval, channel, cubic_coeff]
  - returns N_ORB/MAX_ANG/MAX_ANG_OCC/TORE/N_S/N_P/N_D/(N_F)
  - returns ES/EP/ED/(EF), US/UP/UD/(UF)
  |
  v
Constants
  - registers tensors as non-trainable Parameters unless GRAD_PARAM applies
  - exposes pair_lookup, coeffs_tensor, shell/orbital metadata
  |
  v
Structure / StructureBatch
  - maps atom TYPE -> n_orbitals_per_atom
  - builds H_INDEX_START/H_INDEX_END and HDIM
  - builds onsite diagonal in AO order
  - builds shell-resolved Hubbard vectors and shell index ranges
  - builds D0 atomic reference density
  |
  v
ESDriver
  - builds neighbor lists and pair type tensors
  - calls H0_and_S_vectorized / batch
  |
  v
_h0ands
  - computes distances, direction cosines, spline interval indices
  - builds f-aware pair masks from const.n_orb
  - calls Slater_Koster_Pair_SKF_vectorized for H and S
  |
  v
_slater_koster_pair
  - evaluates radial channels from coeffs_tensor
  - applies angular formulas and derivatives
  - writes AO blocks into flattened H0/S and dH0/dS
  |
  v
SCF / forces / stress / simulations
```

## Shape and Orbital Invariants

Protect these invariants with tests before full simulation work:

| Invariant | Expected Contract | Why It Matters |
|-----------|-------------------|----------------|
| Universal channel count | `coeffs_tensor.shape[2] == 40` for SKF mode. | Existing simple data and new extended data must share one downstream layout. |
| Channel order | Internal order remains `_CHANNELS`: `Hff*`, `Hdf*`, `Hdd*`, `Hpf*`, `Hpd*`, `Hpp*`, `Hsf`, `Hsd`, `Hsp`, `Hss`, then S equivalents. | A single off-by-channel bug silently corrupts H0/S. |
| Simple-format compatibility | 20-column SKF rows map into their matching 40-channel positions and all f channels remain zero. | Preserves existing tutorial and CH4 behavior. |
| AO counts | Supported nested basis counts are `1=s`, `4=sp`, `9=spd`, `16=spdf`. | Current code hard-codes `1/4/9`; f support must add `16` without changing old meanings. |
| AO order | `[s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2, f1..f7]`. | `Structure.diagonal`, `D0`, SK writes, force/stress derivatives must agree exactly. |
| Shell counts | `max_ang` is shell count, not AO count: `1=s`, `2=sp`, `3=spd`, `4=spdf`. | Shell-resolved Coulomb, spin, DFTB3, and Hubbard paths index shells through this value. |
| Shell dimensions | `const.shell_dim` must become `[0, 1, 3, 5, 7]`. | SCF spin/open-shell code repeats shell values into AO values through `shell_dim[shell_types]`. |
| Occupation fields | `TORE = n_s + n_p + n_d + n_f`; `D0` distributes f occupation over 7 AOs. | Wrong reference charge shifts Mulliken charges and SCF convergence. |
| Matrix dimensions | Single: `H0/S/density` are `(HDIM, HDIM)`; batch: padded `(B, max_HDIM, max_HDIM)`. | f atoms increase HDIM sharply; shape bugs surface as wrong scatter offsets. |
| Symmetry | H0 and S remain explicitly symmetrized after assembly; derivative antisymmetry convention remains unchanged. | Existing force/stress code assumes these conventions. |

## Suggested Build Order and Checkpoints

### Phase 1: Parser Normalization and Spline Checks

**Touch:** `src/dftorch/_bond_integral.py`, `tests/f_orbital_data/`, focused parser tests.

**Build:**
- Keep `_CHANNELS` as the canonical 40-channel order.
- Keep simple 20-column rows converted into 40-channel rows before spline construction.
- Return f fields from `read_skf_table()` and `get_skf_tensors()`: `N_F`, `EF`, `UF`.
- Preserve `wfc.hsd` override behavior for `N_ORB`, `MAX_ANG`, and `MAX_ANG_OCC`.

**Checkpoint:**
- For every f fixture, evaluate cubic splines at original grid points and compare to source SKF channel values.
- For existing `tests/data_skf_mio-1-1/`, assert f channels are zero and legacy channel values are unchanged.
- Assert compact names like `EuN.skf` and dashed names resolve consistently.

### Phase 2: Constants Storage

**Touch:** `src/dftorch/Constants.py`.

**Build:**
- Expand returned tuple and registered parameters with `n_f`, `Ef`, `Uf`.
- Change `shell_dim` to `[0, 1, 3, 5, 7]`.
- Keep existing attributes `n_s`, `n_p`, `n_d`, `Es`, `Ep`, `Ed`, `U`, `Up`, `Ud` intact for compatibility.

**Checkpoint:**
- `Constants` loads CH4 simple-format data with the same `n_orb`, `coeffs_tensor`, and onsite values as before.
- `Constants` loads Eu/Ga/N f fixtures with Eu `n_orb == 16`, `max_ang == 4`, and nonzero f metadata where present.

### Phase 3: Structure AO and Shell Indexing

**Touch:** `src/dftorch/Structure.py`, `src/dftorch/_atomic_density_matrix.py`.

**Build:**
- Replace `has_d = n_orb == 9` with `has_d = n_orb >= 9`.
- Add `has_f = n_orb == 16`.
- Expand onsite diagonal templates from 9 to 16 AO slots.
- Expand shell templates from 3 to 4 shell slots.
- Extend atomic density helpers to distribute `n_f / 7` to AO offsets `9:16`.

**Checkpoint:**
- For a single f atom, `HDIM == 16`, `H_INDEX_END == H_INDEX_START + 15`, diagonal length is 16, shell count is 4, and `D0.sum()` equals `TORE`.
- For existing simple systems, `HDIM`, `D0`, `Nocc`, and diagonal values are unchanged.
- Repeat the same checks for `StructureBatch`.

### Phase 4: H0/S Pair Routing

**Touch:** `src/dftorch/_h0ands.py`, `src/dftorch/_stress.py`.

**Build:**
- Introduce f-aware pair classes without changing the existing H/X/Y meanings. A practical naming is:
  - `H`: `n_orb == 1`
  - `X`: `n_orb == 4`
  - `Y`: `n_orb == 9`
  - `Z`: `n_orb == 16`
- Add masks for all pairs involving `Z`: `HZ`, `XZ`, `YZ`, `ZH`, `ZX`, `ZY`, `ZZ`.
- Store `n_orb_I/J` metadata as today, but reconstruct the expanded masks in stress replay.

**Checkpoint:**
- f-free systems produce byte-for-byte or tolerance-equivalent H0/S relative to pre-change tests.
- f-containing systems reach `_slater_koster_pair` with correct pair masks and no out-of-range AO writes.

### Phase 5: Slater-Koster f Angular Blocks

**Touch:** `src/dftorch/_slater_koster_pair.py`; batch function in the same file.

**Build:**
- Keep radial lookup through `_get_val_dR(pair_type, idx, dx, channel, mask, direction)`.
- Add formula blocks in increasing coupling complexity:
  1. s-f / f-s using `Hsf0` / `Ssf0`.
  2. p-f / f-p using `Hpf0`, `Hpf1`, `Spf0`, `Spf1`.
  3. d-f / f-d using `Hdf0`, `Hdf1`, `Hdf2`, S equivalents.
  4. f-f using `Hff0..Hff3`, S equivalents.
- Add derivative writes beside each value write; stress accumulation `_sg()` must cover every f AO block.

**Checkpoint:**
- Unit tests validate all f channel groups can be selected from `coeffs_tensor`.
- H0/S are finite, symmetric after assembly, and have expected `(HDIM, HDIM)` dimensions.
- dH0/dS contain no NaN/Inf for nonzero-distance neighbor pairs.

### Phase 6: Shell-Resolved SCF, Coulomb, and Forces

**Touch:** `src/dftorch/_coulomb_matrix.py`, batch Coulomb helpers if used, `src/dftorch/_spin.py`, `src/dftorch/_forces.py`, `src/dftorch/_forces_batch.py`, `src/dftorch/_scf.py`, `src/dftorch/_xl_tools.py` as needed.

**Build:**
- Add fourth-shell interactions where code currently enumerates `s/p/d` shell pairs.
- Ensure `const.Uf` is used for f-shell short-range Coulomb terms.
- Confirm open-shell/spin code handles `shell_types == 4` via expanded `shell_dim`.
- Decide explicitly whether batch f-orbital SCF is supported in the milestone; if not, raise a clear `ValueError` for unsupported f batch paths.

**Checkpoint:**
- Closed-shell f fixture can run H0/S and enter SCF without shape errors.
- Shell charge vectors have length `sum(max_ang[TYPE])`.
- Simple CH4 PME smoke test remains finite and unchanged within tolerance.

### Phase 7: Full Simulation Validation and Regression Lock

**Touch:** tests and any validation harness, not public APIs.

**Build:**
- Add reference-paper f fixture simulation tests at the smallest stable scope first.
- Add simple-format regression checks for tutorial outputs or extracted deterministic numeric snapshots from `experiments/1_tutorial.ipynb`.
- Keep long reference simulations marked slow if needed; keep parser/H0/S tests fast.

**Checkpoint:**
- f fixture reproduces reference values within agreed tolerances.
- Existing simple-format tests pass without numerical drift.

## Patterns to Follow

### Pattern: Normalize Early

**What:** Convert every SKF file to the 40-channel internal representation in `_bond_integral.py`.

**Why:** H0/S, stress, and tests can then reason about one channel index map. This is already partially implemented through `_normalize_skf_row()` and should remain the central seam.

### Pattern: Keep AO and Shell Contracts Separate

**What:** AO dimensions use `n_orb` values `1/4/9/16`; shell dimensions use `max_ang` values `1/2/3/4`.

**Why:** H0/S is AO-space, while Hubbard/Coulomb/spin are shell-space. Mixing the two is the easiest way to create shape-correct but physically wrong results.

### Pattern: Add Single and Batch Together or Guard Batch

**What:** For each f change, either update both single and batch paths or add a precise unsupported-mode error.

**Why:** The codebase has mirrored `Structure`/`StructureBatch`, `_h0ands` single/batch, and `_slater_koster_pair` single/batch implementations. Silent single-only f support would be hard to diagnose.

## Anti-Patterns to Avoid

### Anti-Pattern: Raw SKF Format Checks in H0/S

**What goes wrong:** H0/S code branches on simple versus extended SKF files.

**Why bad:** It duplicates parser knowledge and makes simple-format regression fragile.

**Instead:** H0/S should only see `coeffs_tensor` in canonical 40-channel order.

### Anti-Pattern: Treating f Support as Only More Channels

**What goes wrong:** Parser returns f channels, but `Structure`, `D0`, shell Coulomb, and spin still assume s/p/d.

**Why bad:** H0/S may assemble, but SCF charges and shell-resolved terms will be wrong.

**Instead:** Promote f metadata through constants and structure before enabling full SCF.

### Anti-Pattern: Replacing Existing Explicit Formula Blocks with a Large Abstraction First

**What goes wrong:** A broad refactor of `_slater_koster_pair.py` obscures whether numerical changes come from refactoring or f formulas.

**Why bad:** This milestone needs a troubleshootable prototype and regression safety.

**Instead:** Add f blocks explicitly, test them, then consider cleanup after reference validation.

## Roadmap Implications

The roadmap should not jump directly from parser support to full SCF validation. The riskiest dependency is not file parsing; it is the chain of shape assumptions from `n_orb == 9` and `max_ang == 3` into AO offsets, shell offsets, Coulomb matrices, stress replay, and batch paths.

Recommended phase sequence:

1. **SKF Canonicalization** - prove both simple and extended files produce correct spline channels.
2. **Constants and Structure Contracts** - add f metadata, AO indexing, shell indexing, and D0 without touching angular formulas.
3. **H0/S f Assembly** - add pair masks and f angular SK formulas; validate matrices before SCF.
4. **SCF Shell Support** - extend shell-resolved Coulomb/spin/force paths to fourth shell.
5. **Reference Simulation Validation** - run f fixtures and preserve tutorial/simple-format regression outputs.

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/CONVENTIONS.md`
- `.planning/codebase/CONCERNS.md`
- `src/dftorch/_bond_integral.py`
- `src/dftorch/Constants.py`
- `src/dftorch/Structure.py`
- `src/dftorch/_atomic_density_matrix.py`
- `src/dftorch/_h0ands.py`
- `src/dftorch/_slater_koster_pair.py`
- `src/dftorch/_stress.py`
- `src/dftorch/_coulomb_matrix.py`
- `tests/test_scf.py`
- `tests/f_orbital_data/`
