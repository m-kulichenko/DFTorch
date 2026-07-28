# F-Orbital DFTB/SKF Pitfalls

**Project:** DFTorch f-orbital support  
**Research type:** Pitfalls dimension  
**Researched:** 2026-07-17  
**Overall confidence:** HIGH for codebase-specific risks; MEDIUM for scientific validation risks until reference simulations are reproduced.

## Scope

This document focuses on likely mistakes when adding f-orbital SKF parsing, Constants storage, Slater-Koster angular transformations, H0/S construction, and full simulation validation to the existing PyTorch DFTB codebase. The highest-risk theme is that parsing can appear to work while downstream tensors still assume only `s`, `sp`, or `spd` bases with orbital counts `1`, `4`, and `9`.

## Critical Pitfalls

### 1. Parser Success Without Scientific Channel Correctness

**What goes wrong:** Extended `.skf` rows are accepted as 40 values, but the internal channel order, simple-to-extended mapping, or Hamiltonian/overlap offset (`SH_shift`) does not match the formulas used later.

**Why it happens:** `_bond_integral.py` now defines a 40-channel order with f channels first, while simple 20-column rows are copied into selected positions. The SK kernel still retrieves channels by numeric indexes such as `9`, `7`, `3`, and `0`, so any channel-order mismatch silently puts the wrong radial integral into the right-shaped tensor.

**Warning signs:**
- Spline reconstruction tests pass only shape checks, not per-channel values.
- Simple MIO fixtures still run, but f-fixture channels evaluate as zeros or shifted values.
- `Hss0`, `Hsp0`, `Hsd0`, `Hsf0`, `Hdd*`, `Hdf*`, or `Hff*` values are correct in `coeffs_tensor` only when inspected manually.
- Changing `_CHANNELS` requires edits in `_slater_koster_pair.py` magic channel indexes.

**Prevention strategy:**
- Treat `_CHANNELS` as a formal contract. Add named channel lookup constants and remove magic indexes in SK code before wiring f formulas deeply.
- Add direct parser tests that evaluate cubic splines at original grid points and compare all nonzero channels from `tests/f_orbital_data/EuEu.skf`, `EuN.skf`, `NEu.skf`, and a simple file such as `tests/data_skf_mio-1-1/C-C.skf`.
- Test both 20-column and 40-column rows through the same normalized representation.
- Keep units explicit: `.skf` electronic values are converted by `27.21138625`, radial grids by `0.52917721`; tests should compare in the same unit after conversion.

**Phase/checkpoint:** Phase 1, SKF parser and spline checkpoint. Do not proceed to Constants until per-channel reconstruction is proven.

**Tests/fixtures that catch it:**
- `tests/f_orbital_data/EuEu.skf` for full f-channel coverage.
- `tests/f_orbital_data/EuN.skf` and `NEu.skf` for heteronuclear ordering.
- `tests/data_skf_mio-1-1/C-C.skf` for simple 20-column regression.
- A unit test that asserts `channels_to_matrix(...)[..., _CHANNELS.index("Hss0")]` reproduces the source `Hss0`, not a positional assumption.

### 2. Missing f Shell State in Constants

**What goes wrong:** The parser reads `Ef`, `Uf`, and f occupations, but `Constants` exposes only `n_s`, `n_p`, `n_d`, `Es`, `Ep`, `Ed`, `U`, `Up`, and `Ud`. Downstream code then has no authoritative f-shell on-site energy, Hubbard U, or occupancy.

**Why it happens:** `read_skf_table()` parses `Ef`, `Uf`, and `ff`, but the returned tuple and `Constants` storage currently stop at d-shell fields. A 16-orbital atom can be counted in `N_ORB` while its f-shell diagonal and density inputs are absent or zero.

**Warning signs:**
- `const.n_orb[Eu] == 16`, but there is no `const.n_f`, `const.Ef`, or `const.Uf`.
- `Structure.diagonal` length is less than `sum(const.n_orb[TYPE])`.
- f simulations converge to finite numbers with f-shell diagonals all omitted.
- Mulliken charge or initial density is wrong for Eu even though `TORE` includes f occupation.

**Prevention strategy:**
- Extend the storage contract atomically: parser return tuple, `Constants` parameters, Structure shell templates, atomic density matrix, shell-resolved Hubbard arrays, and batch equivalents must all include f fields in the same phase.
- Add invariants after `Constants` and `Structure` construction: `len(structure.diagonal) == int(structure.n_orbitals_per_atom.sum())`; shell-resolved arrays must sum to `const.max_ang[TYPE].sum()`.
- Store f-shell absence explicitly for non-f elements rather than inferring from zero-valued tensors in multiple places.

**Phase/checkpoint:** Phase 2, Constants and Structure storage checkpoint.

**Tests/fixtures that catch it:**
- Construct `Constants` from `tests/f_orbital_data/` and assert Eu has `n_orb == 16`, `max_ang == 4`, nonzero or explicitly parsed `n_f`, `Ef`, `Uf` where the file provides them.
- Construct a minimal Eu-containing `Structure` and assert `HDIM == sum(n_orb)`, `diagonal` has 16 AO entries for Eu, and shell arrays include `s/p/d/f`.
- Re-run the existing CH4 SCF smoke test to ensure `n_f` additions do not affect simple elements.

### 3. Hard-Coded 1/4/9 Orbital Masks Drop f Pairs

**What goes wrong:** Neighbor pairs involving f atoms are excluded from H0/S assembly because masks only recognize `1`, `4`, and `9` orbitals.

**Why it happens:** `_h0ands.py` builds masks such as `HH`, `HX`, `XY`, and `YY` from `const.n_orb == 1/4/9`. A f-bearing `spdf` atom has 16 orbitals, so it will not match existing masks unless every pair category is expanded.

**Warning signs:**
- f-containing neighbor pairs exist in the neighbor list but no off-diagonal matrix entries are written for their AO blocks.
- H0/S dimensions grow, but the f rows/columns remain all zero except diagonal.
- Matrix assembly returns finite tensors, hiding the missing interactions.
- Eu-N or Eu-Ga systems produce results dominated by onsite terms.

**Prevention strategy:**
- Replace count-name masks with shell-presence masks or explicit `(lmax_i, lmax_j)` categories.
- Add a debug/invariant check in H0/S assembly during development: every valid neighbor pair must be consumed by exactly one pair-category route.
- After matrix assembly, assert nonzero expected AO block activity for fixture pairs whose SKF channels are nonzero.
- Keep a clear pair taxonomy: `s`, `sp`, `spd`, `spdf`, and direction-reversed equivalents.

**Phase/checkpoint:** Phase 4, H0/S construction checkpoint, with precursor design in Phase 3 SK transformations.

**Tests/fixtures that catch it:**
- A tiny two-atom Eu-N geometry with one neighbor pair and f data; assert Eu f-row blocks receive expected nonzero entries where source channels are nonzero.
- A pair-coverage unit test that counts valid neighbor pairs and mask coverage.
- Existing simple CH4 and C/N/O/S/P/Zn fixtures to prove 1/4/9 paths remain unchanged.

### 4. Incomplete f Angular Slater-Koster Formulas

**What goes wrong:** Parser and tensor shapes support f channels, but angular transformations for `s-f`, `p-f`, `d-f`, and `f-f` are missing, sign-flipped, direction-reversed incorrectly, or differentiated incorrectly.

**Why it happens:** `_slater_koster_pair.py` is a large handwritten formula file. It currently contains extensive s/p/d formulas and derivatives. Adding f orbitals multiplies the number of AO block entries and derivative terms, and copy/paste errors can pass finite-value smoke tests.

**Warning signs:**
- H0/S values differ under swapping atom order beyond expected transpose behavior.
- Values on high-symmetry bond axes do not reduce to simple sigma/pi/delta/phi channel expectations.
- Force or stress tests fail while energy-only tests pass.
- Derivative tensors are finite but disagree with finite differences.

**Prevention strategy:**
- Prototype f angular formulas in small isolated helpers before integrating into the large vectorized kernel.
- Validate formulas at high-symmetry directions: bond along x, y, z, and normalized diagonal directions.
- Add atom-order symmetry tests: `A-B` block should match the transposed/reoriented `B-A` block within tolerance using `EuN.skf` and `NEu.skf`.
- Add finite-difference derivative checks for a two-atom fixture before using f derivatives in forces/stress.
- Use named AO offsets for the seven f functions; do not scatter with naked `+9` through `+15` indexes without a single documented ordering contract.

**Phase/checkpoint:** Phase 3, Slater-Koster angular transformation checkpoint; Phase 5 for derivative/force validation.

**Tests/fixtures that catch it:**
- Formula unit tests with synthetic channel tensors where one channel is nonzero at a time.
- Eu-Eu, Eu-N, N-Eu, Eu-Ga, and Ga-Eu two-atom fixtures.
- Finite-difference comparison of `dH0`/`dS` against recomputed H0/S under small displacement.

### 5. AO Ordering Drift Between Diagonal, Density, SK Blocks, and Reference Data

**What goes wrong:** Different modules use different f-orbital ordering, so tensors have correct sizes but entries refer to different basis functions.

**Why it happens:** Current code documents AO order for s/p/d as `[s, px, py, pz, dxy, dyz, dzx, dx2-y2, dz2]`. f orbitals introduce seven more functions and the reference data may assume a particular real-harmonic ordering. If `Structure`, `_atomic_density_matrix.py`, and SK scatter offsets do not share one ordering, errors are silent.

**Warning signs:**
- Matrix symmetry holds, but reference energy or band results are wrong.
- Rotational/high-symmetry tests fail only for f blocks.
- Occupations per shell look right, but per-AO density initialization is assigned to wrong f functions.
- Batch and single implementations disagree.

**Prevention strategy:**
- Create a single AO ordering definition for each shell and import/use it everywhere.
- Document f offsets and names in one source of truth, then use constants in Structure, density initialization, and SK scatter code.
- Add a test that constructs a one-atom Eu Structure and asserts exact AO labels/order, diagonal values, and initial occupations.

**Phase/checkpoint:** Phase 2 for AO state, Phase 3 for SK formulas, Phase 4 for H0/S integration.

**Tests/fixtures that catch it:**
- Synthetic one-atom Eu structure test for diagonal and `D0`.
- Synthetic two-atom Eu-Eu one-channel tests for every f AO offset.
- Single-vs-batch AO order parity test if batch f support is implemented.

### 6. Initial Density and Shell-Resolved Hubbard Arrays Remain s/p/d Only

**What goes wrong:** SCF starts from an invalid density and DFTB3/spin-related shell arrays omit f-shell information.

**Why it happens:** `_atomic_density_matrix.py` distributes occupations only over s, p, and d offsets. `Structure` shell-resolved arrays stack only `(UsA, UpA, UdA)` and `(ns, np, nd)`. These are consumed by charge, shell, spin, and DFTB3 pathways.

**Warning signs:**
- `D0.sum()` does not match total reference electrons represented by `const.tore[TYPE]`.
- `n_shells_per_atom` says 4 for Eu, but shell-resolved arrays have only 3 entries.
- SCF convergence is unstable for f fixtures but parser/H0 construction appears correct.
- Spin or magnetic Hubbard setup indexes `MAX_ANG_OCC` into missing shell data.

**Prevention strategy:**
- Update density initialization and shell-resolved arrays in the same checkpoint as Constants.
- Add invariant tests: per-atom `D0` sum equals `n_s + n_p + n_d + n_f`; shell-resolved arrays length equals `max_ang`.
- Decide explicitly whether f-shell Hubbard U and magnetic Hubbard data are supported, approximated, or rejected with a clear error.

**Phase/checkpoint:** Phase 2, Constants/Structure checkpoint; revisit in Phase 5 for SCF validation.

**Tests/fixtures that catch it:**
- One-atom Eu Structure invariant tests.
- Closed-shell/open-shell validation for reference electron counts.
- Existing CH4 SCF regression to ensure density initialization for s/p elements is unchanged.

## Moderate Pitfalls

### 7. Heteronuclear Direction and Filename Resolution Bugs

**What goes wrong:** Compact filenames like `EuN.skf` and `NEu.skf` are resolved, but pair lookup, `IJ_pair_type`, `JI_pair_type`, and channel direction are inconsistent.

**Warning signs:**
- `EuN` and `NEu` produce identical or transposed data when source files differ.
- Heteronuclear tests pass only when both directions are present and symmetric by accident.
- Pair lookup falls back to dashed names for compact fixtures.

**Prevention strategy:**
- Test `_resolve_skf_path()` and pair lookup using compact and dashed names.
- Assert both ordered directions are loaded when present.
- In two-atom H0/S tests, compare `Eu-N` and `N-Eu` blocks against expected directional channel usage.

**Phase/checkpoint:** Phase 1 parser and Phase 4 H0/S pair wiring.

**Tests/fixtures that catch it:** `EuN.skf`, `NEu.skf`, `EuGa.skf`, `GaEu.skf`, plus a dashed simple pair such as `C-H.skf`/`H-C.skf`.

### 8. Grid and Padding Assumptions Hide Cutoff Errors

**What goes wrong:** Fixed tensor sizes and padding allow out-of-range indexing or zero-tail behavior that masks incorrect cutoff handling for extended files.

**Warning signs:**
- `torch.searchsorted()` indexes the artificial zero tail rather than the last physical interval.
- Results change when `RCUT_ELECTRONIC` is near the SKF cutoff.
- Files with more grid points than the padded allocation fail late or truncate.

**Prevention strategy:**
- Assert loaded grid length fits allocated `npts`.
- Store per-pair physical grid/cutoff lengths, not only `R_orb_master`.
- Add tests at first grid point, interior points, last physical grid point, and just beyond cutoff.

**Phase/checkpoint:** Phase 1 parser/spline checkpoint.

**Tests/fixtures that catch it:** Eu extended files with 433 grid points, MIO simple files with 500/519 behavior, and synthetic mini tables if needed.

### 9. Simple-Format Regression From Universal Representation

**What goes wrong:** Normalizing simple SKF rows into the extended 40-channel layout changes existing s/p/d calculations through channel offset mistakes, padding, dtype/device changes, or pair lookup behavior.

**Warning signs:**
- CH4 SCF energy, forces, or overlap differ after parser-only changes.
- Tutorial notebook outputs change before f formulas are used.
- Simple files still parse but non-f channels move positions.

**Prevention strategy:**
- Freeze simple-format regression outputs before and after parser normalization.
- Add tests that simple 20-column input normalized to 40 columns has zeros only in f-related positions and preserves every original s/p/d channel exactly.
- Keep old expected CH4 smoke behavior as a CI guard.

**Phase/checkpoint:** Phase 1 and every later checkpoint as a regression gate.

**Tests/fixtures that catch it:**
- `tests/test_scf.py` CH4 smoke test.
- `experiments/1_tutorial.ipynb` converted to a lightweight scripted regression if possible.
- Parser-only equality tests for `tests/data_skf_mio-1-1/`.

### 10. Batch Path Divergence

**What goes wrong:** Single-system f support works, but `StructureBatch`, `H0_and_S_vectorized_batch`, and `Slater_Koster_Pair_SKF_vectorized_batch` still assume 9-orbital atoms.

**Warning signs:**
- Single Eu fixture runs, batch Eu fixture crashes or silently omits f blocks.
- Batch templates are still shape `(batch, Nats, 9)`.
- Batch pair masks still only include `1/4/9`.

**Prevention strategy:**
- Decide phase-by-phase whether batch f support is in scope. If not, raise a precise error for f atoms in batch workflows.
- If in scope, update batch and single code together and add parity tests.

**Phase/checkpoint:** Phase 2 for StructureBatch decision; Phase 4 for batched H0/S if supported.

**Tests/fixtures that catch it:** Two small structures batched together, one simple and one Eu-containing; compare batch output to individual single runs or assert a clear unsupported error.

### 11. Force and Stress Validation Lags Energy Validation

**What goes wrong:** H0/S matrices and SCF energies look plausible, but `dH0`, `dS`, force, and stress paths are wrong for f interactions.

**Warning signs:**
- Energy-only reference checks pass while finite-difference force checks fail.
- f derivative blocks are zero or mismatched in shape.
- Stress calculations fail only under periodic cells.

**Prevention strategy:**
- Do not treat energy validation as full simulation validation.
- Add finite-difference force tests on two-atom and small-cell f fixtures.
- Gate stress support separately; if not validated, raise or mark unsupported for f systems.

**Phase/checkpoint:** Phase 5, force/stress validation after H0/S is stable.

**Tests/fixtures that catch it:** Two-atom finite-difference force checks; reference-paper geometry fixtures once available; existing force smoke test for simple regression.

## Minor Pitfalls

### 12. Integer Dtype for Occupations and Charges

**What goes wrong:** Fractional occupations from extended SKF headers are stored in integer tensors, truncating values.

**Warning signs:**
- `N_S`, `N_P`, `N_D`, or future `N_F` drop fractional values.
- `TORE` differs from the sum of parsed shell occupations.

**Prevention strategy:** Store occupations and `TORE` in floating dtype unless every consumer requires integer electron counts and performs explicit conversion.

**Phase/checkpoint:** Phase 1 parser and Phase 2 Constants.

**Tests/fixtures that catch it:** Header parse tests for `EuEu.skf`, whose occupations include fractional shell values.

### 13. Silent Unsupported Physics for f Shells

**What goes wrong:** Optional features such as spin, DFTB3, magnetic Hubbard, ML-SK, stress, or batch mode run with f systems without scientific validation.

**Warning signs:**
- Feature branches execute because tensors have compatible shapes, not because formulas/data are supported.
- Missing `spinw.txt` or f magnetic data is downgraded to a warning while the user requested spin-sensitive f calculations.

**Prevention strategy:** Add explicit capability checks for f systems. Unsupported feature combinations should raise clear `ValueError` or `NotImplementedError` until validated.

**Phase/checkpoint:** Phase 5 full simulation validation.

**Tests/fixtures that catch it:** Negative-path tests for f + unsupported feature combinations; positive tests only after reference validation.

### 14. Large Handwritten Formula File Becomes Unreviewable

**What goes wrong:** f support is added as thousands of inline scatter statements in `_slater_koster_pair.py`, making errors hard to find and making single/batch parity hard to maintain.

**Warning signs:**
- Formula blocks contain repeated numeric offsets and channel indexes.
- Single and batch formulas are manually duplicated and drift.
- Reviewers cannot tell which formulas are new versus moved.

**Prevention strategy:** Introduce small named helpers for channel evaluation, AO offsets, and formula blocks. Keep prototype code explicit, but isolate f blocks enough that tests can target them directly.

**Phase/checkpoint:** Phase 3 SK transformations.

**Tests/fixtures that catch it:** Formula-level unit tests with synthetic channels and parity tests between helper output and assembled matrix blocks.

## Phase-Specific Warning Matrix

| Phase/checkpoint | Likely pitfall | Mitigation | Required tests/fixtures |
|---|---|---|---|
| Phase 1: SKF parser and spline validation | 20/40-column channel mapping, compact filename resolution, fractional occupation truncation | Named channel contract; per-channel spline reconstruction; float occupations | `EuEu.skf`, `EuN.skf`, `NEu.skf`, `C-C.skf` |
| Phase 2: Constants and Structure storage | `n_orb == 16` without f diagonal/U/occupation storage; 9-orbital templates | Add `n_f`, `Ef`, `Uf`, `has_f`; AO invariant checks | One-atom Eu Structure; CH4 regression |
| Phase 3: f Slater-Koster transforms | Wrong f angular formulas, signs, derivatives, AO ordering | Isolated formula helpers; high-symmetry and atom-swap tests | Synthetic one-channel tests; Eu-Eu/Eu-N/N-Eu |
| Phase 4: H0/S construction | f neighbor pairs not covered by masks; all-zero f blocks | Shell-presence pair routing; coverage assertions | Two-atom Eu-N H0/S block tests; simple CH4 unchanged |
| Phase 5: SCF, forces, and full validation | Plausible energies with wrong forces or unsupported options | Finite-difference force gates; explicit unsupported-feature errors | Reference f fixtures; force smoke; negative-path feature tests |
| Ongoing regression gate | Simple-format behavior changes | Run simple parser, SCF, force, and tutorial-style checks after every phase | `tests/data_skf_mio-1-1/`, `tests/test_scf.py`, tutorial script |

## Minimum Test Checklist Before Roadmap Completion

- Parser reconstructs all nonzero source SKF electronic channels for extended f files and simple files.
- Simple 20-column rows normalize to 40 columns with f-only zeros and exact s/p/d preservation.
- `Constants` exposes f shell values and preserves non-f values unchanged.
- `Structure` and `StructureBatch` either support 16-orbital atoms or reject them clearly.
- `D0.sum()` and shell-resolved arrays agree with parsed occupations for Eu.
- Every valid neighbor pair is consumed by exactly one H0/S route.
- f AO blocks in H0/S are nonzero when source f channels are nonzero.
- H0/S symmetry and atom-order tests pass for `EuN`/`NEu`.
- `dH0`/`dS` finite-difference tests pass before force validation is accepted.
- Existing CH4 SCF/force smoke test and simple parser regressions remain unchanged.

## Sources

- `.planning/PROJECT.md` (milestone goals and constraints)
- `.planning/codebase/ARCHITECTURE.md` (Constants -> Structure -> ESDriver data flow)
- `.planning/codebase/CONCERNS.md` (fragile areas and regression gaps)
- `.planning/codebase/TESTING.md` (pytest patterns and fixtures)
- `src/dftorch/_bond_integral.py` channel and parser implementation
- `src/dftorch/Constants.py` current Constants storage contract
- `src/dftorch/Structure.py` current 9-orbital Structure templates
- `src/dftorch/_h0ands.py` current `1/4/9` pair masks
- `src/dftorch/_atomic_density_matrix.py` current s/p/d occupation initialization
- `tests/f_orbital_data/` and `tests/data_skf_mio-1-1/` fixture formats
