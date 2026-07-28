# Project Research Summary

DFTorch is a brownfield PyTorch DFTB implementation. f-orbital support should extend the current `Constants -> Structure -> ESDriver -> _h0ands -> _slater_koster_pair` pipeline rather than introduce a new chemistry framework, parser generator, backend, or public API. The core implementation move is to normalize simple and extended SKF files into one canonical 40-channel tensor layout at the parser boundary, then propagate f-shell metadata through constants, AO indexing, H0/S assembly, SCF, and validation.

The research is unanimous that parsing wider `.skf` rows is necessary but not sufficient. A scientifically valid milestone requires per-channel spline reconstruction, `spdf` orbital metadata, 16-orbital AO indexing, f angular Slater-Koster blocks, H/S symmetry and direction checks, shell-resolved SCF support, and regression gates proving current simple-format calculations remain unchanged. The roadmap should therefore be staged as a dependency chain: parser correctness first, metadata and shape contracts second, matrix assembly third, SCF/reference validation fourth, cleanup and parity decisions last.

The dominant risks are silent correctness failures: wrong channel order, missing f onsite/Hubbard/occupation fields, pair masks that drop 16-orbital atoms, AO ordering drift, and finite but physically invalid matrices. Mitigate these with named channel/AO contracts, axis-aligned and atom-swap tests before full SCF, explicit unsupported-mode errors for batch/forces/stress until validated, and stored DFTB+/paper-derived reference outputs instead of runtime DFTB+ dependencies.

## Key Findings

### Stack

- Keep the work inside the existing Python/PyTorch stack; correctness and convention validation are the active risks, not runtime performance.
- Extend `src/dftorch/_bond_integral.py` rather than replacing it. It already has 40-channel scaffolding, simple-to-extended normalization, extended-format detection, compact/dashed filename resolution, and f metadata parsing hooks.
- Extend `src/dftorch/Constants.py` as the parameter boundary with `n_f`, `Ef`, `Uf`, `shell_dim = [0, 1, 3, 5, 7]`, and f-aware `n_orb` / `max_ang` handling.
- Extend `_h0ands.py` and `_slater_koster_pair.py` explicitly from supported AO counts `1/4/9` to include `16`.
- Use `pytest`, `torch.float64`, CPU-first fixtures, and stored reference values. DFTB+ and ASE are useful offline reference generators, not package or CI dependencies.

### Features

- Must-have: parse simple 20-column and extended 40-column SKF rows into a universal 40-channel representation.
- Must-have: reconstruct source electronic table values from cubic splines for f fixtures and simple fixtures.
- Must-have: support homonuclear f metadata, nested `s/sp/spd/spdf` basis validation, f Constants storage, and 16-orbital Structure indexing.
- Must-have: implement f Slater-Koster angular transforms for `s-f`, `p-f`, `d-f`, and `f-f`, including overlap paths and derivatives if force/stress support is claimed.
- Must-have: preserve existing simple-format SCF/tutorial outputs within tolerance.
- Should-have after the prototype: clear unsupported-mode errors, documentation of channel/basis conventions, targeted negative-path parser tests, and cleanup of scattered `1/4/9` assumptions.
- Defer: batch f support, force/stress/MD parity, SEDACS, ML-SK f channels, broad performance optimization, and generalized arbitrary-l abstractions until single-system f correctness is proven.

### Architecture

- `_bond_integral.py` should be the only layer that knows raw SKF format differences; downstream code consumes canonical tensors only.
- `Constants` owns parsed tensor storage and shell/orbital metadata, not parser-specific structs.
- `Structure` owns AO ranges, onsite diagonal vectors, shell-resolved Hubbard arrays, shell labels, and atomic reference density.
- `_h0ands.py` routes neighbor pairs and spline intervals; `_slater_koster_pair.py` owns angular formulas and AO block writes.
- AO contracts and shell contracts must stay separate: `n_orb` is AO count (`1/4/9/16`), while `max_ang` is shell count (`1/2/3/4`).
- Single and batch paths must either be updated together or batch must reject f systems clearly.

### Pitfalls

- Channel mapping can look shape-correct while using the wrong radial integral. Prevent with named channel indexes and per-channel spline reconstruction tests.
- Constants can report `n_orb == 16` while omitting f onsite energy, Hubbard U, or occupations. Prevent by extending parser returns, Constants, Structure, density initialization, and shell arrays in the same phase.
- H0/S masks can silently ignore f atom pairs because they only match `1/4/9`. Prevent with explicit `spdf` routing and pair-coverage assertions.
- f angular formulas can be sign-flipped, direction-reversed, or differentiated incorrectly. Prevent with axis-aligned, one-channel, atom-swap, and finite-difference tests before accepting simulation results.
- AO ordering can drift between diagonal construction, density initialization, SK block writes, and reference data. Prevent with one source of truth for AO labels/offsets and tests on one-atom Eu structures.

## Implications for Roadmap

### Phase 1: SKF Canonicalization and Spline Foundation

Rationale: every downstream tensor depends on stable channel order and correct radial interpolation.

Delivers:
- Simple and extended SKF parsing into one 40-channel layout.
- Compact and dashed pair-file resolution.
- Homonuclear f header parsing sufficient for later metadata.
- Spline reconstruction tests against `tests/f_orbital_data/` and simple fixtures.
- Simple-format regression checks for existing datasets.

Must avoid:
- Shape-only parser tests.
- Magic channel indexes in downstream code.
- Integer truncation of fractional shell occupations.

Research flag: standard patterns; no deeper research needed unless fixture semantics contradict the documented channel order.

### Phase 2: Constants, Structure, and Basis Contracts

Rationale: H0/S and SCF cannot be trusted until f shell state and AO indexing are represented consistently.

Delivers:
- `n_f`, `Ef`, `Uf`, `max_ang == 4`, `n_orb == 16`, and expanded `shell_dim`.
- 16-AO Structure diagonal, AO offsets, shell arrays, and `D0` reference density.
- Nested basis validation and clear errors for skipped-shell or unsupported f paths.
- Existing CH4/simple-system invariants unchanged.

Must avoid:
- Treating f support as only `N_ORB == 16`.
- Letting SCF see f systems before diagonal, occupation, and Hubbard state are correct.
- Silent batch acceptance if `StructureBatch` is still s/p/d only.

Research flag: standard patterns; implementation needs code inspection, not external research.

### Phase 3: H0/S Pair Routing and f Angular Blocks

Rationale: this is the highest-risk numerical phase; it converts parsed f channels into actual Hamiltonian/overlap blocks.

Delivers:
- Pair classes for all interactions involving `spdf` atoms.
- Explicit f angular SK formulas for `s-f`, `p-f`, `d-f`, and `f-f`.
- Matrix shape, symmetry, finite-value, channel-selection, and atom-order tests.
- Guarded or implemented derivative paths depending on force/stress scope.

Must avoid:
- All-zero f blocks caused by missing masks.
- Large untestable formula dumps with naked offsets and channel numbers.
- Claiming force/stress support before derivative finite-difference checks pass.

Research flag: needs deeper phase research if f formula source, real-harmonic ordering, or DFTB+ basis conventions are not already pinned by fixtures.

### Phase 4: SCF and Reference Simulation Validation

Rationale: full simulation should be the final correctness gate, not the first signal.

Delivers:
- Single-system f H0/S -> SCF -> energy smoke path.
- Shell-resolved Coulomb/spin/DFTB3 checks or explicit unsupported errors.
- Stored reference-paper or DFTB+ values with documented geometry, settings, observables, and tolerances.
- Existing simple-format tutorial/SCF/force regressions preserved.

Must avoid:
- Accepting plausible energies without validating H0/S, shell charges, and simple regressions.
- Mixing parameter sets across validation cases.
- Requiring DFTB+ or ASE at runtime or in normal CI.

Research flag: needs deeper phase research to define exact reference observables, tolerances, and any force/stress acceptance criteria.

### Phase 5: Cleanup and Parity Decisions

Rationale: cleanup is valuable after the prototype proves scientific behavior; doing it first increases regression risk.

Delivers:
- Consolidated channel lookup helpers and AO offset definitions.
- Refactoring of temporary f branches where tests prove behavior.
- Documented support policy for batch, forces, stress, MD, SEDACS, ML-SK, and optional physics.
- Negative-path tests for unsupported f feature combinations.

Must avoid:
- Broad driver rewrites.
- General arbitrary-l abstractions before f correctness is locked.
- Performance work before reference validation.

Research flag: standard patterns for cleanup; deeper research only if a deferred feature is pulled into scope.

## Requirement Guidance

- Define acceptance around layered correctness, not just end-to-end SCF. The first acceptance gate should be parser channel reconstruction at source grid points.
- Require a canonical channel-order contract and a canonical AO-order contract before implementation of f matrix blocks begins.
- Require existing simple-format behavior to remain unchanged after every phase.
- Require explicit unsupported errors for any f combination not validated in the current milestone, especially batch, force, stress, spin, DFTB3, SEDACS, ML-SK, and MD paths.
- Require reference simulation acceptance to specify exact geometries, parameter files, observables, units, tolerances, and whether forces/stress are included.
- Prefer small testable helpers for channel lookup, AO labels, and f block assembly; avoid public API changes unless a user-facing option is genuinely necessary.

Confidence assessment:

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Repo-grounded; all research points to extending the current PyTorch code and pytest fixtures. |
| Features | HIGH | Table-stakes and dependencies are clear from existing code paths and project goals. |
| Architecture | HIGH | Component responsibilities align across PROJECT, codebase docs, and research files. |
| Pitfalls | HIGH | Most risks are codebase-specific and tied to known hard-coded assumptions. |
| Scientific references | MEDIUM | DFTB+/SKF format semantics are clear enough for staging, but final f formulas, ordering, and paper tolerances need validation. |
| Final simulation acceptance | MEDIUM | The project has f fixtures, but exact target observables and tolerances still need to be specified. |

Gaps to address during planning:

- Exact f AO ordering and real-harmonic convention to use in `_slater_koster_pair.py`.
- Exact reference-paper geometries, settings, observables, and tolerances.
- Whether force/stress support is in scope for the first f milestone or explicitly deferred.
- Whether batch f support is required or should raise a clear unsupported error.
- How shell-resolved optional physics should behave for f systems before full validation.

## Major Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Wrong 20-to-40 or f channel mapping | Shape-correct but scientifically wrong H/S matrices | Named channel constants, per-channel spline reconstruction, simple and extended fixture tests. |
| Missing f metadata in Constants/Structure | Wrong onsite diagonal, density, shell charges, and SCF behavior | Extend parser return, Constants storage, Structure templates, `D0`, and shell arrays together. |
| H0/S masks omit `n_orb == 16` pairs | f interactions silently absent | Add explicit `spdf` pair classes and assert every neighbor pair is routed exactly once. |
| Incorrect f angular formulas or derivatives | Wrong energies, forces, and stress despite finite tensors | Axis-aligned tests, synthetic one-channel tests, atom-order swap tests, and finite-difference derivative checks. |
| AO ordering drift | Reference mismatch that is hard to diagnose | Single AO-order source of truth used by Structure, density, and SK scatter code. |
| Simple-format regression | Existing users get changed numerical results | Run simple parser, CH4 SCF/force, and tutorial-style regressions after each phase. |
| Over-scoped prototype | Slow delivery and unclear regression source | Defer batch, force/stress parity, SEDACS, ML-SK, performance, and broad abstractions unless required by validation. |

## Sources

- `.planning/PROJECT.md`
- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/TESTING.md`
- `.planning/codebase/CONCERNS.md`
- `.planning/codebase/CONVENTIONS.md`
- `src/dftorch/_bond_integral.py`
- `src/dftorch/Constants.py`
- `src/dftorch/Structure.py`
- `src/dftorch/_atomic_density_matrix.py`
- `src/dftorch/_h0ands.py`
- `src/dftorch/_slater_koster_pair.py`
- `src/dftorch/_stress.py`
- `src/dftorch/_coulomb_matrix.py`
- `src/dftorch/ESDriver.py`
- `tests/f_orbital_data/`
- `tests/data_skf_mio-1-1/`
- `tests/test_scf.py`
- `experiments/1_tutorial.ipynb`
- DFTB.org parameter introduction: https://dftb.org/parameters/introduction.html
- DFTB+ Recipes first calculation: https://dftbplus-recipes.readthedocs.io/en/stable/basics/firstcalc.html
- Slater-Koster file format technical guide mirror: https://studylib.net/doc/26065153/slakoformat
- DFTB+ user-list format discussion: https://mailman.zfn.uni-bremen.de/pipermail/dftb-plus-user/2019/002918.html
- ASE DFTB calculator source docs: https://ase-lib.org/_modules/ase/calculators/dftb.html
