# Feature Landscape: f-Orbital DFTB/SKF Support

**Domain:** Brownfield f-orbital Slater-Koster support in DFTorch
**Researched:** 2026-07-17
**Scope:** Features and staging for requirements definition
**Overall confidence:** HIGH for codebase-derived staging; MEDIUM for final reference-paper acceptance until exact target observables/tolerances are specified.

## Executive Takeaway

The valid f-orbital implementation is not "parse a wider SKF file" by itself. The table-stakes feature set is a full data-path extension: 40-channel SKF parsing, universal channel normalization, f-shell basis metadata, `spdf` orbital indexing, f angular Slater-Koster block assembly, and regression tests that prove both f fixtures and existing simple-format workflows still behave correctly.

Stage this as a prototype-first sequence. The prototype should start with `_bond_integral.py` and prove that every relevant source channel in `tests/f_orbital_data/` can be reconstructed by evaluating generated cubic splines. Only after that should requirements move into `Constants`, `Structure`, `_h0ands.py`, and `_slater_koster_pair.py`, because every downstream feature depends on stable channel order and orbital-count semantics.

Cleanup should come after a working simulation path exists. The important cleanup is not broad abstraction; it is removing hard-coded 1/4/9 assumptions, documenting channel order and basis order, adding precise errors, and consolidating duplicated single/batch logic where the f path actually touches it.

## Table Stakes

Features users and maintainers should treat as required for a scientifically valid f-orbital milestone. Missing any of these means the implementation is incomplete or unsafe.

| Feature | Stage | Why Expected | Complexity | Notes |
|---------|-------|--------------|------------|-------|
| Parse extended 40-column SKF electronic tables | Prototype | f fixtures such as Eu-containing files include f-shell H/S channels that cannot fit the old 20-column layout. | Medium | Must handle the leading `@` extended marker, Fortran repetition tokens, comments, compact filenames like `EuGa.skf`, and conventional dashed names. |
| Preserve simple 20-column SKF parsing | Prototype | Existing s/p and s/p/d parameter sets must continue to load without changing user behavior. | Medium | Convert simple rows into the same 40-channel internal representation with f-related channels zero-filled. |
| Universal internal channel order | Prototype | Downstream code should consume one canonical layout regardless of raw SKF format. | Medium | Make `_CHANNELS` the single documented source of truth; tests should verify simple channels land in the intended extended positions. |
| Per-channel spline reconstruction tests | Prototype | The first checkpoint is spline validation, and all downstream calculations depend on correct radial interpolation. | Medium | For every f fixture pair, evaluate cubic coefficients at tabulated grid points and compare against source SKF channel values with tight floating-point tolerances. |
| Robust pair-file resolution | Prototype | The current fixture set uses compact names (`EuN.skf`, `NN.skf`) while existing datasets use dashed names (`C-N.skf`). | Low | Requirement should include clear missing-file errors naming both attempted forms. |
| Homonuclear onsite/header parsing for f shells | Prototype | Element-level orbital counts, onsite energies, Hubbard values, and reference occupations come from homonuclear SKF headers. | Medium | Current `Constants` only exposes s/p/d fields; requirements must add f occupancy/Hubbard/onsite support or explicitly define temporary prototype handling. |
| Nested basis validation | Prototype | DFTorch's AO layout assumes contiguous shell expansion: s, sp, spd, spdf. | Low | Reject unsupported skipped-shell files early with explicit `ValueError`; do not let shape bugs appear in H/S assembly. |
| Constants storage for f orbital metadata | Prototype | `Constants` is the project-wide parameter database consumed by `Structure`, `ESDriver`, and kernels. | Medium | Add `n_f`, `Uf`, `Ef`, and `shell_dim`/basis metadata capable of representing 16 orbitals for `spdf`. |
| Structure orbital indexing for 16-orbital atoms | Prototype | H0/S matrix dimensions and AO offsets must include 7 f orbitals per atom. | High | Audit `Structure` assumptions around `n_orb`, shell counts, diagonal onsite construction, occupations, and Mulliken charge layout. |
| H0/S pair classification for f-capable atoms | Prototype | Current assembly masks are based on orbital counts 1, 4, and 9; f atoms need 16-orbital participation. | High | Add clear pair classes for H/X/Y/Z or replace with basis-dimension-aware handling. Prototype may be explicit rather than abstract. |
| f angular Slater-Koster formulas | Prototype | Without angular transforms for s-f, p-f, d-f, and f-f couplings, parsed f channels cannot affect Hamiltonian/overlap matrices correctly. | High | Implement and test formulas for both Hamiltonian and overlap paths, including derivatives used by forces/stress if those paths are in scope. |
| H/S symmetry and direction handling | Prototype | Pair direction conventions affect signs and transposes for mixed shells. | High | Requirements should include cross-checks for `IJ` versus `JI` pair use, especially mixed Eu-Ga/Eu-N/Ga-Eu/N-Eu files. |
| Preserve existing simple-format simulations | Prototype and Cleanup | Existing tutorial and smoke-test behavior is an explicit project constraint. | Medium | Add regression tests around current CH4/simple SKF path; f changes must not alter s/p/d numerical results beyond tolerance. |
| End-to-end f simulation smoke path | Prototype | Final milestone requires full simulation through `Constants` -> `Structure` -> `ESDriver`. | High | Start with finite H0/S/SCF/energy assertions on a minimal f fixture system before matching paper-level results. |
| Reference-paper reproduction tests | Cleanup/Final | Scientific validity is measured against the reference simulations represented by `tests/f_orbital_data/`. | High | Needs exact geometry, settings, observables, and tolerances documented before it can be a hard acceptance gate. |
| Clear unsupported-mode errors | Prototype and Cleanup | Batch, PME, stress, forces, SEDACS, ML-SK, and optional corrections may not all be f-ready immediately. | Medium | Prototype can raise precise errors for unsupported f combinations rather than silently producing wrong tensors. |
| Documentation of channel and basis conventions | Cleanup | Human troubleshooting is a project requirement. | Low | Document 40-channel order, shell order, units, pair direction, and zero-fill semantics near the parser/tests. |

## Prototype-Stage Requirements

Build these first. They form the narrowest credible prototype and directly support the first checkpoint.

| Requirement | Acceptance Signal | Complexity | Dependencies |
|-------------|-------------------|------------|--------------|
| Normalize simple and extended SKF rows to 40 channels | Unit tests pass for both `tests/data_skf_mio-1-1/` and `tests/f_orbital_data/`. | Medium | Pair-file resolution, `_CHANNELS` contract |
| Validate f fixture parsing without running SCF | All files in `tests/f_orbital_data/` parse into finite `R`, channel, repulsive, close-exp, and metadata tensors. | Medium | Extended marker parsing, header parsing |
| Reconstruct source electronic table values from splines | Cubic spline evaluation at source grid points matches selected/all H/S channels from source rows. | Medium | Normalized rows, `cubic_spline_coeffs` |
| Surface f basis metadata in `Constants` | `Constants` exposes correct 16-orbital counts and f onsite/Hubbard/occupation fields for Eu while simple elements remain unchanged. | Medium | Homonuclear header parse |
| Run a minimal f Constants/Structure construction | A small Eu/Ga/N fixture or synthetic geometry constructs without dimension errors. | Medium | Constants metadata, Structure indexing |
| Add explicit unsupported errors for unimplemented f matrix assembly paths | If f atoms reach unsupported H0/S code before formulas are complete, the error says which pair class/path is missing. | Low | f detection in constants/structure |
| Keep existing smoke tests green | Current import, IO, neighbor-list, SCF, and force smoke tests pass. | Medium | Simple-format preservation |

## Cleanup-Stage Requirements

Do these after a working f path exists, unless a cleanup item becomes necessary to unblock correctness.

| Requirement | Value | Complexity | Dependencies |
|-------------|-------|------------|--------------|
| Replace scattered hard-coded orbital-count masks | Reduces bugs as 1/4/9 grows to 16 and makes future shell support less fragile. | High | Working explicit f prototype |
| Consolidate H/S channel lookup helpers | Keeps Hamiltonian and overlap channel indexing consistent after expanding from 20 to 40 channels. | Medium | f angular formulas |
| Add targeted negative-path parser tests | Prevents invalid skipped-shell or malformed SKF files from failing late in numerical kernels. | Low | Parser validation behavior |
| Document f fixture assumptions and paper-reference mapping | Makes the final acceptance tests reproducible by humans. | Medium | Reference simulation details |
| Decide batch f support policy | Avoids silent divergence between `ESDriver` and `ESDriverBatch`. | Medium | Single-system f path |
| Decide force/stress f support policy | Clarifies whether f support means energy/SCF only or includes derivatives. | High | Angular derivative implementation |
| Refactor temporary prototype branches | Keeps troubleshootability without leaving duplicated fragile code in the main path. | Medium | Passing f simulation tests |

## Differentiators

These are valuable but should not block the first working prototype unless they become necessary for reference reproduction.

| Feature | Value Proposition | Complexity | Stage Recommendation |
|---------|-------------------|------------|----------------------|
| Full force and stress support for f systems | Enables MD, geometry optimization, stress calculations, and more complete scientific workflows. | High | Defer until energy/SCF path is validated unless reference reproduction needs forces. |
| Batch f simulations | Keeps parity with existing single/batch architecture for high-throughput workflows. | High | Defer; add explicit unsupported error first. |
| SEDACS/distributed f compatibility | Supports large-scale simulations once f matrices are stable. | High | Defer well beyond prototype. |
| ML-SK f-channel compatibility | Lets the ML Slater-Koster replacement cover f channels later. | High | Anti-feature for this milestone unless already required by paper reproduction. |
| Performance optimization for 16-orbital blocks | Reduces cost from larger H/S blocks. | Medium/High | Defer until correctness and regressions are locked. |
| Developer diagnostic script for SKF channel inspection | Speeds human troubleshooting of channel order and spline reconstruction. | Low | Useful in prototype if kept test-backed and not treated as public API. |
| Formula-level golden tests for selected f angular couplings | Catches sign and direction-cosine mistakes earlier than full SCF tests. | Medium | Add alongside angular formula implementation. |

## Anti-Features

Things to explicitly avoid in the milestone requirements.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Rewriting `ESDriver` or the whole kernel stack | The project goal is a quick, brownfield prototype; broad rewrites increase regression risk. | Extend the current `Constants` -> `Structure` -> `ESDriver` path with localized changes. |
| Designing a perfect general arbitrary-l shell abstraction first | It delays the f prototype and may not match the current code's stateful tensor style. | Implement clear `spdf` support, then refactor once tests prove behavior. |
| Changing public API names or tutorial workflow | Existing users and notebooks are regression targets. | Keep user-facing construction unchanged; add internal metadata only. |
| Optimizing f kernels before correctness | Faster wrong f blocks are not useful and obscure scientific validation. | Prioritize spline, H/S, SCF, and reference checks. |
| Silently zeroing unsupported f interactions downstream | This can produce plausible but scientifically invalid simulations. | Zero-fill only when converting simple SKF formats; otherwise raise explicit errors for missing formulas/channels. |
| Supporting skipped-shell bases now | Current AO ordering assumes contiguous shells and skipped shells complicate offsets. | Require nested s/p/d/f shells for f elements. |
| Making f support depend on optional GPU/Triton/compile paths | Prototype tests should be CPU-readable and deterministic. | Follow current pytest pattern that disables Torch compile features. |
| Treating scripts/notebooks as the only validation | Scripts are useful for inspection but weak as acceptance criteria. | Put source reconstruction and smoke paths in pytest. |
| Expanding unrelated physics corrections | GBSA, D3, DFTB3, PME, SEDACS, and MD changes can balloon scope. | Touch them only when they block the f simulation path or need explicit unsupported errors. |

## Feature Dependencies

```text
Pair-file resolution
  -> SKF row normalization to 40 channels
  -> Per-channel spline coefficient generation
  -> Spline reconstruction regression tests
  -> Constants f metadata
  -> Structure 16-orbital AO indexing
  -> H0/S f pair classification
  -> f angular Slater-Koster formulas
  -> Single-system f H0/S construction
  -> SCF/energy smoke test
  -> Reference-paper reproduction

Simple 20-column normalization
  -> Existing simple-format regression tests
  -> Confidence that f support did not break current users

Unsupported-mode detection
  -> Safe prototype release
  -> Later batch/force/stress/SEDACS cleanup decisions
```

## MVP Recommendation

Prioritize:

1. Parser and channel normalization in `_bond_integral.py`, including compact/dashed filename handling and nested-shell validation.
2. Spline reconstruction tests against `tests/f_orbital_data/` plus regression coverage for existing simple SKF fixtures.
3. `Constants` and `Structure` metadata changes sufficient to construct f systems and expose correct orbital counts/onsite/Hubbard/occupation values.
4. Explicit unsupported errors at H0/S if formulas are not yet implemented, then replace those errors with f angular matrix assembly.
5. Minimal single-system f SCF/energy smoke test, followed by reference-paper reproduction once target observables are specified.

Defer:

- Batch f support until the single-system path is correct.
- Force/stress/MD support until the derivative requirements are explicit.
- SEDACS, ML-SK, and broad performance work until after reference simulation correctness.

## Suggested Requirement Buckets

### Phase 1: Parser and Spline Foundation

- Load simple and extended SKF files into one 40-channel representation.
- Parse all `tests/f_orbital_data/` files without shape or metadata failures.
- Verify spline coefficients reproduce source table values.
- Preserve simple-format parser behavior and existing smoke tests.

### Phase 2: Constants and Basis Metadata

- Add f orbital count, f occupancy, f onsite energy, and f Hubbard storage.
- Ensure Eu/Ga/N fixture systems construct through `Constants` and `Structure`.
- Add clear errors for unsupported f downstream paths not implemented yet.

### Phase 3: H0/S f Matrix Assembly

- Add `spdf` pair classification or basis-aware block selection.
- Implement f angular Slater-Koster formulas for required pair classes.
- Validate H0/S shapes, symmetry, finite values, and selected golden couplings.

### Phase 4: Simulation Validation

- Run minimal f SCF/energy smoke tests.
- Preserve simple tutorial/reference outputs.
- Reproduce reference-paper cases from `tests/f_orbital_data/` with documented tolerances.

### Phase 5: Cleanup and Parity Decisions

- Refactor temporary hard-coded branches only after tests pass.
- Decide and document batch, force, stress, MD, SEDACS, and ML-SK support status.
- Add documentation for f channel order, basis order, units, and limitations.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table-stakes categories | HIGH | Derived from project requirements, codebase maps, parser/kernel inspection, and f fixture structure. |
| Prototype staging | HIGH | The dependency chain is explicit: parser/channel correctness must precede constants and H/S assembly. |
| Cleanup staging | HIGH | Matches project constraint to prioritize a working prototype, then clean hard-coded assumptions. |
| Differentiators | MEDIUM | Force/stress/batch/SEDACS value is clear from architecture, but exact milestone need depends on reference-paper reproduction scope. |
| Reference-paper final tests | MEDIUM | Required by project goals, but exact geometries, observables, and tolerances still need specification. |

## Sources

- `.planning/PROJECT.md`
- `.planning/codebase/ARCHITECTURE.md`
- `.planning/codebase/STRUCTURE.md`
- `.planning/codebase/TESTING.md`
- `src/dftorch/_bond_integral.py`
- `src/dftorch/Constants.py`
- `src/dftorch/_h0ands.py`
- `src/dftorch/_slater_koster_pair.py`
- `tests/f_orbital_data/`
