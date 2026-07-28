# Requirements: DFTorch F-Orbital Support

**Defined:** 2026-07-20
**Core Value:** DFTorch can run scientifically valid f-orbital DFTB simulations without changing numerical results for existing simple-format calculations.

## v1 Requirements

Requirements for the initial f-orbital support milestone. Each maps to roadmap phases.

### SKF Parsing and Channel Canonicalization

- [x] **SKF-01**: DFTorch can read simple-format `.skf` files with the existing 20-column electronic table layout.
- [x] **SKF-02**: DFTorch can read extended-format `.skf` files with the wider f-orbital electronic table layout.
- [x] **SKF-03**: Simple-format and extended-format files are normalized to one documented internal 40-channel representation before downstream code consumes them.
- [x] **SKF-04**: Simple-format files zero-fill f-only channels while preserving existing s/p and s/p/d channel values.
- [x] **SKF-05**: Compact pair filenames such as `EuN.skf` and dashed pair filenames such as `Eu-N.skf` resolve consistently with clear missing-file errors.
- [x] **SKF-06**: Homonuclear SKF parsing captures f-shell onsite energy, Hubbard value, and reference occupation when present.
- [x] **SKF-07**: Parser validation rejects unsupported skipped-shell basis layouts with actionable errors.

### Spline and Fixture Validation

- [x] **SPL-01**: Cubic spline coefficients generated from `tests/f_orbital_data/` reproduce source electronic table values at original grid points.
- [x] **SPL-02**: Cubic spline coefficients generated from existing simple-format fixtures reproduce legacy source channel values at original grid points.
- [x] **SPL-03**: Parser tests cover representative f fixture pairs, including homonuclear and heteronuclear files.
- [x] **SPL-04**: Parser and spline tests run through pytest without optional GPU, Triton, DFTB+, or ASE dependencies.

### Constants and Basis Metadata

- [x] **BAS-01**: `Constants` stores f-orbital counts, onsite energies, Hubbard values, and reference occupations.
- [x] **BAS-02**: `Constants` exposes correct `n_orb == 16` and `max_ang == 4` metadata for spdf elements.
- [x] **BAS-03**: `Constants` preserves existing s-only, sp, and spd metadata for simple-format datasets.
- [x] **BAS-04**: Shared shell dimensions support s, p, d, and f shells as `[0, 1, 3, 5, 7]`.
- [x] **BAS-05**: `Structure` constructs 16-AO atoms with consistent AO ranges, onsite diagonal entries, shell ranges, shell labels, and reference density.
- [x] **BAS-06**: f-free systems keep existing `HDIM`, onsite diagonal, shell indexing, and reference density behavior.

### H0/S Routing and Angular Slater-Koster Blocks

- [ ] **HSK-01**: H0/S assembly routes all neighbor pairs involving 16-orbital atoms instead of silently dropping f pairs.
- [ ] **HSK-02**: f-aware pair routing preserves current H/X/Y behavior for 1-, 4-, and 9-orbital atoms.
- [ ] **HSK-03**: Slater-Koster angular transforms are implemented for s-f and f-s couplings.
- [ ] **HSK-04**: Slater-Koster angular transforms are implemented for p-f and f-p couplings.
- [ ] **HSK-05**: Slater-Koster angular transforms are implemented for d-f and f-d couplings.
- [ ] **HSK-06**: Slater-Koster angular transforms are implemented for f-f couplings.
- [ ] **HSK-07**: H0 and S matrices for f-containing systems are finite, correctly shaped, and symmetric after assembly.
- [ ] **HSK-08**: Direction, atom-order, and AO-order conventions are covered by targeted formula or matrix tests.

### SCF and Full Simulation Validation

- [ ] **SIM-01**: A minimal f-containing system reaches H0/S construction without shape or routing failures.
- [ ] **SIM-02**: A minimal f-containing system reaches closed-shell SCF/energy calculation or raises an explicit unsupported-mode error for the unimplemented blocker.
- [ ] **SIM-03**: Shell-resolved charge, Hubbard, and Coulomb data include f-shell dimensions where the supported SCF path requires them.
- [ ] **SIM-04**: Reference-paper simulation cases represented by `tests/f_orbital_data/` are documented with geometry, parameters, observables, units, and tolerances.
- [ ] **SIM-05**: Supported f-orbital reference simulations reproduce the selected reference-paper observables within agreed tolerances.

### Regression Safety

- [ ] **REG-01**: Existing pytest import, IO, neighbor-list, SCF, and force smoke tests pass after each f-orbital phase.
- [ ] **REG-02**: Existing simple-format calculations in `experiments/1_tutorial.ipynb` remain scientifically unchanged within documented tolerances.
- [ ] **REG-03**: f-orbital changes do not require public API changes for existing simple-format users.
- [ ] **REG-04**: Unsupported f combinations fail with explicit errors rather than silently producing zero or malformed matrices.

### Prototype Cleanup and Support Policy

- [ ] **CLN-01**: Temporary prototype branches are documented where they are introduced.
- [ ] **CLN-02**: Channel lookup and AO ordering are centralized enough for human troubleshooting.
- [ ] **CLN-03**: Remaining hard-coded `1/4/9` orbital assumptions are audited and either extended to `16` or guarded.
- [ ] **CLN-04**: Batch, force, stress, MD, SEDACS, and ML-SK f-orbital support status is explicitly documented as supported, deferred, or unsupported.
- [ ] **CLN-05**: Cleanup preserves the working prototype and simple-format regression tests.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap unless needed for reference-paper reproduction.

### Extended Physics and Performance

- **PHY-01**: f-orbital force derivatives are validated against finite-difference checks.
- **PHY-02**: f-orbital stress contributions are validated against finite-difference or reference checks.
- **PHY-03**: f-orbital molecular dynamics workflows run through `MDXL` with validated forces.
- **PHY-04**: f-orbital batch calculations are supported through `StructureBatch` and `ESDriverBatch`.
- **PHY-05**: f-orbital SEDACS workflows are supported for distributed simulations.
- **PHY-06**: ML-SK models can consume or predict f-orbital channel data.
- **PHY-07**: 16-orbital H0/S assembly is profiled and optimized after correctness is locked.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Rewriting the whole driver/kernel architecture | The project needs a working brownfield prototype quickly; broad rewrites increase regression risk. |
| General arbitrary-angular-momentum abstractions | spdf support is the target; generalized arbitrary-l support can wait until f correctness is proven. |
| Public API redesign | Existing simple-format users and tutorials must remain stable. |
| GPU/Triton-specific f validation as a v1 requirement | CPU pytest validation is simpler, deterministic, and enough to prove the core math first. |
| Runtime dependency on DFTB+ or ASE | Reference values may be generated offline, but normal tests should use stored fixtures and values. |
| Broad PME, GBSA, D3, SEDACS, or ML-SK expansion | These are touched only if they block the single-system f simulation path or need explicit unsupported errors. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SKF-01 | Phase 1 | Complete |
| SKF-02 | Phase 1 | Complete |
| SKF-03 | Phase 1 | Complete |
| SKF-04 | Phase 1 | Complete |
| SKF-05 | Phase 1 | Complete |
| SKF-06 | Phase 1 | Complete |
| SKF-07 | Phase 1 | Complete |
| SPL-01 | Phase 1 | Complete |
| SPL-02 | Phase 1 | Complete |
| SPL-03 | Phase 1 | Complete |
| SPL-04 | Phase 1 | Complete |
| BAS-01 | Phase 2 | Complete |
| BAS-02 | Phase 2 | Complete |
| BAS-03 | Phase 2 | Complete |
| BAS-04 | Phase 2 | Complete |
| BAS-05 | Phase 2 | Complete |
| BAS-06 | Phase 2 | Complete |
| HSK-01 | Phase 3 | Pending |
| HSK-02 | Phase 3 | Pending |
| HSK-03 | Phase 3 | Pending |
| HSK-04 | Phase 3 | Pending |
| HSK-05 | Phase 3 | Pending |
| HSK-06 | Phase 3 | Pending |
| HSK-07 | Phase 3 | Pending |
| HSK-08 | Phase 3 | Pending |
| SIM-01 | Phase 4 | Pending |
| SIM-02 | Phase 4 | Pending |
| SIM-03 | Phase 4 | Pending |
| SIM-04 | Phase 4 | Pending |
| SIM-05 | Phase 4 | Pending |
| REG-01 | Phase 5 | Pending |
| REG-02 | Phase 5 | Pending |
| REG-03 | Phase 5 | Pending |
| REG-04 | Phase 5 | Pending |
| CLN-01 | Phase 5 | Pending |
| CLN-02 | Phase 5 | Pending |
| CLN-03 | Phase 5 | Pending |
| CLN-04 | Phase 5 | Pending |
| CLN-05 | Phase 5 | Pending |

**Coverage:**

- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-07-20*
*Last updated: 2026-07-20 after roadmap creation*
