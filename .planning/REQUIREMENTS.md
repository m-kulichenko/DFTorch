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

- [x] **HSK-01**: H0/S assembly routes all neighbor pairs involving 16-orbital atoms instead of silently dropping f pairs.
- [x] **HSK-02**: f-aware pair routing preserves current H/X/Y behavior for 1-, 4-, and 9-orbital atoms.
- [x] **HSK-03**: Slater-Koster angular transforms are implemented for s-f and f-s couplings.
- [x] **HSK-04**: Slater-Koster angular transforms are implemented for p-f and f-p couplings.
- [x] **HSK-05**: Slater-Koster angular transforms are implemented for d-f and f-d couplings.
- [x] **HSK-06**: Slater-Koster angular transforms are implemented for f-f couplings.
- [x] **HSK-07**: H0 and S matrices for f-containing systems are finite, correctly shaped, and symmetric after assembly.
- [x] **HSK-08**: Direction, atom-order, and AO-order conventions are covered by targeted formula or matrix tests.

### SCF and Full Simulation Validation

- [x] **SIM-01**: A minimal f-containing system reaches H0/S construction without shape or routing failures.
- [x] **SIM-02**: A minimal f-containing system reaches closed-shell SCF/energy calculation or raises an explicit unsupported-mode error for the unimplemented blocker.
- [x] **SIM-03**: Shell-resolved charge, Hubbard, and Coulomb data include f-shell dimensions where the supported SCF path requires them.
- [x] **SIM-04**: The isolated Eu-N diatomic validation case represented by `tests/f_orbital_data/` is documented with geometry, SKF parameter files, observables (energy vs. interatomic separation), units, and tolerances.
- [x] **SIM-05**: The supported f-orbital Eu-N energy scan locates an energy minimum within the agreed loose sanity band (~10-20%) of the known Eu-N separation.

### Regression Safety

- [x] **REG-01**: Existing pytest import, IO, neighbor-list, SCF, and force smoke tests pass after each f-orbital phase.
- [x] **REG-02**: Existing simple-format calculations in `experiments/1_tutorial.ipynb` remain scientifically unchanged within documented tolerances.
- [ ] **REG-03**: f-orbital changes do not require public API changes for existing simple-format users.
- [ ] **REG-04**: Unsupported f combinations fail with explicit errors rather than silently producing zero or malformed matrices.
- [ ] **REG-05**: Loading an `SKFPATH` whose SKF files do not share one radial grid step fails with an explicit error naming the offending files, instead of silently evaluating some pairs against another pair's grid. (Guard for the shared-`R_orb` hazard: `_bond_integral.get_skf_tensors` exports one global `R_orb` — the longest grid seen — and `_h0ands` derives `idx`/`dx` from it for every pair type. Currently inert because all nine `tests/f_orbital_data` fixtures share a 0.04 Bohr step, but the f dataset is 0.04 while `mio-1-1`/`3ob-3-1`/`pbc-0-3`/`trans3d-0-1` are all 0.02, so any mixed load is wrong by a factor of two. Same-step/different-length is benign and must not be rejected.)
- [x] **REG-06**: Slater-Koster radial lookup uses each pair's own grid, so an `SKFPATH` mixing radial grid steps produces correct interpolation rather than only a refusal. (The real fix behind REG-05's guard; changes the `coeffs_tensor` / `R_orb` interface that the ML-SK and stress paths also consume.)

### Prototype Cleanup and Support Policy

- [ ] **CLN-01**: Temporary prototype branches are documented where they are introduced.
- [ ] **CLN-02**: Channel lookup and AO ordering are centralized enough for human troubleshooting.
- [ ] **CLN-03**: Remaining hard-coded `1/4/9` orbital assumptions are audited and either extended to `16` or guarded.
- [ ] **CLN-04**: Batch, force, stress, MD, SEDACS, and ML-SK f-orbital support status is explicitly documented as supported, deferred, or unsupported.
- [ ] **CLN-05**: Cleanup preserves the working prototype and simple-format regression tests.

### Self-Consistency and f Coulomb Blocks

Promoted into the roadmap 2026-07-29 (Phase 6).

- [ ] **SCC-01**: f-containing systems run a true self-consistent charge loop to convergence, not a single diagonalization. Non-convergence warns and returns the last iterate with a convergence flag (D-13), rather than raising.
- [ ] **SCC-02**: The seven f angular blocks of the shell-resolved Coulomb matrix (s-f, f-s, p-f, f-p, d-f, f-d, f-f) are implemented and validated, replacing `FShellResolvedCoulombUnsupportedError`. Depends on SCC-01: the matrix is `(n_shells, n_shells)` while `energy()`/`SCFx` consume `(Nats, Nats)` per-atom charges, so nothing can validate these values until shell-resolved charges flow through SCF.
- [ ] **SCC-03**: The shell-resolved f plumbing validated but unconsumed in Phase 4 (D-14) is actually consumed by the self-consistent path.

### f Derivatives, Forces, Stress, and Dynamics

Promoted from v2 into the roadmap 2026-07-29 (Phases 7-9).

- [ ] **DRV-01**: f angular derivative formulas are implemented so `dH0`/`dS` are non-zero in f blocks, and `F_ANGULAR_DERIVATIVES_AVAILABLE` is `True`.
- [ ] **DRV-02**: The `FDerivativeUnsupportedError` guards in `ESDriver.calc_forces`, `ESDriverBatch.calc_forces`, and `_stress.py` are removed or narrowed to whatever genuinely remains unsupported, rather than blanket-refusing every 16-orbital system.
- [ ] **PHY-01**: f-orbital force derivatives are validated against finite-difference checks.
- [ ] **PHY-02**: f-orbital stress contributions are validated against finite-difference or reference checks.
- [ ] **PHY-03**: f-orbital molecular dynamics workflows run through `MDXL` with validated forces.
- [ ] **PHY-04**: f-orbital batch calculations are supported through `StructureBatch` and `ESDriverBatch`, replacing the `FAngularFormulaSourceError` raised by the batched H0/S path. **On the critical path for PHY-03, not conditional** — `MDXL.__init__` and `MDXLBatch.__init__` both require an `ESDriverBatch` (`MD.py:38`, `MD.py:1216`), so MD cannot bypass the batch path. Verified 2026-07-29.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap unless needed.

### Extended Physics and Performance

- **SPN-01**: Spin-polarized / collinear-spin support for open-shell 4f, replacing `FSpinPolarizationUnsupportedError`. Physically the correct treatment for rare earths (Eu 4f7 is genuinely open-shell); deferred by D-12.
- **PHY-05**: f-orbital SEDACS workflows are supported for distributed simulations.
- **PHY-06**: ML-SK models can consume or predict f-orbital channel data.
- **PHY-07**: 16-orbital H0/S assembly is profiled and optimized after correctness is locked.
- **PHY-08**: Ga-containing and periodic EuN crystal cases are supported (pulls in k-points and Ewald/PME). Deferred pending confirmation that they need f-specific changes at all.

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
| HSK-01 | Phase 3 | Complete |
| HSK-02 | Phase 3 | Complete |
| HSK-03 | Phase 3 | Complete |
| HSK-04 | Phase 3 | Complete |
| HSK-05 | Phase 3 | Complete |
| HSK-06 | Phase 3 | Complete |
| HSK-07 | Phase 3 | Complete |
| HSK-08 | Phase 3 | Complete |
| SIM-01 | Phase 4 | Complete |
| SIM-02 | Phase 4 | Complete |
| SIM-03 | Phase 4 | Complete |
| SIM-04 | Phase 4 | Complete |
| SIM-05 | Phase 4 | Complete |
| REG-01 | Phase 5 | Complete |
| REG-02 | Phase 5 | Complete |
| REG-03 | Phase 5 | Pending |
| REG-04 | Phase 5 | Pending |
| REG-05 | Phase 5 | Pending |
| REG-06 | Phase 5 | Complete |
| CLN-01 | Phase 5 | Pending |
| CLN-02 | Phase 5 | Pending |
| CLN-03 | Phase 5 | Pending |
| CLN-04 | Phase 5 | Pending |
| CLN-05 | Phase 5 | Pending |
| SCC-01 | Phase 6 | Pending |
| SCC-02 | Phase 6 | Pending |
| SCC-03 | Phase 6 | Pending |
| DRV-01 | Phase 7 | Pending |
| DRV-02 | Phase 7 | Pending |
| PHY-01 | Phase 8 | Pending |
| PHY-02 | Phase 8 | Pending |
| PHY-03 | Phase 9 | Pending |
| PHY-04 | Phase 9 | Pending |

**Coverage:**

- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-07-20*
*Last updated: 2026-07-20 after roadmap creation*
