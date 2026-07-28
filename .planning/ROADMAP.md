# Roadmap: DFTorch F-Orbital Support

## Overview

This milestone extends DFTorch through the technical dependency chain required for scientifically valid f-orbital DFTB: normalize simple and extended SKF data at the parser boundary, carry spdf basis metadata through constants and structures, assemble f-aware H0/S matrices with Slater-Koster angular blocks, validate SCF/reference simulations, then lock regression behavior and support policy before broader physics parity work.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: SKF Canonicalization and Spline Validation** - Parser and spline paths prove simple and extended SKF files converge to one 40-channel representation. (completed 2026-07-21)
- [x] **Phase 2: Constants and Structure Basis Metadata** - Constants and Structure expose consistent spdf shell, orbital, onsite, Hubbard, and density metadata. (completed 2026-07-27)
- [ ] **Phase 3: H0/S Routing and f Angular Blocks** - H0/S assembly routes 16-orbital atom pairs and writes tested f-containing Slater-Koster blocks.
- [ ] **Phase 4: SCF and Reference Simulation Validation** - f-containing single-system simulations reach the supported SCF path and are compared to documented reference observables.
- [ ] **Phase 5: Regression Safety and Support Policy Cleanup** - Existing simple-format behavior, unsupported f modes, and prototype cleanup are locked behind tests and explicit policy.

## Phase Details

### Phase 1: SKF Canonicalization and Spline Validation

**Goal**: DFTorch can parse simple and extended SKF files into a shared, verifiable channel representation before downstream code consumes them.
**Depends on**: Nothing (first phase)
**Requirements**: SKF-01, SKF-02, SKF-03, SKF-04, SKF-05, SKF-06, SKF-07, SPL-01, SPL-02, SPL-03, SPL-04
**Success Criteria** (what must be TRUE):

  1. User can load simple 20-column and extended f-orbital SKF files without changing the downstream parser contract.
  2. User can inspect one documented 40-channel internal representation for both SKF formats, with simple-format f-only channels zero-filled.
  3. User can resolve compact and dashed pair filenames consistently, and missing or unsupported basis layouts fail with actionable errors.
  4. User can run CPU pytest parser and spline checks that reconstruct original f-fixture and simple-fixture electronic table values at source grid points.
  5. Homonuclear f metadata is captured from SKF files for later constants and structure use.

**Plans**: 1/1 plans executed
Plans:

- [x] 01-01-PLAN.md — Harden SKF parser canonicalization and spline validation in the standalone Phase 1 harness.

### Phase 2: Constants and Structure Basis Metadata

**Goal**: DFTorch calculation state represents spdf basis data consistently for f-containing and f-free systems.
**Depends on**: Phase 1
**Requirements**: BAS-01, BAS-02, BAS-03, BAS-04, BAS-05, BAS-06
**Success Criteria** (what must be TRUE):

  1. User can construct `Constants` for spdf elements and observe f-shell counts, onsite energies, Hubbard values, reference occupations, `n_orb == 16`, and `max_ang == 4`.
  2. User can construct `Structure` for a 16-AO atom and observe consistent AO ranges, shell ranges, shell labels, onsite diagonal entries, and reference density.
  3. User can construct existing s-only, sp, and spd systems and observe unchanged `HDIM`, shell indexing, onsite diagonal, and reference density behavior.
  4. Shared shell dimensions are represented as `[0, 1, 3, 5, 7]` wherever basis metadata is consumed.

**Plans**: 1/1 plans executed
Plans:

- [x] 02-01-PLAN.md — Promote Constants/Structure metadata validation to pytest and add f-free simple-format metadata regressions.

### Phase 3: H0/S Routing and f Angular Blocks

**Goal**: DFTorch can assemble finite, symmetric H0 and S matrices for neighbor pairs involving f orbitals.
**Depends on**: Phase 2
**Requirements**: HSK-01, HSK-02, HSK-03, HSK-04, HSK-05, HSK-06, HSK-07, HSK-08
**Success Criteria** (what must be TRUE):

  1. User can build H0/S for systems containing 16-orbital atoms without neighbor pairs being silently dropped.
  2. User can still build H0/S for 1-, 4-, and 9-orbital atoms with the existing H/X/Y routing behavior preserved.
  3. User can exercise s-f, p-f, d-f, and f-f angular Slater-Koster blocks through targeted matrix or formula tests.
  4. User can verify f-containing H0 and S matrices are finite, correctly shaped, and symmetric after assembly.
  5. Direction, atom-order, and AO-order conventions are covered by tests that fail on sign, ordering, or channel-selection drift.

**Plans**: TBD

### Phase 4: SCF and Reference Simulation Validation

**Goal**: Supported f-orbital single-system calculations reach SCF/reference validation with explicit blockers for unsupported modes.
**Depends on**: Phase 3
**Requirements**: SIM-01, SIM-02, SIM-03, SIM-04, SIM-05
**Success Criteria** (what must be TRUE):

  1. User can run a minimal f-containing system through H0/S construction without shape or routing failures.
  2. User can run a supported f-containing system through closed-shell SCF/energy calculation, or receive an explicit unsupported-mode error for a remaining blocker.
  3. Shell-resolved charge, Hubbard, and Coulomb data include f-shell dimensions wherever the supported SCF path requires them.
  4. Reference-paper cases have documented geometry, parameter files, observables, units, and tolerances.
  5. Supported f-orbital reference simulations reproduce selected reference observables within agreed tolerances.

**Plans**: TBD

### Phase 5: Regression Safety and Support Policy Cleanup

**Goal**: The working f-orbital prototype is supportable, regression-safe, and explicit about deferred f-mode coverage.
**Depends on**: Phase 4
**Requirements**: REG-01, REG-02, REG-03, REG-04, CLN-01, CLN-02, CLN-03, CLN-04, CLN-05
**Success Criteria** (what must be TRUE):

  1. User can run existing import, IO, neighbor-list, SCF, force smoke, and tutorial-style simple-format checks with unchanged supported behavior.
  2. Existing simple-format users can keep public API names and workflows unchanged.
  3. Unsupported f combinations fail with explicit errors instead of silent zero blocks or malformed matrices.
  4. Channel lookup, AO ordering, temporary branches, and remaining `1/4/9` assumptions are documented or centralized enough for human troubleshooting.
  5. Batch, force, stress, MD, SEDACS, and ML-SK f-orbital status is explicitly documented as supported, deferred, or unsupported while preserving the validated prototype.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. SKF Canonicalization and Spline Validation | 1/1 | Complete    | 2026-07-21 |
| 2. Constants and Structure Basis Metadata | 1/1 | Complete    | 2026-07-27 |
| 3. H0/S Routing and f Angular Blocks | 0/TBD | Not started | - |
| 4. SCF and Reference Simulation Validation | 0/TBD | Not started | - |
| 5. Regression Safety and Support Policy Cleanup | 0/TBD | Not started | - |
