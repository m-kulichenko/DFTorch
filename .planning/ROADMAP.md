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
- [x] **Phase 3: H0/S Routing and f Angular Blocks** - H0/S assembly routes 16-orbital atom pairs and writes tested f-containing Slater-Koster blocks. (completed 2026-07-28)
- [x] **Phase 4: SCF and Reference Simulation Validation** - f-containing single-system simulations reach the supported single-shot energy path and the Eu-N binding minimum is validated against a documented loose-band reference. (completed 2026-07-29)
- [x] **Phase 5: Regression Safety and Support Policy Cleanup** - Existing simple-format behavior, unsupported f modes, and prototype cleanup are locked behind tests and explicit policy.
- [x] **Phase 6: Self-Consistent SCF for f Systems** - f systems converge a real self-consistent charge loop, unlocking the shell-resolved Coulomb f angular blocks. (completed 2026-08-07)
- [ ] **Phase 7: f Angular Derivatives** - dH0/dS become correct and non-zero in f blocks, retiring the blanket derivative refusal.
- [ ] **Phase 8: f Forces and Stress** - Forces and stress for f systems are produced and validated against finite differences.
- [ ] **Phase 8.1: Batched f H0/S Routing** *(INSERTED 2026-08-02)* - Batched H0/S routes 16-orbital atoms instead of refusing, unblocking MD.
- [ ] **Phase 9: f Molecular Dynamics** - f systems run MDXL trajectories with validated forces.

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

**Plans**: 1/1 plans executed

- [x] 03-01-PLAN.md

### Phase 4: SCF and Reference Simulation Validation

**Goal**: Supported f-orbital single-system calculations reach SCF/reference validation with explicit blockers for unsupported modes.
**Depends on**: Phase 3
**Requirements**: SIM-01, SIM-02, SIM-03, SIM-04, SIM-05
**Success Criteria** (what must be TRUE):

  1. User can run a minimal f-containing system through H0/S construction without shape or routing failures.
  2. User can run a supported f-containing system through closed-shell SCF/energy calculation, or receive an explicit unsupported-mode error for a remaining blocker.
  3. Shell-resolved charge, Hubbard, and Coulomb data include f-shell dimensions wherever the supported SCF path requires them.
  4. The isolated Eu-N diatomic validation case has documented geometry, parameter files, observables, units, and tolerances.
  5. The supported f-orbital Eu-N energy scan locates a minimum within the agreed loose sanity band (~10-20%) of the known Eu-N separation.

**Plans**: 5/5 plans executed
Plans:
**Wave 1**

- [x] 04-01-PLAN.md — Single-shot non-SCC energy path for f systems, plus the spin-polarization guard (SIM-01, SIM-02).
- [x] 04-02-PLAN.md — PME float32/float64 repair and the codebase-wide dtype audit (D-17, D-18).

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 04-03-PLAN.md — Validate f dimensions in shell-resolved Hubbard/charge data, gate the Coulomb Hubbard U on MAGNETIC_HUBBARD_LDEP, refuse silent f zeros (SIM-03).

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 04-04-PLAN.md — Document the Eu-N diatomic case and ship the energy-scan minimum gate (SIM-04, SIM-05).

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 04-05-PLAN.md — Human read of the Eu-N binding curve, applying the D-24 asymmetric failure rule (SIM-05).

### Phase 5: Regression Safety and Support Policy Cleanup

**Goal**: The working f-orbital prototype is supportable, regression-safe, and explicit about deferred f-mode coverage.
**Depends on**: Phase 4
**Requirements**: REG-01, REG-02, REG-03, REG-04, REG-05, REG-06, CLN-01, CLN-02, CLN-03, CLN-04, CLN-05
**Success Criteria** (what must be TRUE):

  1. User can run existing import, IO, neighbor-list, SCF, force smoke, and tutorial-style simple-format checks with unchanged supported behavior.
  2. Existing simple-format users can keep public API names and workflows unchanged.
  3. Unsupported f combinations fail with explicit errors instead of silent zero blocks or malformed matrices.
  4. Channel lookup, AO ordering, temporary branches, and remaining `1/4/9` assumptions are documented or centralized enough for human troubleshooting.
  5. Batch, force, stress, MD, SEDACS, and ML-SK f-orbital status is explicitly documented as supported, deferred, or unsupported while preserving the validated prototype.
  6. An `SKFPATH` mixing radial grid steps either interpolates correctly per pair, or refuses loudly — never silently evaluates one pair's distances against another pair's grid.

**Plans**: 8/8 plans executed

Plans:
**Wave 1**

- [x] 05-01-PLAN.md — Characterize today's knot arithmetic, then route each pair through its own radial grid row (D-01, REG-06)
- [x] 05-02-PLAN.md — Settle REG-02 by developer decision and pin the simple-format baseline before any code changes
- [x] 05-03-PLAN.md — Port `script.py`'s independent-oracle SKF checks to pytest, then delete the file (D-03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 05-04-PLAN.md — Refuse mixed-grid-step SKF paths, disposition every remaining grid consumer, pin the public import surface (REG-05, REG-03)
- [x] 05-05-PLAN.md — Classify every library print and gate the status ones behind a noisy-by-default flag; delete `ConstantsTest` (D-02)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 05-06-PLAN.md — Sweep, inventory and disposition every hardcoded orbital-count site (D-04, CLN-03, REG-04)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 05-07-PLAN.md — Basis metadata map, prototype-branch record, f support status matrix, requirement ledger (CLN-01, CLN-02, CLN-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 05-08-PLAN.md — Human sign-off on the phase records and the unchanged-behaviour claim

### Phase 6: Self-Consistent SCF for f Systems

**Goal**: f-containing systems reach a converged self-consistent charge solution, and the shell-resolved Coulomb f angular blocks become implementable and verifiable.
**Depends on**: Phase 5
**Requirements**: SCC-01, SCC-02, SCC-03

> **Scope note (2026-08-02).** User decision 6.1 originally put "fix the CH4 SCF divergence"
> at the head of this phase, conditional on seeing the error first. That is now **discharged**
> — the divergence was diagnosed and fixed during Phase 5 (commit `4dbffaa`). Its root cause
> was not in the SCF machinery at all: the SKF parser gave hydrogen a phantom p shell, making
> the overlap matrix indefinite so the density matrix never conserved electrons. CH4 now
> converges in 6 iterations. **Phase 6 therefore starts from a working f-free SCF baseline**
> and does not need to open with a repair task.
>
> Also settled: the convergence criterion this phase must assert against already exists —
> `ResNorm <= SCF_TOL` **and** `dEc <= SCF_TOL * 100` (`_scf.py:381`, default `SCF_TOL = 1e-6`,
> `SCF_MAX_ITER = 100`). Per decision 6.2(a), the f test asserts that `ResNorm` genuinely
> reaches tolerance rather than the loop exhausting `MaxIt`. Per 6.3(a), closed-shell only —
> spin stays deferred, and the converged number is recorded as not physically complete for
> open-shell Eu 4f7.

> **Scope note (2026-08-04), closing the conditional above.** Written by plan 06-04, which
> owed this update.
>
> **The runaway charge loop was diagnosed.** The cause is the low-rank Krylov convergence
> accelerator (`kernel_update_lr`, engaged once the pass count passes `KRYLOV_START`). It is
> **not** the f orbitals and **not** the coarse per-atom charge description. Evidence: with the
> accelerator disabled and nothing else changed, the Eu-N scan goes from 12 of 21 separations
> converging to **21 of 21**, and at the 12 that converged either way the two runs agree on the
> total energy to better than 1e-6 eV — the accelerator was failing to find the fixed point the
> plain mixer finds reliably, not finding a different one. Corroborated on 16 unrelated mio-1-1
> diatomics at 61 separations each: 61/61 converged in every system with the accelerator off.
>
> **A human ruling of the same date deferred repairing the accelerator to a later phase**, and
> had Phase 6 disable it in the interim for f systems only. Plan 06-01 did that; f-free
> calculations are untouched and a test drives methane through to prove it.
>
> **Decision D-6.06 made the shell-resolved work conditional on that diagnosis. The same ruling
> resolved the condition in favour of doing the work.** The justification changed rather than
> disappearing: the shell-resolved blocks were never on the divergence's code path, so they are
> not a convergence fix — but europium is charged its s shell's electron-repulsion strength of
> 5.71 eV while seven of its nine outer electrons live in the f shell at 13.61 eV, and that is
> a measured 30 percent effect on the dissociation charge. **Success criteria 3 and 4, and
> requirements SCC-02 and SCC-03, are therefore unconditional Phase 6 deliverables after all.**
> Plans 06-02 and 06-03 built them. Nothing above this note is still conditional; do not
> re-open it.
>
> **Correcting the 2026-08-02 note directly above.** "Phase 6 therefore starts from a working
> f-free SCF baseline and does not need to open with a repair task" is accurate about the
> **f-free** methane history it was written about, and misleading about this phase. The f loop
> did need a repair task, and got one: plan 06-01 is that task.

**Success Criteria** (what must be TRUE):

  1. A supported f-containing system runs a real self-consistent charge loop to convergence, not the Phase 4 single-shot path.
  2. Non-convergence warns and returns the last iterate with a convergence flag rather than raising (Phase 4 decision D-13, already pre-decided).
  3. The seven f angular blocks of the shell-resolved Coulomb matrix are implemented and validated, and `FShellResolvedCoulombUnsupportedError` no longer fires for supported systems.
  4. The shell-resolved f charge/Hubbard plumbing validated but unconsumed in Phase 4 (D-14) is consumed by the self-consistent path.
  5. The Phase 4 single-shot path and its pinned reference energy remain available and unbroken.

**Plans**: 5/5 plans executed
Plans:
**Wave 1**

- [x] 06-01-PLAN.md — Tracer: disable the Krylov accelerator for f systems, get Eu-N settling at 2.655 A, and return a real convergence result (iteration count, or -1) from all four SCF loops.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 06-02-PLAN.md — Build the seven missing f angular blocks of the shell-resolved Coulomb matrix, widen the six existing masks that silently skipped f pairs, and retire the refusal across code and both support documents.

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 06-03-PLAN.md — Consume shell-resolved charges inside the closed-shell SCF loop and its energy, so Eu's f electrons are charged at the f Hubbard U rather than the s one.
- [x] 06-04-PLAN.md — Write the verdict on the 10.6 eV single-shot / self-consistent difference, record the Constants.py:232 s-shell Hubbard U defect in place, and close the D-6.06 roadmap tension.

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 06-05-PLAN.md — Recompute the Eu-N binding curves from live code and put the graph in front of a human. Blocking final gate.

### Phase 7: f Angular Derivatives

**Goal**: dH0/dS are correct and non-zero in f blocks, so derivative-consuming paths stop being blanket-refused.
**Depends on**: Phase 6
**Requirements**: DRV-01, DRV-02
**Success Criteria** (what must be TRUE):

  1. f angular derivative formulas are implemented and `F_ANGULAR_DERIVATIVES_AVAILABLE` is `True`.
  2. dH0/dS are non-zero and correctly shaped in s-f, p-f, d-f, and f-f blocks.
  3. Derivative values are checked against finite differences of the Phase 3 angular values, so a sign or ordering error cannot pass.
  4. The `FDerivativeUnsupportedError` guards are removed or narrowed to whatever genuinely remains unsupported, rather than refusing every 16-orbital system.
  5. The derivative source basis is recorded the way Phase 3 source-locked the angular values, so a future reader can trace where each formula came from.

**Plans**: TBD

### Phase 8: f Forces and Stress

**Goal**: Forces and stress for f-containing systems are produced and validated against finite differences.
**Depends on**: Phase 7
**Requirements**: PHY-01, PHY-02
**Success Criteria** (what must be TRUE):

  1. `ESDriver.calc_forces` returns forces for a supported f-containing system instead of raising.
  2. Forces agree with finite differences of the total energy within a documented tolerance.
  3. Stress contributions agree with finite differences or a documented reference within tolerance.
  4. Geometry optimization of an f-containing system runs — the capability Phase 4 explicitly could not use (D-19 forced an energy scan instead).
  5. Existing f-free force and stress behavior is unchanged.

**Plans**: TBD

### Phase 8.1: Batched f H0/S Routing

**INSERTED 2026-08-02.** Split out of Phase 9 by user decision 9.1 in
`.planning/DECISIONS-NEEDED.md` (recorded there as "Phase 9a"; renumbered to 8.1 to match
this roadmap's decimal-insertion convention). It is a substantial capability with its own
validation story — batched H0/S must reproduce single-system H0/S per structure — and
bundling it into Phase 9 would have made that phase two phases wearing one hat, with its
energy-conservation criterion unreachable until the batch work landed.

**Goal**: Batched H0/S assembly routes 16-orbital atoms correctly instead of refusing, so MD can run on f systems.
**Depends on**: Phase 8
**Requirements**: PHY-04
**Success Criteria** (what must be TRUE):

  1. `H0_and_S_vectorized_batch` accepts 16-orbital atoms and `FAngularFormulaSourceError` no longer fires for the batched path.
  2. Batched H0/S reproduces single-system H0/S for each structure in the batch, to bit identity or a stated tolerance.
  3. Batched assembly for f-free systems is unchanged — the existing `n_orb in {1, 4, 9}` masks keep their behavior.
  4. A batch mixing f-containing and f-free structures assembles correctly, with no structure silently dropped.

**Plans**: TBD

### Phase 9: f Molecular Dynamics

**Goal**: f-containing systems run molecular dynamics through `MDXL` with validated forces.
**Depends on**: Phase 8
**Requirements**: PHY-03, PHY-04
**Success Criteria** (what must be TRUE):

  1. An f-containing system runs an `MDXL` trajectory without hitting an unsupported-mode guard.
  2. Energy conservation over the trajectory is within a documented tolerance for the chosen integrator and timestep.
  3. Batched f H0/S routing is implemented and `FAngularFormulaSourceError` no longer fires for it. **This is on the critical path, not conditional** — `MDXL.__init__` and `MDXLBatch.__init__` both take an `ESDriverBatch` (`MD.py:38`, `MD.py:1216`), so MD cannot run on the single-system path. Verified 2026-07-29.
  4. Existing f-free MD behavior is unchanged.

**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 8.1 -> 9

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. SKF Canonicalization and Spline Validation | 1/1 | Complete    | 2026-07-21 |
| 2. Constants and Structure Basis Metadata | 1/1 | Complete    | 2026-07-27 |
| 3. H0/S Routing and f Angular Blocks | 1/1 | Complete    | 2026-07-28 |
| 4. SCF and Reference Simulation Validation | 5/5 | Complete    | 2026-07-29 |
| 5. Regression Safety and Support Policy Cleanup | 8/8 | Complete    | 2026-08-03 |
| 6. Self-Consistent SCF for f Systems | 5/5 | Complete    | 2026-08-07 |
| 7. f Angular Derivatives | 0/TBD | Not started | - |
| 8. f Forces and Stress | 0/TBD | Not started | - |
| 8.1. Batched f H0/S Routing | 0/TBD | Not started | - |
| 9. f Molecular Dynamics | 0/TBD | Not started | - |
