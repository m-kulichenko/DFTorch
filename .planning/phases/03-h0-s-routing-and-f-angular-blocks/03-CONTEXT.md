# Phase 3: H0/S Routing and f Angular Blocks - Context

**Gathered:** 2026-07-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 3 delivers the first single-system f-orbital H0/S assembly path. It should route neighbor pairs involving 16-orbital atoms, implement paper-derived f angular Slater-Koster blocks, and prove finite symmetric H0/S matrices for f-containing systems while preserving existing 1-, 4-, and 9-orbital behavior.

This phase should not run full SCF/reference-paper simulation validation. Batch f routing, gradients, forces, stress, MD, SEDACS, and broad performance cleanup are deferred unless a small explicit guard is needed to avoid silent misuse.

</domain>

<decisions>
## Implementation Decisions

### Prototype Scope
- **D-01:** Implement the single-system path first. Do not include `StructureBatch`, `ESDriverBatch`, or batched f-orbital H0/S assembly as Phase 3 deliverables.
- **D-02:** Preserve existing H/X/Y behavior for 1-, 4-, and 9-orbital atoms while adding 16-orbital f-aware routing. Existing simple-format tutorial and smoke behavior must remain numerically stable.

### Formula Source and Convention
- **D-03:** Use the Sharma f-electron Slater-Koster paper as the canonical formula source. The likely source found during discussion is “Slater-Koster tables for f electrons,” Journal of Physics C: Solid State Physics 13(4), DOI `10.1088/0022-3719/13/4/016`.
- **D-04:** Before planning implementation details, verify that the paper's cubic harmonic ordering and sign convention match the current `Structure.py` f AO labels: `fx3`, `fy3`, `fz3`, `fx_y2_z2`, `fy_z2_x2`, `fz_x2_y2`, `fxyz`.
- **D-05:** If the Sharma convention differs from the current AO ordering, planning must make the mismatch explicit and choose the smallest correct adaptation. Do not silently transpose/relabel f blocks.

### Testing and Completion Bar
- **D-06:** Phase 3 is done when a minimal Eu-containing single-system H0/S construction succeeds, produces finite matrices, preserves symmetry, and keeps existing simple-format smoke tests passing.
- **D-07:** Include targeted angular tests for selected directions such as x-axis, y-axis, and z-axis.
- **D-08:** Include selected H0/S block-entry checks against hand-calculated values from the paper formulas, not only shape/finite/symmetry checks.

### Deferred Work
- **D-09:** Defer gradients, forces, stress, MD, and batch f support to later phases. If those paths are reachable with f-containing systems before support exists, Phase 3 may add explicit unsupported errors or documentation so they do not fail silently.
- **D-10:** Full SCF/reference-paper result reproduction belongs to Phase 4 after H0/S assembly is runnable.

### the agent's Discretion

The implementation may use the simplest auditable prototype structure that fits the existing codebase. Prefer clear formulas and tests over premature abstraction. Cleanup can follow after the single-system f path is proven.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Defines the overall f-orbital support milestone, prototype priority, and out-of-scope areas.
- `.planning/REQUIREMENTS.md` — Phase 3 requirements `HSK-01` through `HSK-08`, plus deferred simulation and regression requirements.
- `.planning/ROADMAP.md` — Phase 3 goal, acceptance criteria, and dependency on Phase 2.

### Prior Phase Contracts
- `.planning/phases/01-skf-canonicalization-and-spline-validation/01-01-SUMMARY.md` — Parser/channel normalization contract that H0/S must consume.
- `.planning/phases/02-constants-and-structure-basis-metadata/02-01-SUMMARY.md` — Locked `Constants`, `Structure`, and `StructureBatch` metadata contract; Phase 3 should consume the single-structure subset.
- `.planning/phases/02-constants-and-structure-basis-metadata/02-CONTEXT.md` — Boundary that kept H0/S out of Phase 2 and prepared metadata for Phase 3.

### Runtime Code
- `src/dftorch/_bond_integral.py` — Canonical 40-channel order and spline coefficient tensors consumed by H0/S.
- `src/dftorch/Constants.py` — Registers `coeffs_tensor`, `pair_lookup`, `n_orb`, shell metadata, and f fields.
- `src/dftorch/Structure.py` — Defines the current atom-major AO order and f labels that must be verified against Sharma before formula implementation.
- `src/dftorch/_h0ands.py` — Builds single-system H0/S, currently routes by `const.n_orb` masks for 1-, 4-, and 9-orbital pairs.
- `src/dftorch/_slater_koster_pair.py` — Existing Slater-Koster angular block implementation; Phase 3 likely extends this for s-f, p-f, d-f, and f-f blocks.
- `src/dftorch/ESDriver.py` — Calls H0/S assembly from the single-system driver.
- `src/dftorch/script.py` and `tests/test_f_orbital_skf.py` — Existing f-orbital validation patterns to reuse for targeted H0/S and angular tests.

### External Literature
- Sharma, “Slater-Koster tables for f electrons,” Journal of Physics C: Solid State Physics 13(4), DOI `10.1088/0022-3719/13/4/016` — Canonical f angular formula source to verify before planning. If the user provides the PDF, prefer the attached/local copy.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/dftorch/_bond_integral.py` exposes 40 normalized channels, including f-only channels, through `coeffs_tensor`.
- `src/dftorch/Structure.py` exposes atom-major AO starts and fixed local f labels for single structures.
- `src/dftorch/script.py` already creates synthetic XYZ files and direct module-loading validation helpers that can be extended for single-system H0/S tests.
- `tests/test_f_orbital_skf.py` already wraps the standalone validation harness under pytest while restoring module state.

### Established Patterns
- H0/S routing currently uses `const.n_orb[TYPE[neighbor_I/J]]` masks in `_h0ands.py`.
- Existing non-f behavior distinguishes 1-orbital, 4-orbital, and 9-orbital pair routes; Phase 3 should add 16-orbital routes without altering those existing masks.
- Existing Slater-Koster assembly writes flattened AO blocks with `index_add_`, then `_h0ands.py` reshapes and symmetrizes H0/S.
- Tests should use real SKF fixtures where possible and synthetic XYZ only to drive species/geometry setup.

### Integration Points
- Single-system entry: `ESDriver.forward()` calls `H0_and_S_vectorized()` in `src/dftorch/_h0ands.py`.
- Angular block entry: `H0_and_S_vectorized()` calls `Slater_Koster_Pair_SKF_vectorized()` in `src/dftorch/_slater_koster_pair.py`.
- Data source: `const.coeffs_tensor` is indexed by pair type, distance interval, channel, and cubic coefficient.
- AO placement: `structure.H_INDEX_START`, `structure.H_INDEX_END`, and `structure.diagonal` define the single-system matrix layout.

</code_context>

<specifics>
## Specific Ideas

- Start with a minimal Eu-containing single-system H0/S construction test before attempting broader SCF or batch coverage.
- Add angular tests for simple directions where formulas collapse cleanly, especially x-axis, y-axis, and z-axis.
- Compare selected H0/S block entries against hand-calculated Sharma formula values so formula signs and ordering are tested, not only matrix shape.
- Keep the prototype easy to troubleshoot; prioritize explicit formulas and clear tests over a generalized angular-momentum engine.

</specifics>

<deferred>
## Deferred Ideas

- Batch f-orbital H0/S support.
- f-orbital gradients and derivative validation.
- Forces, stress, MD, SEDACS, ML-SK, and performance optimization.
- Full SCF/reference-paper simulation reproduction.

</deferred>

---

*Phase: 3-H0/S Routing and f Angular Blocks*
*Context gathered: 2026-07-27*
