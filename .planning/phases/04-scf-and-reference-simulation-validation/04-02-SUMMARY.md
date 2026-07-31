---
phase: 04-scf-and-reference-simulation-validation
plan: 02
subsystem: numerics
tags: [pytorch, dtype, precision, pme, ewald, neighbor-list, tdd, pytest]

# Dependency graph
requires: []
provides:
  - "A green tests/test_scf.py — the SCF/energy smoke gate is usable again as a regression signal"
  - "calculate_PME_kspace_stress works at the project's float64 default instead of raising in the einsum"
  - "_min_image_sort_key: a testable, dtype-correct min-image composite sort key"
  - "tests/test_dtype_contract.py — precision contract pins for both sites"
affects: [04-03, 04-04, 04-05, "phase 05 REG-01 regression work", "any periodic run with min_image_only and >4200 atoms"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Constructed tensors take their dtype from an operand's .dtype at runtime, never from a fixed-width literal"
    - "Index arithmetic that is added to physical quantities is accumulated at the physical quantity's precision"
    - "Extract-then-fix: a behavior-preserving extraction commit makes a latent precision defect observable at a test boundary before the fix lands"

key-files:
  created:
    - tests/test_dtype_contract.py
  modified:
    - src/dftorch/ewald_pme/PME_torch.py
    - src/dftorch/_nearestneighborlist.py

key-decisions:
  - "The PME mask dtype is tied to E_G.dtype (the first einsum operand), following the in-file eye/dtype=box.dtype precedent, rather than to torch.get_default_dtype()"
  - "The min-image sort key was extracted in a separate behavior-preserving commit so the precision collision could be observed as a real test failure rather than an ImportError"
  - "04-RESEARCH.md Finding 3's 'BENIGN' verdict on the neighbor-list site is overturned: it is a precision hazard, not a mismatch hazard"

patterns-established:
  - "Dtype-contract tests live in tests/test_dtype_contract.py and reuse test_f_orbital_skf.run_with_float64"
  - "A contract-pin test whose passing does NOT prove a defect fixed must say so in its own docstring and name the authoritative gate"

requirements-completed: [SIM-02]

coverage:
  - id: D1
    description: "The PME reciprocal-space stress mask is built in the dtype of the tensors it is contracted against, so the SCF/energy smoke gate runs to completion at float64"
    requirement: SIM-02
    verification:
      - kind: integration
        ref: "tests/test_scf.py::test_energy_smoke_import_and_call[cpu]"
        status: pass
      - kind: unit
        ref: "tests/test_dtype_contract.py#test_pme_kspace_stress_returns_box_dtype"
        status: pass
    human_judgment: false
  - id: D2
    description: "The min-image sort key stays injective over (i, j) at realistic system sizes, so the dedup keeps the nearest periodic image"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "tests/test_dtype_contract.py#test_min_image_sort_key_separates_distinct_pairs_at_scale"
        status: pass
      - kind: unit
        ref: "tests/test_dtype_contract.py#test_min_image_sort_key_orders_by_i_then_j_then_distance"
        status: pass
      - kind: unit
        ref: "tests/test_dtype_contract.py#test_min_image_sort_key_dtype_follows_distances"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every float-cast site in src/dftorch/ has an explicit recorded verdict"
    requirement: SIM-02
    verification:
      - kind: manual
        ref: "This SUMMARY § Complete dtype audit table"
        status: pass
    human_judgment: false
  - id: D4
    description: "No regression in plan 04-01's work or the Phase 3 gate"
    requirement: SIM-02
    verification:
      - kind: unit
        ref: "uv run pytest -q — 39 passed, 0 failed"
        status: pass
    human_judgment: false

# Metrics
duration: 7min
completed: 2026-07-29
status: complete
---

# Phase 4 Plan 02: PME dtype Fix and the D-18 Audit Summary

**`tests/test_scf.py` is green for the first time in this milestone — the PME reciprocal-space stress mask no longer forces 32-bit into a float64 einsum — and the second, latent defect the audit turned up (a min-image sort key that silently merges distinct neighbor pairs above ~4200 atoms) is extracted, tested and fixed.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-29T19:03:43Z
- **Completed:** 2026-07-29T19:10:10Z
- **Tasks:** 2 (4 commits)
- **Files modified:** 3 (1 created, 2 modified)

## The observed pre-fix failures (required by the plan's `<output>`)

**1. `tests/test_scf.py`, before Task 1.** `uv run pytest tests/test_scf.py -x -q --tb=line` reported, verbatim:

```
C:\...\torch\functional.py:383: RuntimeError: expected scalar type Double but found Float
FAILED tests/test_scf.py::test_energy_smoke_import_and_call[cpu] - RuntimeErr...
```

The plan's premise held exactly: `torch/functional.py:383` is `torch.einsum`, reached from `PME_torch.py:264`. After the fix the same command reports `1 passed`.

**2. `test_min_image_sort_key_separates_distinct_pairs_at_scale`, before Task 2's fix.** With the helper extracted but still accumulating in 32 bits:

```
AssertionError: sort key collided for distinct (i, j) pairs:
  3635495168.0 == 3635495168.0 — the min-image dedup would drop a real
  neighbor at this system size
assert tensor(3.6355e+09) != tensor(3.6355e+09)
```

The two inputs differ only in `j` (1000 vs. 1001), which is worth `d2_max = 101.0` in the key. At 3.64e9 the 32-bit spacing is 256, so both round to the identical value. After the fix the same test passes and the two keys differ by exactly 101.0.

Note the ordering of evidence here. Before the extraction commit the three sort-key tests failed with `ImportError: cannot import name '_min_image_sort_key'`, which proves nothing about precision. The behavior-preserving extraction (`67773a8`) exists specifically so the failure above is a *measurement of the defect* rather than a measurement of a missing symbol — and so it stays reproducible from git history.

## Complete dtype audit table (D-18)

Sweep executed at execution time over `src/dftorch/` for `.float()`, `.double()`, `.half()`, `torch.float32`, `dtype=torch.float`, `torch.FloatTensor`, `np.float32`, and `.astype(...)`. **Every site found is listed with a verdict; none is omitted.**

### Float-width casts — the class D-18 targets

| # | File:line | Expression | Verdict | Disposition |
|---|---|---|---|---|
| 1 | `ewald_pme/PME_torch.py:261` | `g_mask = (m_2 > 0).float()` | **REAL BUG** — unconditionally float32 while `E_G` and `metric` are float64; the einsum on line 264 raises. This is the `tests/test_scf.py` failure. | **FIXED** in `da0574d` → `(m_2 > 0).to(dtype=E_G.dtype)` |
| 2 | `_energy.py:12` | `if dtype == torch.float32:` inside `_entropy_eps` | **CORRECT — leave alone.** A dtype *comparison* selecting an entropy epsilon, not a cast. Removing it would degrade float32 runs. | No change |
| 3 | `_nearestneighborlist.py:178` | `ri.float() * (N * d2_max) + j_all.float() * d2_max + d2_all` | **LATENT PRECISION HAZARD.** Not a mismatch (the trailing `+ d2_all` promotes the result to float64), so it never raises — the index terms are simply *rounded before* the distance is added, destroying the key's injectivity over `(i, j)`. Collides once `n_atoms**2 * d2_max` passes ~1.7e9, i.e. above roughly 4200 atoms. The dedup at lines 184-186 then keeps the wrong periodic image, silently. | **FIXED** in `6ed3cd7` → extracted to `_min_image_sort_key`, index terms cast with `.to(dtype=d2_all.dtype)` |

These are the only three float-width sites in `src/dftorch/`, matching the plan's table exactly — the execution-time sweep found no additional ones.

**Overturned prior verdict.** `04-RESEARCH.md` Finding 3 recorded site 3 as *"BENIGN — indices for sorting; dtype mismatch not possible"*. The premise is right and the conclusion is wrong: a mismatch is indeed impossible here, and that is precisely why the bug is dangerous — it produces wrong neighbor lists instead of an exception. The plan's correction is upheld, and the SUMMARY records it so the research finding is not cited later as clearance.

### Non-float casts surfaced by the same sweep

The sweep's `.astype(...)` pattern also matched integer/bool conversions. They are not in D-18's class, but the plan requires a verdict for every site found rather than a silent skip:

| # | File:line(s) | Expression | Verdict |
|---|---|---|---|
| 4 | `sedacs/sedacs_interface.py:266, 315, 319, 323` | `.astype(np.int64)` | **CORRECT.** Widening integer graph/partition indices to int64. No precision loss. |
| 5 | `sedacs/sedacs_interface.py:373` | `.astype(bool)` | **CORRECT.** Adjacency mask to bool. |
| 6 | `sedacs/sedacs_interface.py:390, 391, 397` | `.astype(prev_graph.dtype)` | **CORRECT — and exemplary.** Already takes its dtype from an operand at runtime, which is the pattern this plan is enforcing elsewhere. |
| 7 | `_io.py:494, 626` | `SPECIES = COORDINATES[:, :, 0].astype(int)` | **CORRECT.** Atomic species are small integers read from a coordinate array; `int` is platform int64 here and the values are < 120. |
| 8 | `ESDriver.py:405, 1736` | `structure.TYPE.cpu().numpy().astype(int)` | **CORRECT.** Same — atomic numbers to int for the D3 interface. |

### Deliberately not touched

`init_PME_data` (`PME_torch.py:129-148`) builds its grid with `torch.fft.fftfreq(g)` at the *default* dtype and then `.to(box.dtype)`. That is a default-dtype dependence rather than a fixed-width cast, and it is already reconciled to `box.dtype` one line later, so it is outside D-18's class and outside this plan's `files_modified`. Recorded here rather than changed.

## Task Commits

1. **Task 1: PME reciprocal-space mask dtype (D-17)**
   - `da0574d` (fix) — `(m_2 > 0).to(dtype=E_G.dtype)`, plus a comment stating why the dtype is tied to the operand, so a future edit does not reintroduce a fixed-width cast
2. **Task 2: min-image sort-key precision hazard (D-18, TDD)**
   - `42ea658` (test) — RED: 4 tests, 3 failing
   - `67773a8` (refactor) — behavior-preserving extraction; turns the RED into a genuine precision measurement
   - `6ed3cd7` (fix) — GREEN: `.to(dtype=d2_all.dtype)`

**TDD gate compliance:** Task 2 shows `test(42ea658)` → `refactor(67773a8)` → `fix(6ed3cd7)` in git log order. The RED commit was verified failing before the fix was authored, and the intermediate refactor is byte-equivalent in behavior to the pre-existing inline expression. Task 1 was not a TDD task (the plan marks only Task 2 `tdd="true"`); its red→green evidence is the `tests/test_scf.py` run recorded above.

## Verification results

| Gate | Result |
|---|---|
| `uv run pytest tests/test_scf.py -q` | **1 passed** (was RED) |
| `uv run pytest tests/test_dtype_contract.py -q` | **4 passed** |
| `uv run pytest -q` (whole suite) | **39 passed, 0 failed** in 27.1 s |
| `tests/test_f_orbital_skf.py` (Phase 3 gate) + `test_single_shot_energy.py` + `test_spin_guard.py` + `test_scf.py` | **27 passed** — plan 04-01's 10 tests and Phase 3's 16 all still green |

The suite went from 34 passed / 1 failed at the end of plan 04-01 to 39 passed / 0 failed: the one pre-existing failure is gone and 4 new tests were added.

## Files Created/Modified

- `src/dftorch/ewald_pme/PME_torch.py` — one expression plus its comment. `git diff` confirms the einsum on line 264, the division by `V` and the symmetrisation on line 265 are byte-identical.
- `src/dftorch/_nearestneighborlist.py` — added module-level `_min_image_sort_key` (above `_pair_lookup_for_const`) with a docstring explaining the injectivity requirement and the ~4200-atom collision threshold; replaced the inline expression with a call to it. `git diff` confirms `torch.argsort(..., stable=True)` and every line from there on are unchanged.
- `tests/test_dtype_contract.py` — new, 4 tests. Reuses `test_f_orbital_skf.run_with_float64` for the float64 default and the dftorch module-reset, and sets the same `TORCHDYNAMO_DISABLE` guards `tests/test_scf.py` uses.

## Decisions Made

- **The PME mask takes `E_G.dtype`, not `box.dtype` and not `torch.get_default_dtype()`.** `E_G` is the first einsum operand and is derived from `grid_multip`, which is itself built from `box`, so the three agree in practice — but binding to the actual operand is what makes the contract locally checkable at the call site. The plan specified this and it was followed exactly.
- **The neighbor-list fix casts to `d2_all.dtype` rather than hardcoding `torch.float64`.** A caller running the whole pipeline in float32 gets a key that is still consistent with its own distances; hardcoding float64 would have been a second fixed-width assumption in place of the first.
- **The extraction was committed separately from the fix.** This is the load-bearing methodological choice of the plan. Had the helper been extracted and widened in one commit, the only recorded "RED" would have been an `ImportError`, which is indistinguishable from a typo and proves nothing about precision. The intermediate commit is behavior-identical to its parent, so it introduces no risk, and it leaves the collision reproducible from history.
- **The `.astype(...)` hits were audited and recorded rather than filtered out of the report.** They are not D-18's class, but the plan's acceptance criterion says no site is listed without a verdict, and the honest reading is that a site the sweep surfaced needs a verdict even when the verdict is "not this class of bug".

## Deviations from Plan

None — plan executed exactly as written.

One elaboration, required by the plan's own text rather than added scope: the plan's Task 2 action says *"Confirm that `test_min_image_sort_key_separates_distinct_pairs_at_scale` is red and the other three are green, then implement."* That state does not exist while the helper is absent — all three sort-key tests fail with `ImportError`. Producing the state the plan describes therefore required the behavior-preserving extraction commit `67773a8` between the test and fix commits. This is how the plan's stated observation was actually obtained, not a change to what it asked for.

## Issues Encountered

**The pre-existing CH4 SCF non-convergence is still present and was deliberately not touched.** `tests/test_scf.py` still prints `Did not converge` after 25 iterations, with the residual *growing* over the last iterations (`Res = 0.155 → 0.389 → 0.466`). The test does not assert convergence, so it passes. D-13 governs this and CONTEXT.md marks it pre-decided rather than Phase 4 implementation work, so no auto-fix attempts were spent on it. Recorded here because a growing residual is a stronger signal than a merely-slow one, and whoever owns the SCF phase should see it.

**No blast radius for the neighbor-list fix in the current test suite.** The min-image dedup path lives in the alchemi (`nvalchemiops`) backend, which is not installed here, and `min_image_only` additionally auto-disables when the box exceeds `2*Rcut`. The fix is therefore covered by the new unit tests at the helper boundary rather than end-to-end. This is honest coverage, not a gap the plan overlooked — the helper extraction is precisely what made a testable boundary exist.

## Threat Flags

None. Threat **T-04-03** (DoS at large `n_atoms`) is unchanged: the widened key is the same single tensor of the same length, and float64 vs float32 does not change allocation shape or asymptotic cost. Threat **T-04-04** (silently wrong dedup from key collisions) was the defect fixed, and its `mitigate` disposition is now asserted by `test_min_image_sort_key_separates_distinct_pairs_at_scale`. Threat **T-04-SC** (package installs) stayed inactive — no packages were installed and no dependency was added.

## Known Stubs

None. Both changes are complete implementations; no placeholder values, empty returns, TODO markers or skipped tests were introduced.

## User Setup Required

None.

## Next Phase Readiness

- **`tests/test_scf.py` is now a live regression signal.** That was D-17's entire purpose: from here on, an f-related break introduced by plans 04-03/04-04/04-05 shows up instead of hiding behind a pre-existing red.
- **D-18 is closed.** Every float-cast site in `src/dftorch/` has a recorded verdict, and the two genuine defects are fixed. Phase 5 retains formal REG-01 ownership but inherits no outstanding dtype work from this sweep.
- **Carried forward, unchanged by this plan:** f derivatives still raise, batched f H0/S routing still raises, and the `_bond_integral` shared-`R_orb` concern from Phase 3 remains open. No f-orbital code path was touched here.

## Self-Check: PASSED

All 3 claimed files exist on disk (`src/dftorch/ewald_pme/PME_torch.py`, `src/dftorch/_nearestneighborlist.py`, `tests/test_dtype_contract.py`) and all 4 claimed commit hashes (`da0574d`, `42ea658`, `67773a8`, `6ed3cd7`) resolve in git log.

---
*Phase: 04-scf-and-reference-simulation-validation*
*Completed: 2026-07-29*
